/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cooperative_groups.h>
#include <cublasdx.hpp>
#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <vector>

#include "../common/cudart.hpp"
#include "../common/error_checking.hpp"
#include "../common/random.hpp"
#include "../common/device_io.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/cublas_reference_geqrf_gels.hpp"
#include "../common/measure.hpp"
#include "../common/numeric.hpp"

// This example implements batched TSQRHR (Tall-Skinny QR with Householder
// Reconstruction) for MxN matrices with M >> N using Hopper+ clusters.
//
// One cluster handles one matrix, and one block in that cluster handles one
// NBxN row tile:
//   A = [A_0; A_1; ...; A_{BPC-1}]   BPC : Blocks per cluster
//
// The kernel works in three stages:
//   1. Leaf QR:
//      each block loads its tile into `a_local`, runs GEQRF, and copies the
//      local NxN R into `r_curr` while keeping the tile in compact QR form.
//   2. Reduction hierarchy:
//      `BPC == 2`: block 0 reduces the two leaf R factors directly in the root panel.
//      `BPC == 4`: leaders 0 and 2 reduce 2N×N subgroup panels, then block 0
//                  reduces the two subgroup R factors in a final 2N×N panel.
//      `BPC == 8`: leaders 0 and 4 reduce 4N×N subgroup panels, then block 0
//                  reduces the two subgroup R factors in a final 2N×N panel.
//      Each block keeps the explicit reduction-Q slices it needs for the final
//      local Q reconstruction.
//   3. Householder reconstruction:
//      each block seeds an NB×N workspace with its final reduction-Q slice in
//      the top N×N block, applies UNMQR with the compact leaf panel to form
//      explicit Q, then copies that result back into `a_local`. Block 0 runs
//      MODIFIED_LU on `panel_nn`, broadcasts the result through DSMEM, and all
//      blocks apply TRSM to convert explicit Q back to compact Householder
//      form. Block 0 also writes the final `tau_out` values and top-tile R.
//
// Output matches cublasXgeqrfBatched:
//   A_out : upper triangle = R, strict lower = Householder vectors
//   tau   : Householder scalars
//
// Shared-memory roles:
//   `a_local`    : local NBxN tile / explicit Q / compact QR output
//   `tau_leaf`   : leaf Householder scalars
//   `tau_aux`    : reduction τ, later the sign vector S
//   `r_curr`     : current R (leaf-local at first, final root R on block 0)
//   `q_slice_nn` : this block's NxN subgroup reduction-Q slice
//                  (`BPC == 2` uses I because there is no subgroup stage)
//   `q_root_nn`  : this block's NxN root reduction-Q slice
//   `scratch_q`  : unified scratch region sized for the largest reduction panel
//                  or the NB×N UNMQR workspace:
//                    [0:stage_blocks*N²]  staged reduction panel for subgroup/root QR
//                    [0:NB*N-1]           UNMQR input (top N rows = Q slice) and output
//                    [N²:2N²]             panel_nn alias (NxN modified-LU workspace)
//
// Fast path:
//   when `BPC == 1`, the kernel skips TSQRHR reconstruction and falls back to
//   a single compact GEQRF, so only `a_local` and `tau_leaf` are allocated.
//
// Limitations of this implementation (based on Algorithm 9 of the reference paper):
//    M must be a multiple of N * BPC. NB = M / BPC
//    i.e., M % NB == 0 and NB % N == 0
//
// Reference: G. Ballard et al., "Reconstructing Householder Vectors from
//            Tall-Skinny QR", Journal of Parallel and Distributed Computing, Volume 85, 2015


inline unsigned read_requested_bpc() {
    constexpr unsigned default_bpc = 4;
    const char*        env         = std::getenv("CUSOLVERDX_CGA_BPC");
    if (env == nullptr) {
        return default_bpc;
    }
    const unsigned long parsed = std::strtoul(env, nullptr, 10);
    return static_cast<unsigned>(parsed);
}

constexpr unsigned example_m = 512;
constexpr unsigned example_n = 32;

template<class GEQRF, unsigned NB, unsigned N, unsigned BPC, class T>
constexpr size_t kernel_smem_bytes() {
    if constexpr (BPC == 1) {
        return GEQRF::shared_memory_size;
    } else {
        constexpr unsigned reduction_blocks = (BPC >= 8) ? 4u : 2u;
        constexpr unsigned q_buffer_count   = 2u;
        constexpr unsigned scratch_q_elems  = (NB > (reduction_blocks * N)) ? NB * N : reduction_blocks * N * N;
        return sizeof(T) * (NB * N +          // local tile / explicit Q / compact QR
                            2 * N +           // tau_leaf and tau_red / S
                            N * N +           // current R for this block / final root R
                            q_buffer_count * N * N + // reduction-Q slices
                            scratch_q_elems); // scratch_q: stacked root panel or UNMQR output
    }
}

template<cusolverdx::arrangement Arrange>
__host__ __device__ inline constexpr unsigned leading_dimension(unsigned rows, unsigned cols) {
    return (Arrange == cusolverdx::col_major) ? rows : cols;
}

template<cusolverdx::arrangement Arrange>
__host__ __device__ inline constexpr unsigned matrix_offset(unsigned row, unsigned col, unsigned ld) {
    return (Arrange == cusolverdx::col_major) ? (row + col * ld) : (row * ld + col);
}

template<cusolverdx::arrangement Arrange>
__host__ __device__ inline std::pair<unsigned, unsigned> linear_index_to_coords(unsigned idx,
                                                                                 unsigned rows,
                                                                                 unsigned cols) {
    if constexpr (Arrange == cusolverdx::col_major) {
        return {idx % rows, idx / rows};
    } else {
        return {idx / cols, idx % cols};
    }
}

template<unsigned NT, cusolverdx::arrangement SrcArrange, cusolverdx::arrangement DstArrange, class T>
inline __device__ void copy_R_upper_triangle_nxn(const T* src,
                                                 unsigned ld_src,
                                                 T*       dst,
                                                 unsigned ld_dst,
                                                 unsigned n) {
    for (unsigned k = threadIdx.x; k < n * n; k += NT) {
        const auto [i, j]      = linear_index_to_coords<DstArrange>(k, n, n);
        const unsigned dst_idx = matrix_offset<DstArrange>(i, j, ld_dst);
        dst[dst_idx]           = (i <= j) ? src[matrix_offset<SrcArrange>(i, j, ld_src)] : common::convert<T>(0.0);
    }
}

// Helper to compute tile pointer in global memory
// Tall-skinny layout: multiple row-tiles along one matrix column (col_tile is always 0 here).
template<unsigned NB, cusolverdx::arrangement Arrange, class T>
inline __device__ T* tile(T* A, unsigned lda, unsigned row_tile) {
    if constexpr (Arrange == cusolverdx::col_major) {
        return A + row_tile * NB;
    } else {
        return A + row_tile * NB * lda;
    }
}

template<unsigned NB, cusolverdx::arrangement Arrange, class T>
inline __device__ const T* tile(const T* A, unsigned lda, unsigned row_tile) {
    if constexpr (Arrange == cusolverdx::col_major) {
        return A + row_tile * NB;
    } else {
        return A + row_tile * NB * lda;
    }
}

// Placeholder type for unused solver template parameters (BPC==1 path).
struct NullOp {};

template<class GEQRF_LEAF,
         class UNMQR_LEAF,
         class GEQRF_STAGE,
         class UNGQR_STAGE,
         class GEQRF_ROOT,
         class UNGQR_ROOT,
         class MODLU_TOP,
         class TRSM_TILE,
         class GEMM_COMBINE,
         unsigned NB,
         unsigned N,
         unsigned NT,
         unsigned BPC,
         class T = typename GEQRF_LEAF::a_data_type>
__global__ __launch_bounds__(NT) void cga_tsqrhr_kernel(const T* __restrict__ A,
                                                        unsigned lda,
                                                        T* __restrict__ A_out,
                                                        T* __restrict__ tau_out,
                                                        unsigned batches) {
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(GEQRF_LEAF);

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
    (void)A;  // avoid unused parameter warning
    (void)lda;
    (void)A_out;
    (void)tau_out;
    (void)batches;
    return;
#else
    namespace cg = cooperative_groups;

    constexpr unsigned nn           = N * N;
    constexpr auto     Arrange      = GEQRF_LEAF::a_arrangement;
    constexpr unsigned a_ld         = leading_dimension<Arrange>(NB, N);
    constexpr unsigned root_ld      = leading_dimension<Arrange>(2 * N, N);
    constexpr unsigned stage_blocks = (BPC >= 8) ? 4u : 2u;
    constexpr unsigned stage_ld     = leading_dimension<Arrange>(stage_blocks * N, N);

    cg::cluster_group cluster  = cg::this_cluster();
    const unsigned    batch_id = blockIdx.x / BPC;
    const unsigned    block_id = cluster.block_rank();

    if (batch_id >= batches) {
        return;
    }

    const unsigned one_batch_elems = Arrange == cusolverdx::col_major ? lda * N : NB * BPC * lda;
    const T*       A_batch         = A + batch_id * one_batch_elems;
    T* const       A_out_batch     = A_out + batch_id * one_batch_elems;

    extern __shared__ __align__(16) cusolverdx::byte shared_mem[];
    cusolverdx::byte*                                dsmem[BPC];
    for (unsigned i = 0; i < BPC; ++i) {
        dsmem[i] = cluster.map_shared_rank(shared_mem, i);
    }

    T* a_local         = nullptr;
    T* tau_leaf        = nullptr;
    T* tau_aux         = nullptr;
    T* r_curr          = nullptr;
    T* q_slice_nn      = nullptr;
    T* q_root_nn       = nullptr;
    T* scratch_q       = nullptr;
    T* panel_nn        = nullptr;

    size_t r_curr_offset  = 0;
    size_t q_slice_offset = 0;
    size_t q_root_offset  = 0;
    size_t scratch_offset = 0;
    size_t panel_offset   = 0;
    size_t tau_red_offset = 0;

    // Per-block dynamic shared memory (sizes must match kernel_smem_bytes()).
    if constexpr (BPC == 1) {
        auto [smem_a_local, smem_tau_leaf] =
            cusolverdx::shared_memory::slice<T, T>(shared_mem,
                                                   alignof(T),
                                                   NB * N, // a_local: NB×N tile / compact QR output
                                                   alignof(T),
                                                   N); // tau_leaf: leaf τ
        a_local  = smem_a_local;
        tau_leaf = smem_tau_leaf;
    } else {
        constexpr unsigned scratch_q_elems = (NB > (stage_blocks * N)) ? NB * N : stage_blocks * N * N;
        auto [smem_a_local, smem_tau_leaf, smem_tau_aux, smem_r_curr, smem_q_slice_nn, smem_q_root_nn, smem_scratch_q] =
            cusolverdx::shared_memory::slice<T, T, T, T, T, T, T>(
                shared_mem,
                alignof(T), NB * N,          // a_local: NB×N tile, then explicit Q, then compact QR
                alignof(T), N,               // tau_leaf: leaf τ
                alignof(T), N,               // tau_aux: reduction τ, later sign vector S
                alignof(T), N * N,           // r_curr: current R for this block / final root R
                alignof(T), N * N,           // q_slice_nn: stage-1 reduction-Q slice
                alignof(T), N * N,           // q_root_nn: root reduction-Q slice
                alignof(T), scratch_q_elems); // scratch_q: stacked root panel or UNMQR output

        a_local         = smem_a_local;
        tau_leaf        = smem_tau_leaf;
        tau_aux         = smem_tau_aux;
        r_curr          = smem_r_curr;
        q_slice_nn      = smem_q_slice_nn;
        q_root_nn       = smem_q_root_nn;
        scratch_q       = smem_scratch_q;
        panel_nn        = scratch_q + nn;
    }
    if constexpr (BPC > 1) {
        r_curr_offset  = reinterpret_cast<cusolverdx::byte*>(r_curr) - shared_mem;
        scratch_offset = reinterpret_cast<cusolverdx::byte*>(scratch_q) - shared_mem;
        panel_offset   = reinterpret_cast<cusolverdx::byte*>(panel_nn) - shared_mem;
        tau_red_offset = reinterpret_cast<cusolverdx::byte*>(tau_aux) - shared_mem;
        q_slice_offset = reinterpret_cast<cusolverdx::byte*>(q_slice_nn) - shared_mem;
        q_root_offset  = reinterpret_cast<cusolverdx::byte*>(q_root_nn) - shared_mem;
    }
    T* const tau_red = tau_aux;

    // Load the local NB x N tile for this block using the selected layout helper.
    cusolverdx::copy_2d<NT, NB, N, Arrange>(tile<NB, Arrange>(A_batch, lda, block_id), lda, a_local, a_ld);
    __syncthreads();

    GEQRF_LEAF().execute(a_local, a_ld, tau_leaf);

    if constexpr (BPC == 1) {
        for (unsigned j = threadIdx.x; j < N; j += NT) {
            tau_out[batch_id * N + j] = tau_leaf[j];
        }
    } else { // CGA path
        copy_R_upper_triangle_nxn<NT, Arrange, Arrange>(a_local, a_ld, r_curr, N, N);

        cluster.sync();

        if constexpr (BPC > 1) {
            constexpr unsigned stage_group_size = stage_blocks;
            constexpr unsigned second_leader    = BPC / 2;
            const unsigned     group_base       = (block_id / stage_group_size) * stage_group_size;
            const unsigned     group_leader     = group_base;

            if constexpr (BPC == 2) {
                for (unsigned k = threadIdx.x; k < nn; k += NT) {
                    const auto [i, j] = linear_index_to_coords<Arrange>(k, N, N);
                    q_slice_nn[matrix_offset<Arrange>(i, j, N)] =
                        (i == j) ? common::convert<T>(1.0) : common::convert<T>(0.0);
                }
                __syncthreads();
            } else {
                // Stage 1: explicit subgroup reductions using either 2N×N (BPC=4)
                // or 4N×N (BPC=8) panels.
                if (block_id == group_leader) {
                    for (unsigned src = 0; src < stage_group_size; ++src) {
                        const unsigned src_block = group_base + src;
                        const T* remote_r        = reinterpret_cast<const T*>(dsmem[src_block] + r_curr_offset);
                        for (unsigned k = threadIdx.x; k < nn; k += NT) {
                            const auto [i, j] = linear_index_to_coords<Arrange>(k, N, N);
                            scratch_q[matrix_offset<Arrange>(src * N + i, j, stage_ld)] =
                                remote_r[matrix_offset<Arrange>(i, j, N)];
                        }
                    }
                    __syncthreads();

                    GEQRF_STAGE().execute(scratch_q, stage_ld, tau_red);
                    copy_R_upper_triangle_nxn<NT, Arrange, Arrange>(scratch_q, stage_ld, r_curr, N, N);
                    UNGQR_STAGE().execute(scratch_q, stage_ld, tau_red);

                    for (unsigned dst = 0; dst < stage_group_size; ++dst) {
                        T* q_dst = reinterpret_cast<T*>(dsmem[group_base + dst] + q_slice_offset);
                        cusolverdx::copy_2d<NT, N, N, Arrange>(scratch_q + dst * N, stage_ld, q_dst, N);
                    }
                }
                cluster.sync();
            }

            // Stage 2: block 0 combines the two subgroup R factors with a 2N×N QR.
            if (block_id == 0) {
                const unsigned subgroup_leaders[2] = {0u, second_leader};
                for (unsigned src = 0; src < 2; ++src) {
                    const T* remote_r = reinterpret_cast<const T*>(
                        dsmem[(BPC == 2) ? src : subgroup_leaders[src]] + r_curr_offset);
                    for (unsigned k = threadIdx.x; k < nn; k += NT) {
                        const auto [i, j] = linear_index_to_coords<Arrange>(k, N, N);
                        scratch_q[matrix_offset<Arrange>(src * N + i, j, root_ld)] =
                            remote_r[matrix_offset<Arrange>(i, j, N)];
                    }
                }
                __syncthreads();

                GEQRF_ROOT().execute(scratch_q, root_ld, tau_red);
                copy_R_upper_triangle_nxn<NT, Arrange, Arrange>(scratch_q, root_ld, r_curr, N, N);
                UNGQR_ROOT().execute(scratch_q, root_ld, tau_red);
            }
            cluster.sync();

            {
                const T* root_q = reinterpret_cast<const T*>(dsmem[0] + scratch_offset);
                const unsigned root_child = (block_id >= second_leader) ? 1u : 0u;
                cusolverdx::copy_2d<NT, N, N, Arrange>(root_q + root_child * N, root_ld, q_root_nn, N);
            }
            cluster.sync();

            // Combine the subgroup and root Q slices directly into the top N×N block
            // of the NB×N UNMQR workspace layout.
            GEMM_COMBINE().execute(common::convert<T>(1.0),
                                   q_slice_nn,
                                   q_root_nn,
                                   common::convert<T>(0.0),
                                   scratch_q);
            __syncthreads();

            for (unsigned i = threadIdx.x + N; i < NB; i += NT) {
                for (unsigned j = 0; j < N; ++j) {
                    scratch_q[matrix_offset<Arrange>(i, j, a_ld)] = common::convert<T>(0.0);
                }
            }
            __syncthreads();
        }

        UNMQR_LEAF().execute(a_local, a_ld, tau_leaf, scratch_q, a_ld);

        cusolverdx::copy_2d<NT, NB, N, Arrange>(scratch_q, a_ld, a_local, a_ld);
        __syncthreads();

        // Algorithm 9, Step 3: Modifed-LU(Q)
        // 1. Top-panel_nn modified LU on block 0 only
        if (block_id == 0) {
            cusolverdx::copy_2d<NT, N, N, Arrange>(a_local, a_ld, panel_nn, N);
            __syncthreads();

            MODLU_TOP().execute(panel_nn, N, tau_red);
        }
        cluster.sync();

        // 2. Broadcast U locally and solve tile * U = Q_full_tile.
        const T* root_panel = reinterpret_cast<const T*>(dsmem[0] + panel_offset);
        const T* root_s     = reinterpret_cast<const T*>(dsmem[0] + tau_red_offset);

        copy_R_upper_triangle_nxn<NT, Arrange, Arrange>(root_panel, N, scratch_q, N, N);
        __syncthreads();

        // 3. All blocks compute Q U^-1 via TRSM
        TRSM_TILE().execute(scratch_q, N, a_local, a_ld);

        cluster.sync();

        // 4. Restore the exact top panel_nn from modified_lu for block 0.
        // TRSM updates a_local in block 0 for the lower N x N, so that part must be restored from panel_nn.
        if (block_id == 0) {
            for (unsigned k = threadIdx.x; k < nn; k += NT) {
                const auto [i, j] = linear_index_to_coords<Arrange>(k, N, N);
                T value;
                if (i > j) {
                    value = panel_nn[matrix_offset<Arrange>(i, j, N)];
                } else {
                    // root block writes R.
                    value = root_s[i] * r_curr[matrix_offset<Arrange>(i, j, N)];
                }
                a_local[matrix_offset<Arrange>(i, j, a_ld)] = value;
            }

            // root block builds tau.
            // Only diag(T) is returned as tau. Since T = -U S Y_1^{-H} and Y_1
            // is unit lower triangular, diag(T) = -diag(U) * diag(S).
            for (unsigned j = threadIdx.x; j < N; j += NT) {
                tau_out[static_cast<unsigned long long>(batch_id) * N + j] =
                    common::convert<T>(-1.0) * panel_nn[matrix_offset<Arrange>(j, j, N)] * root_s[j];
            }
        }
        __syncthreads();
    } // end BPC > 1

        // Write the compact QR tile back using the same layout helper.
        cusolverdx::copy_2d<NT, NB, N, Arrange>(a_local, a_ld, tile<NB, Arrange>(A_out_batch, lda, block_id), lda);
#endif
}

template<int Arch, unsigned BPC, unsigned M, unsigned N = 32>
int cga_tsqrhr_impl() {
    if constexpr (Arch < 900) {
        printf("CGA TSQRHR requires Hopper (SM90+). Arch=%d, example uses BPC=%u, M=%u, N=%u, skipping.\n",
               Arch, BPC, M, N);
        return 0;
    } else {
        using T = float; // Change this to cusolverdx::complex<> for complex mode.
        //using T                    = cusolverdx::complex<double>;
        using precision_type       = common::get_precision_t<T>;
        constexpr auto solver_type = common::is_complex<T>() ? cusolverdx::type::complex : cusolverdx::type::real;
        constexpr auto gemm_type   = common::is_complex<T>() ? cublasdx::type::complex : cublasdx::type::real;

        constexpr auto Arr = cusolverdx::col_major;
        //constexpr auto     Arr     = cusolverdx::row_major; // both col and row major are supported in the example
        constexpr unsigned NB      = M / BPC;
        constexpr unsigned NT      = Arch >= 1000 ? 256 : 128;
        constexpr unsigned batches = 500;
        constexpr unsigned lda     = leading_dimension<Arr>(M, N);

        static_assert(M % BPC == 0, "M must be divisible by BPC");
        static_assert(NB >= N, "Each tile must have at least N rows");
        static_assert(NB % N == 0, "NB must be a multiple of N for strip GEMM");

        using GEQRF_LEAF = decltype(cusolverdx::Function<cusolverdx::function::geqrf>() + cusolverdx::Size<NB, N>() +
                                    cusolverdx::Precision<precision_type>() + cusolverdx::Type<solver_type>() +
                                    cusolverdx::Arrangement<Arr>() + cusolverdx::BatchesPerBlock<1>() + cusolverdx::Block() +
                                    cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());

        using cuda_data_type = typename GEQRF_LEAF::a_cuda_data_type;

        constexpr size_t smem_per_block = kernel_smem_bytes<GEQRF_LEAF, NB, N, BPC, T>();
        printf("Matrix size %u x %u (M x N), BPC=%u, shared memory per block %.2f KB (%zu bytes)\n",
               M,
               N,
               BPC,
               static_cast<double>(smem_per_block) / 1024.0,
               smem_per_block);

        int device = 0;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));

        int max_smem = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

        if (static_cast<int>(smem_per_block) > max_smem) {
            printf(
                "Skipping BPC=%u: requires %zu bytes/block shared memory, device limit is %d.\n", BPC, smem_per_block, max_smem);
            return 0;
        }

        constexpr size_t one_batch_elems = Arr == cusolverdx::col_major ? lda * N : M * lda;
        std::vector<T>   h_A(batches * one_batch_elems);
        common::fillup_random_matrix<T>(Arr == cusolverdx::col_major, M, N, h_A.data(), lda, false, false, -2.0, 2.0, batches);
        const std::vector<T> h_A_orig = h_A;

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        T* d_A     = nullptr;
        T* d_A_out = nullptr;
        T* d_tau   = nullptr;

        CUDA_CHECK_AND_EXIT(cudaMalloc(&d_A, sizeof(T) * batches * one_batch_elems));
        CUDA_CHECK_AND_EXIT(cudaMalloc(&d_A_out, sizeof(T) * batches * one_batch_elems));
        CUDA_CHECK_AND_EXIT(cudaMalloc(&d_tau, sizeof(T) * batches * N));

        CUDA_CHECK_AND_EXIT(
            cudaMemcpyAsync(d_A, h_A.data(), sizeof(T) * batches * one_batch_elems, cudaMemcpyHostToDevice, stream));

        // Common launch + verify logic, parameterised by the kernel function pointer.
        // Captures all host-side state by reference; called once from the BPC==1 or BPC>1 branch.
        auto run_and_verify = [&](auto kernel) -> int {
            CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
            CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_per_block));

            cudaLaunchAttribute attr[1];
            attr[0].id               = cudaLaunchAttributeClusterDimension;
            attr[0].val.clusterDim.x = BPC;
            attr[0].val.clusterDim.y = 1;
            attr[0].val.clusterDim.z = 1;

            cudaLaunchConfig_t cfg = {};
            cfg.gridDim            = dim3(batches * BPC, 1, 1);
            cfg.blockDim           = dim3(NT, 1, 1);
            cfg.dynamicSmemBytes   = smem_per_block;
            cfg.stream             = stream;
            cfg.numAttrs           = 1;
            cfg.attrs              = attr;

            unsigned runtime_lda     = lda;
            unsigned runtime_batches = batches;
            void*    args[]          = {reinterpret_cast<void*>(&d_A),
                                        reinterpret_cast<void*>(&runtime_lda),
                                        reinterpret_cast<void*>(&d_A_out),
                                        reinterpret_cast<void*>(&d_tau),
                                        reinterpret_cast<void*>(&runtime_batches)};

            auto run_kernel = [&](cudaStream_t str) {
                cfg.stream = str;
                CUDA_CHECK_AND_EXIT(cudaLaunchKernelExC(&cfg, reinterpret_cast<const void*>(kernel), args));
                CUDA_CHECK_AND_EXIT(cudaGetLastError());
            };

            auto reset = [&](cudaStream_t str) {
                CUDA_CHECK_AND_EXIT(
                    cudaMemcpyAsync(d_A, h_A.data(), sizeof(T) * batches * one_batch_elems, cudaMemcpyHostToDevice, str));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
            };

            constexpr unsigned int warmup_repeats = 1u;
            constexpr unsigned int repeats        = 5u;
            reset(stream);
            const double ms = common::measure::execution(run_kernel, reset, warmup_repeats, repeats, stream) / repeats;
            const double seconds_per_giga_batch = ms / 1e3 / batches * 1e9;
            const double gb_s                   = (sizeof(T) * one_batch_elems * 2.0) / seconds_per_giga_batch;
            const double gflops                 = common::get_flops_geqrf<T>(M, N) / seconds_per_giga_batch;
            common::print_perf("GEQRF using CGA TSQRHR algorithm", batches, M, N, 1, gflops, gb_s, ms, NT, BPC);

            reset(stream);
            run_kernel(stream);
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            std::vector<T> h_A_out(batches * one_batch_elems);
            std::vector<T> h_tau(batches * N);
            CUDA_CHECK_AND_EXIT(
                cudaMemcpyAsync(h_A_out.data(), d_A_out, sizeof(T) * batches * one_batch_elems, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(h_tau.data(), d_tau, sizeof(T) * batches * N, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            std::vector<T> h_A_ref = h_A_orig;
            std::vector<T> h_tau_ref(batches * N, common::convert<T>(0.0));
            std::vector<T> dummy_b;
            const bool     ref_ok = common::reference_cublas_geqrf_gels<T, cuda_data_type, false, true>(
                h_A_ref, dummy_b, h_tau_ref, M, N, 1, batches, Arr == cusolverdx::col_major);

            if (!ref_ok) {
                printf("cuBLAS reference GEQRF failed\n");
                CUDA_CHECK_AND_EXIT(cudaFree(d_A));
                CUDA_CHECK_AND_EXIT(cudaFree(d_A_out));
                CUDA_CHECK_AND_EXIT(cudaFree(d_tau));
                CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
                return 1;
            }

            const double a_err   = common::check_error<T, T>(h_A_out.data(), h_A_ref.data(), h_A_out.size());
            const double tau_err = common::check_error<T, T>(h_tau.data(), h_tau_ref.data(), h_tau.size());
            printf("CGA TSQRHR (BPC=%u)\n", BPC);
            printf("Compact QR relative error vs cuBLAS GEQRF output: %.3e\n", a_err);
            printf("Tau relative error vs cuBLAS GEQRF output: %.3e\n", tau_err);

            const common::HostCompactQrChecks host_q =
                common::host_compact_qr_checks<Arr, T>(h_A_out, h_tau, h_A_orig, batches, M, N, lda);
            printf("Worst residual over all batches ||Q*R - A||/||A||: %.3e\n", host_q.qr_residual);
            printf("Worst orthogonality over all batches ||Q^T*Q - I||: %.3e\n", host_q.orthogonality);

            const bool ok = common::is_error_acceptable<T>(a_err) && common::is_error_acceptable<T>(tau_err) &&
                            common::is_error_acceptable<T>(host_q.qr_residual) &&
                            common::is_error_acceptable<T>(host_q.orthogonality);
            printf("%s\n", ok ? "Success!" : "FAILURE");

            CUDA_CHECK_AND_EXIT(cudaFree(d_A));
            CUDA_CHECK_AND_EXIT(cudaFree(d_A_out));
            CUDA_CHECK_AND_EXIT(cudaFree(d_tau));
            CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
            return ok ? 0 : 1;
        };

        // BPC==1: only GEQRF_LEAF is needed; pass NullOp for the rest to prevent build error due to shared memory limit
        // BPC>1: define all solver types and pass them to the kernel.
        if constexpr (BPC == 1) {
            return run_and_verify(
                cga_tsqrhr_kernel<GEQRF_LEAF,
                                      NullOp, NullOp, NullOp, NullOp, NullOp, NullOp, NullOp, NullOp,
                                      NB, N, NT, 1, T>);
        } else {
            using UNMQR_LEAF = decltype(cusolverdx::Function<cusolverdx::function::unmqr>() + cusolverdx::Size<NB, N, N>() +
                                        cusolverdx::Precision<precision_type>() + cusolverdx::Type<solver_type>() +
                                        cusolverdx::Side<cusolverdx::side::left>() +
                                        cusolverdx::TransposeMode<cusolverdx::transpose::non_transposed>() +
                                        cusolverdx::Arrangement<Arr, Arr>() + cusolverdx::BatchesPerBlock<1>() + cusolverdx::Block() +
                                        cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());

            constexpr unsigned stage_blocks = (BPC >= 8) ? 4u : 2u;

            using GEQRF_STAGE = std::conditional_t<
                (BPC > 2),
                decltype(cusolverdx::Function<cusolverdx::function::geqrf>() + cusolverdx::Size<stage_blocks * N, N>() +
                         cusolverdx::Precision<precision_type>() + cusolverdx::Type<solver_type>() +
                         cusolverdx::Arrangement<Arr>() + cusolverdx::BatchesPerBlock<1>() + cusolverdx::Block() +
                         cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>()),
                NullOp>;

            using UNGQR_STAGE = std::conditional_t<
                (BPC > 2),
                decltype(cusolverdx::Function<cusolverdx::function::ungqr>() + cusolverdx::Size<stage_blocks * N, N, N>() +
                         cusolverdx::Precision<precision_type>() + cusolverdx::Type<solver_type>() +
                         cusolverdx::Arrangement<Arr>() + cusolverdx::BatchesPerBlock<1>() + cusolverdx::Block() +
                         cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>()),
                NullOp>;

            using GEQRF_ROOT = decltype(cusolverdx::Function<cusolverdx::function::geqrf>() + cusolverdx::Size<2 * N, N>() +
                                       cusolverdx::Precision<precision_type>() + cusolverdx::Type<solver_type>() +
                                       cusolverdx::Arrangement<Arr>() + cusolverdx::BatchesPerBlock<1>() + cusolverdx::Block() +
                                       cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());

            using UNGQR_ROOT = decltype(cusolverdx::Function<cusolverdx::function::ungqr>() + cusolverdx::Size<2 * N, N, N>() +
                                       cusolverdx::Precision<precision_type>() + cusolverdx::Type<solver_type>() +
                                       cusolverdx::Arrangement<Arr>() + cusolverdx::BatchesPerBlock<1>() + cusolverdx::Block() +
                                       cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());

            using MODLU_TOP = decltype(cusolverdx::Function<cusolverdx::function::modified_lu>() + cusolverdx::Size<N, N>() +
                                       cusolverdx::Precision<precision_type>() + cusolverdx::Type<solver_type>() +
                                       cusolverdx::Arrangement<Arr>() + cusolverdx::BatchesPerBlock<1>() + cusolverdx::Block() +
                                       cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());

            using TRSM_TILE =
                decltype(cusolverdx::Function<cusolverdx::function::trsm>() + cusolverdx::Size<NB, N>() +
                         cusolverdx::Precision<precision_type>() + cusolverdx::Type<solver_type>() +
                         cusolverdx::Side<cusolverdx::side::right>() + cusolverdx::FillMode<cusolverdx::fill_mode::upper>() +
                         cusolverdx::Diag<cusolverdx::diag::non_unit>() + cusolverdx::Arrangement<Arr, Arr>() +
                         cusolverdx::BatchesPerBlock<1>() + cusolverdx::Block() + cusolverdx::BlockDim<NT>() +
                         cusolverdx::SM<Arch>());

            using GEMM_Arrange =
                std::conditional_t<Arr == cusolverdx::col_major,
                                   cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major, cublasdx::col_major>,
                                   cublasdx::Arrangement<cublasdx::row_major, cublasdx::row_major, cublasdx::row_major>>;

            constexpr unsigned gemm_c_ld = leading_dimension<Arr>(NB, N);

            using GEMM_COMBINE = std::conditional_t<
                (BPC > 1),
                decltype(cublasdx::Size<N, N, N>() + GEMM_Arrange() + cublasdx::Precision<precision_type>() +
                         cublasdx::Type<gemm_type>() + cublasdx::Function<cublasdx::function::MM>() +
                         cublasdx::LeadingDimension<N, N, gemm_c_ld>() + cublasdx::Block() + cublasdx::BlockDim<NT>() + cublasdx::SM<Arch>()),
                NullOp>;

            return run_and_verify(
                cga_tsqrhr_kernel<GEQRF_LEAF, UNMQR_LEAF, GEQRF_STAGE, UNGQR_STAGE, GEQRF_ROOT, UNGQR_ROOT,
                                      MODLU_TOP, TRSM_TILE, GEMM_COMBINE,
                                      NB, N, NT, BPC, T>);
        }
    }
}

template<int Arch>
int cga_tsqrhr() {
    const unsigned requested_bpc = read_requested_bpc();
    switch (requested_bpc) {
        case 1:
            return cga_tsqrhr_impl<Arch, 1, example_m, example_n>();
        case 2:
            return cga_tsqrhr_impl<Arch, 2, example_m, example_n>();
        case 4:
            return cga_tsqrhr_impl<Arch, 4, example_m, example_n>();
        case 8:
            return cga_tsqrhr_impl<Arch, 8, example_m, example_n>();
        default:
            printf("Unsupported CUSOLVERDX_CGA_BPC=%u. Supported values: 1, 2, 4, 8.\n", requested_bpc);
            return 1;
    }
}

template<int Arch>
struct cga_tsqrhr_functor {
    int operator()() { return cga_tsqrhr<Arch>(); }
};


int main() {
    return common::run_example_with_sm<cga_tsqrhr_functor>();
}
