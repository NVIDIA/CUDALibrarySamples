/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cooperative_groups.h>

#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>
#include <cublasdx.hpp>

#include "../common/cudart.hpp"
#include "../common/error_checking.hpp"
#include "../common/random.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/device_io.hpp"
#include "../common/measure.hpp"
#include "../common/print.hpp"
#include "../common/cusolver_reference_cholesky.hpp"

/*
An example of Cholesky factorization with blocked algorithm using cooperative group and thread block clusters.
Note that this implementation uses a right-looking algorithm, in contrast to the left-looking algorithm in the "blocked_potrf.cu" example.
Reference: https://www.cs.utexas.edu/~flame/Notes/NotesOnCholReal.pdf

Memory Strategy for Right-looking Blocked Cholesky Algorithm in this example:
- Panel tiles (row k, columns k+1...n-1): distributed in dsmem across blocks
- Diagonal tile (k,k): loaded to dsmem, factored, kept for TRSM
- Trailing tiles (i,j where i,j>k): loaded from global, updated, stored back

There are a few limitations of this implementation:
 * The matrix size, N, must be a multiple of the block size, NB
 * Only upper-triangular storage is implemented
 * Only real-valued types are supported because of limited support of GEMM for transposed of C
*/

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))

// Helper to compute tile pointer in global memory
template<unsigned NB, cusolverdx::arrangement Arrange, class T>
inline __device__ T* tile(T* A, unsigned lda, unsigned i, unsigned j) {
    if (Arrange == cusolverdx::col_major) {
        return A + i * NB + j * NB * lda;
    } else {
        return A + i * NB * lda + j * NB;
    }
}

// Load/store diagonal blocks (only upper triangle for upper-fill mode)
template<unsigned NB, cusolverdx::arrangement Arrange, unsigned NT, class T>
inline __device__ void load_diagonal_block(const T* A, const int lda, T* As, const int ldas) {
    const int tid = threadIdx.x;
    __builtin_assume(tid < NT);
    if constexpr (NT % NB == 0 && NB * NB >= NT) {
        constexpr unsigned stride_jj = NT / NB;
        const unsigned     i         = tid % NB;
        const unsigned     j         = tid / NB;
        for (int jj = 0; jj < NB; jj += stride_jj) {
            bool is_upper_tri = (Arrange == cusolverdx::col_major) ? (i <= j + jj) : (i >= j + jj);
            if (is_upper_tri) {
                As[i + (jj + j) * ldas] = __ldcg(A + i + (jj + j) * lda);
            }
        }
    } else {
        for (int k = tid; k < NB * NB; k += NT) {
            unsigned i            = k % NB;
            unsigned j            = k / NB;
            bool     is_upper_tri = (Arrange == cusolverdx::col_major) ? (i <= j) : (i >= j);
            if (is_upper_tri) {
                As[i + j * ldas] = __ldcg(A + i + j * lda);
            }
        }
    }
    __syncthreads();
}

template<unsigned NB, cusolverdx::arrangement Arrange, unsigned NT, class T>
inline __device__ void store_diagonal_block(const T* As, const int ldas, T* A, const int lda) {
    const int tid = threadIdx.x;
    __builtin_assume(tid < NT);
    __syncthreads();
    if constexpr (NT % NB == 0 && NB * NB >= NT) {
        constexpr unsigned stride_jj = NT / NB;
        const unsigned     i         = tid % NB;
        const unsigned     j         = tid / NB;
        for (int jj = 0; jj < NB; jj += stride_jj) {
            bool is_upper_tri = (Arrange == cusolverdx::col_major) ? (i <= j + jj) : (i >= j + jj);
            if (is_upper_tri) {
                __stcg(A + i + (jj + j) * lda, As[i + (jj + j) * ldas]);
            }
        }
    } else {
        for (int k = tid; k < NB * NB; k += NT) {
            unsigned i            = k % NB;
            unsigned j            = k / NB;
            bool     is_upper_tri = (Arrange == cusolverdx::col_major) ? (i <= j) : (i >= j);
            if (is_upper_tri) {
                __stcg(A + i + j * lda, As[i + j * ldas]);
            }
        }
    }
}

// Load a full NB×NB tile from global using __ldcg (L1-bypass, L2-cache).
// Matches the memory layout of cusolverdx::copy_2d for use with TRSM/GEMM operators.
// For col_major: element (r,c) at ptr + r + c*ld
// For row_major: element (r,c) at ptr + r*ld + c
template<unsigned NB, cusolverdx::arrangement Arrange, unsigned NT, class T>
inline __device__ void load_tile_ldcg(const T* A, const int lda, T* As, const int ldas) {
    const int tid = threadIdx.x;
    __builtin_assume(tid < NT);
    for (int k = tid; k < NB * NB; k += NT) {
        unsigned r = k / NB;
        unsigned c = k % NB;
        if constexpr (Arrange == cusolverdx::col_major) {
            As[r + c * ldas] = __ldcg(A + r + c * lda);
        } else {
            As[r * ldas + c] = __ldcg(A + r * lda + c);
        }
    }
    __syncthreads();
}

// Store a full NB×NB tile to global using __stcg (write via L2, bypass L1).
template<unsigned NB, cusolverdx::arrangement Arrange, unsigned NT, class T>
inline __device__ void store_tile_stcg(const T* As, const int ldas, T* A, const int lda) {
    const int tid = threadIdx.x;
    __builtin_assume(tid < NT);
    __syncthreads();
    for (int k = tid; k < NB * NB; k += NT) {
        unsigned r = k / NB;
        unsigned c = k % NB;
        if constexpr (Arrange == cusolverdx::col_major) {
            __stcg(A + r + c * lda, As[r + c * ldas]);
        } else {
            __stcg(A + r * lda + c, As[r * ldas + c]);
        }
    }
}

//////// Right-looking Blocked Cholesky Using CGA //////////

// Cluster APIs exist only on Hopper+. For __CUDA_ARCH__ < 900 this kernel compiles to a no-op so
// multi-arch builds (e.g. 80+90) succeed; host code with Arch < 900 skips launch and returns 0.
template<class POTRF, class TRSM, class GEMM, unsigned N, unsigned blocks_per_cluster, class T = typename POTRF::a_data_type>
__global__ __launch_bounds__(POTRF::max_threads_per_block) void cga_blocked_potrf_right_looking(T* A, unsigned lda, int* info, unsigned batches) {
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(POTRF);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    namespace cg               = cooperative_groups;
    cg::cluster_group cluster  = cg::this_cluster();
    const unsigned    batch_id = blockIdx.x / blocks_per_cluster;
    if (batch_id >= batches) {
        return;
    }

    A += batch_id * N * lda;
    info += batch_id;

    constexpr unsigned NB               = POTRF::m_size;
    constexpr unsigned lds              = POTRF::lda;
    constexpr unsigned NT               = POTRF::max_threads_per_block;
    constexpr unsigned n_tiles_per_side = N / NB;
    constexpr auto     Arrange          = POTRF::a_arrangement;

    // Shared memory allocation: diagonal tile + panel tiles + work space
    extern __shared__ __align__(sizeof(T)) cusolverdx::byte shared_mem[];

    // Maximum panel size is (n_tiles - 1) when k=0
    constexpr unsigned max_panel_size        = n_tiles_per_side - 1;
    constexpr unsigned panel_tiles_per_block = CEIL_DIV(max_panel_size, blocks_per_cluster);

    auto [diag_tile, panel_tiles, work_tile, info_s] =
        cusolverdx::shared_memory::slice<T, T, T, int>(shared_mem,
                                                       alignof(T), NB * lds, // Diagonal tile storage
                                                       alignof(T), panel_tiles_per_block * NB * lds, // Panel tiles (distributed across CTA blocks)
                                                       alignof(T), NB * lds, // Work space for trailing matrix
                                                       alignof(int), 1); // Info

    int rinfo = 0;

    // Setup distributed shared memory access
    cusolverdx::byte* dsmem[blocks_per_cluster];
    for (int i = 0; i < blocks_per_cluster; ++i) {
        dsmem[i] = cluster.map_shared_rank(shared_mem, i);
    }

    const auto     diag_offset  = reinterpret_cast<cusolverdx::byte*>(diag_tile) - shared_mem;
    const auto     panel_offset = reinterpret_cast<cusolverdx::byte*>(panel_tiles) - shared_mem;
    const unsigned tile_bytes   = NB * lds * sizeof(T);

    // Helper to get diagonal tile in distributed shared memory (owner's copy)
    auto diag_tile_ptr = [&](unsigned owner) -> T* { return reinterpret_cast<T*>(dsmem[owner] + diag_offset); };

    // Helper to get panel tile in distributed shared memory
    auto panel_tile_ptr = [&](unsigned k, unsigned j) -> T* {
        unsigned panel_idx = j - k - 1; // for upper triangular storage, the panel is the columns from k+1 to j
        unsigned owner     = panel_idx % blocks_per_cluster;
        unsigned local_idx = panel_idx / blocks_per_cluster;
        return reinterpret_cast<T*>(dsmem[owner] + panel_offset + local_idx * tile_bytes);
    };


    for (int k = 0; k < n_tiles_per_side; ++k) {
        T* Akk_gmem = tile<NB, Arrange>(A, lda, k, k);

        // Block 0 always owns the diagonal (it also always does the fused SYRK+POTRF).
        // For k=0: standard POTRF (no previous fused path exists yet).
        // For k>0: diagonal was already factored during k-1's GEMM phase by block 0;
        //          diag_tile in block 0's dsmem already holds factored A[k,k].
        //          Skip POTRF and its cluster.sync() entirely.
        if (k == 0) {
            if (cluster.block_rank() == 0) {
                load_diagonal_block<NB, Arrange, NT>(Akk_gmem, lda, diag_tile, lds);
                POTRF().execute(diag_tile, lds, info_s);
                store_diagonal_block<NB, Arrange, NT>(diag_tile, lds, Akk_gmem, lda);
                if (threadIdx.x == 0 && rinfo == 0 && *info_s != 0) {
                    rinfo = *info_s;
                }
            }
            // Sync: all blocks need the factored A[0,0] before TRSM
            cluster.sync();
        }
        // For k>0: the cluster.sync() at the END of iteration k-1's GEMM phase has
        // already synchronized all blocks and made diag_tile(block 0) visible.

        int panel_size = n_tiles_per_side - k - 1;
        if (panel_size == 0)
            continue; // Last tile, no panel to update

        // STEP 2: Broadcast diagonal + load panel tiles in one pass, then TRSM.
        // Non-rank-0 blocks copy diag from block 0's remote dsmem while all blocks
        // simultaneously issue their panel tile loads from global memory.
        if (cluster.block_rank() != 0) {
            T* remote_diag = diag_tile_ptr(0);
            for (int idx = threadIdx.x; idx < NB * lds; idx += NT) {
                diag_tile[idx] = remote_diag[idx];
            }
        }
        for (int j_offset = cluster.block_rank(); j_offset < panel_size; j_offset += blocks_per_cluster) {
            unsigned j = k + 1 + j_offset;
            cusolverdx::copy_2d<NT, NB, NB, Arrange, 1>(tile<NB, Arrange>(A, lda, k, j), lda, panel_tile_ptr(k, j), lds);
        }
        __syncthreads(); // diag_tile ready; all owned panel tiles loaded

        for (int j_offset = cluster.block_rank(); j_offset < panel_size; j_offset += blocks_per_cluster) {
            unsigned j = k + 1 + j_offset;
            TRSM().execute(diag_tile, lds, panel_tile_ptr(k, j), lds);
        }

        // Sync: all panel tiles must be complete before trailing matrix update
        cluster.sync();

        // STEP 3: Trailing matrix update
        // Each CTA block processes columns it owns (Akj is in LOCAL shared memory).
        // Panel writeback is combined here.
        // For j_offset==0 (j=k+1), block 0 additionally runs POTRF after SYRK,
        // preparing diag_tile for the NEXT iteration without a global memory roundtrip.

        for (int j_offset = 0; j_offset < panel_size; ++j_offset) {
            unsigned j = k + 1 + j_offset;

            if (j_offset % blocks_per_cluster == cluster.block_rank()) {
                T* Akj = panel_tile_ptr(k, j); // LOCAL - B operand for all rows in this column

                // Update all rows i = k+1..j
                for (int i = k + 1; i <= (int)j; ++i) {
                    T* Aij_gmem = tile<NB, Arrange>(A, lda, i, j);

                    if (i == (int)j) {
                        // Diagonal SYRK: A_jj -= A_kj^T * A_kj
                        load_diagonal_block<NB, Arrange, NT>(Aij_gmem, lda, work_tile, lds);
                        GEMM().execute(T(-1.0), Akj, Akj, T(1.0), work_tile);

                        if (j_offset == 0) {
                            // j == k+1: POTRF after SYRK.
                            // work_tile now holds the Schur complement of A[k+1,k+1].
                            // Factor it immediately and store result in diag_tile for
                            // the next iteration — skipping a global roundtrip and a cluster.sync().
                            __syncthreads(); // ensure all threads see GEMM output before POTRF
                            POTRF().execute(work_tile, lds, info_s);
                            if (threadIdx.x == 0 && rinfo == 0 && *info_s != 0) {
                                rinfo = *info_s + (k + 1) * NB;
                            }
                            store_diagonal_block<NB, Arrange, NT>(work_tile, lds, Aij_gmem, lda);
                            // Copy factored diagonal into diag_tile (freed since TRSM is done).
                            // The cluster.sync() at end of GEMM phase makes it visible to all blocks.
                            for (int idx = threadIdx.x; idx < NB * lds; idx += NT) {
                                diag_tile[idx] = work_tile[idx];
                            }
                            // No need to sync here because the next iteration will sync before reading diag_tile
                        } else {
                            // Regular SYRK: just persist to global
                            store_diagonal_block<NB, Arrange, NT>(work_tile, lds, Aij_gmem, lda);
                            // store_diagonal_block starts with sync but not ends with one —
                            // sync here to release work_tile before next j_offset reuses it.
                            __syncthreads();
                        }
                    } else {
                        // Off-diagonal GEMM: A_ij -= A_ki^T * A_kj  (Akj LOCAL, Aki may be remote)
                        T* Aki = panel_tile_ptr(k, i); // may be remote dsmem

                        cusolverdx::copy_2d<NT, NB, NB, Arrange, 1>(Aij_gmem, lda, work_tile, lds);
                        __syncthreads();
                        GEMM().execute(T(-1.0), Aki, Akj, T(1.0), work_tile);
                        __syncthreads();
                        cusolverdx::copy_2d<NT, NB, NB, Arrange, 1>(work_tile, lds, Aij_gmem, lda);
                        __syncthreads(); // release work_tile before next iteration reads it
                    }
                }

                // Panel writeback: write updated Akj to global
                T* Akj_gmem = tile<NB, Arrange>(A, lda, k, j);
                cusolverdx::copy_2d<NT, NB, NB, Arrange, 1>(Akj, lds, Akj_gmem, lda);
            }
        }

        // Sync: ensures all trailing updates, panel writebacks, and diag_tile(k+1) write
        // are complete and visible before the next iteration reads them.
        cluster.sync();
    }

    if (cluster.thread_rank() == 0) {
        *info = rinfo;
    }

#else
    (void)A;
    (void)lda;
    (void)info;
    (void)batches;
    return;
#endif
}

template<int Arch, unsigned BPC>
int cga_blocked_potrf_right_looking_impl() {

    if constexpr (Arch < 900) {
        printf("Arch %d < 900. CGA POTRF requires Hopper or later. Exiting with no error.\n", Arch);
        return 0;
    }

    constexpr unsigned N          = 512;
    constexpr unsigned lda        = N;
    constexpr unsigned input_size = lda * N;
    constexpr unsigned batches    = 400;

    constexpr unsigned NB  = 32;
    constexpr unsigned lds = NB;
    constexpr unsigned NT  = 128;
    static_assert(N % NB == 0, "N must be divisible by block size");

    using precision_type   = double;
    constexpr auto type    = cusolverdx::type::real;
    constexpr auto Arrange = cusolverdx::row_major;
    constexpr auto Fill    = cusolverdx::fill_mode::upper;

    // Define operators
    using POTRF = decltype(cusolverdx::Function<cusolverdx::function::potrf>() + cusolverdx::FillMode<Fill>() + cusolverdx::Size<NB>() +
                           cusolverdx::LeadingDimension<lds>() + cusolverdx::Precision<precision_type>() + cusolverdx::Type<type>() +
                           cusolverdx::Arrangement<Arrange>() + cusolverdx::Block() + cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());
    using TRSM  = decltype(cusolverdx::Function<cusolverdx::function::trsm>() + cusolverdx::Size<NB, NB, NB>() + cusolverdx::LeadingDimension<lds, lds>() +
                          cusolverdx::Precision<precision_type>() + cusolverdx::Type<type>() + cusolverdx::Side<cusolverdx::side::left>() +
                          cusolverdx::Diag<cusolverdx::diag::non_unit>() + cusolverdx::TransposeMode<cusolverdx::transpose::transposed>() +
                          cusolverdx::Arrangement<Arrange, Arrange>() + cusolverdx::FillMode<Fill>() + cusolverdx::Block() + cusolverdx::BlockDim<NT>() +
                          cusolverdx::SM<Arch>());
    using T     = typename POTRF::a_data_type;
    using GEMM_Arrange               = std::conditional_t<Arrange == cusolverdx::col_major,
                                            cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::col_major>,
                                            cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>>;
    constexpr unsigned int alignment = ((sizeof(T) * NB * NB) % 16 == 0) ? 16 : sizeof(T);
    using GEMM = decltype(cublasdx::Size<NB, NB, NB>() + GEMM_Arrange() + cublasdx::Alignment<alignment, alignment, alignment>() + cublasdx::Precision<T>() +
                          cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() + cublasdx::LeadingDimension<lds, lds, lds>() +
                          cublasdx::Block() + cublasdx::BlockDim<NT>() + cublasdx::SM<Arch>());

    // Memory allocation
    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<T> A(input_size * batches);
    common::fillup_random_diagonal_dominant_matrix<T>(Arrange == cusolverdx::col_major, N, N, A.data(), lda, false, -2, 4, batches);

    std::vector<T>   L(input_size * batches);
    std::vector<int> info(batches);
    T*               d_A    = nullptr;
    int*             d_info = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    // Launch configuration
    constexpr unsigned n_tiles_per_side = N / NB;
    constexpr unsigned nthreads         = NT;
    constexpr unsigned nblocks          = batches * BPC;

    // Optimized shared memory
    constexpr unsigned max_panel_size        = n_tiles_per_side - 1;
    constexpr unsigned panel_tiles_per_block = CEIL_DIV(max_panel_size, BPC);
    size_t             smem_size             = (1 + panel_tiles_per_block + 1) * NB * lds * sizeof(T) + sizeof(int);

    printf("Optimized CGA Right-Looking POTRF\n");
    printf("N=%d, NB=%d, NT=%d, blocks_per_cluster=%u\n", N, NB, NT, (unsigned)BPC);
    printf("Shared memory per block: %zu bytes\n", smem_size);
    printf("  Diagonal tile: %zu bytes\n", (size_t)NB * lds * sizeof(T));
    printf("  Panel tiles: %d × %zu bytes = %zu bytes\n", panel_tiles_per_block, (size_t)NB * lds * sizeof(T), panel_tiles_per_block * NB * lds * sizeof(T));
    printf("  Work tile: %zu bytes\n", (size_t)NB * lds * sizeof(T));

    cudaLaunchAttribute attribute[1];
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = BPC;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {};
    config.gridDim            = dim3(nblocks, 1, 1);
    config.blockDim           = dim3(nthreads, 1, 1);
    config.dynamicSmemBytes   = smem_size;
    config.numAttrs           = 1;
    config.attrs              = attribute;
    config.stream             = stream;

    auto     kernel      = cga_blocked_potrf_right_looking<POTRF, TRSM, GEMM, N, BPC, T>;
    unsigned lda_arg     = lda;
    unsigned batches_arg = batches;
    void* args[] = {reinterpret_cast<void*>(&d_A), reinterpret_cast<void*>(&lda_arg), reinterpret_cast<void*>(&d_info), reinterpret_cast<void*>(&batches_arg)};

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

    int maxClusterSize;
    CUDA_CHECK_AND_EXIT(cudaOccupancyMaxPotentialClusterSize(&maxClusterSize, kernel, &config));
    printf("Maximum supported cluster size: %d blocks\n", maxClusterSize);

    auto run_kernel = [&](cudaStream_t) {
        CUDA_CHECK_AND_EXIT(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel), args));
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
    };
    auto reset = [&](cudaStream_t str) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice, str));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
    };

    const unsigned int warmup_repeats = 1;
    const unsigned int repeats        = 5;

    reset(stream);
    double ms                     = common::measure::execution(run_kernel, reset, warmup_repeats, repeats, stream) / repeats;
    double seconds_per_giga_batch = ms / 1e3 / batches * 1e9;
    double gb_s                   = input_size * sizeof(T) * 2 / seconds_per_giga_batch;
    double gflops                 = common::get_flops_potrf<T>(N) / seconds_per_giga_batch;

    common::print_perf("CGA POTRF Optimized", batches, N, N, 1, gflops, gb_s, ms, NT);

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(T) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    for (int i = 0; i < info.size(); ++i) {
        if (info[i] != 0) {
            std::printf("batch %d: Cholesky info=%d (0 ok; >0 not positive definite at that leading minor)\n", i, info[i]);
            exit(1);
        }
    }

    // Verify against cuSolver
    std::vector<T> B;
    common::reference_cusolver_cholesky<T, T, false, true>(A, B, info.data(), N, 1, batches, false, Arrange == cusolverdx::col_major, false, batches);

    const auto total_relative_error = common::check_error<T, T>(L.data(), A.data(), A.size());
    std::cout << "Relative forward error vs cuSolver: " << total_relative_error << std::endl;

    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaDeviceReset());

    if (common::is_error_acceptable<T>(total_relative_error)) {
        std::cout << "Success!" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

template<int Arch>
int cga_blocked_potrf_right_looking(unsigned blocks_per_cluster) {
    if constexpr (Arch < 900) {
        printf("Arch %d < 900. CGA POTRF requires Hopper or later. Exiting.\n", Arch);
        printf("Arch %d < 900. CGA POTRF requires Hopper or later. Exiting.\n", Arch);
        return 0;
    }
    switch (blocks_per_cluster) {
        case 1:
            return cga_blocked_potrf_right_looking_impl<Arch, 1>();
        case 2:
            return cga_blocked_potrf_right_looking_impl<Arch, 2>();
        case 4:
            return cga_blocked_potrf_right_looking_impl<Arch, 4>();
        case 8:
            return cga_blocked_potrf_right_looking_impl<Arch, 8>();
        case 16:
            return cga_blocked_potrf_right_looking_impl<Arch, 16>();
        default:
            std::fprintf(stderr, "Invalid blocks_per_cluster=%u (supported: 1, 2, 4, 8, 16; default 4).\n", blocks_per_cluster);
            return 1;
    }
}

namespace {
    unsigned g_blocks_per_cluster = 4;
}

template<int Arch>
struct cga_blocked_potrf_right_looking_functor {
    int operator()() { return cga_blocked_potrf_right_looking<Arch>(g_blocks_per_cluster); }
};

int main(int argc, char** argv) {
    if (argc >= 2) {
        if (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0) {
            std::printf("Usage: %s [blocks_per_cluster]\n"
                        "  blocks_per_cluster: cluster size in blocks — 1, 2, 4, 8, 16 (default: 4)\n",
                        argv[0]);
            return 0;
        }
        char*         end = nullptr;
        unsigned long v   = std::strtoul(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0') {
            std::fprintf(stderr, "Invalid blocks_per_cluster: %s\n", argv[1]);
            return 1;
        }
        if (v != 1UL && v != 2UL && v != 4UL && v != 8UL && v != 16UL) {
            std::fprintf(stderr, "blocks_per_cluster must be 1, 2, 4, 8, or 16 (got %lu)\n", v);
            return 1;
        }
        g_blocks_per_cluster = static_cast<unsigned>(v);
    }
    return common::run_example_with_sm<cga_blocked_potrf_right_looking_functor>();
}
