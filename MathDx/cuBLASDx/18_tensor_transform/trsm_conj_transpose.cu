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

// This example demonstrates solving a TRSM problem where the triangular matrix A
// is used in its conjugate-transpose form via cublasdx::conj_transpose_view:
//
//   A^H * X = B
//
// where A is a lower-triangular, non-unit matrix of complex single-precision values
// of size (M x M), and B / X are (M x N) complex matrices.  All matrices use
// column-major layout.
//
// The TRSM descriptor is defined for the un-transposed storage of A (lower-triangular,
// col-major).  At execute time, cublasdx::conj_transpose_view(smem_a) presents A^H to
// the solver without any data copy or transposition in memory.
//
// Correctness is verified against a cuBLAS reference using CUBLAS_OP_C.

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cublas_v2.h>
#include <cublasdx/database/trsm/trsm_db.hpp>

#include "../common/trsm_common.hpp"

// -------------------------------------------------------------------------
// trsm_conj_transpose_kernel
//
// Each block processes one TRSM instance.
// Applies conj_transpose_view to smem_a so the solver operates on A^H.
// -------------------------------------------------------------------------
template<class BLAS, class GlobalTensorA, class GlobalTensorB>
__launch_bounds__(BLAS::max_threads_per_block) __global__
    void trsm_conj_transpose_kernel(GlobalTensorA global_a, GlobalTensorB global_b) {
    extern __shared__ __align__(16) cublasdx::byte smem[];

    using T         = typename BLAS::a_value_type;
    using alignment = cublasdx::alignment_of<BLAS>;

    const unsigned total_blocks = cute::size<2>(global_a) / BLAS::batches_per_block;
    if (blockIdx.x >= total_blocks) {
        return;
    }

    auto batch_a = cublasdx::get_batch(global_a, BLAS::get_layout_gmem_a(), blockIdx.x);
    auto batch_b = cublasdx::get_batch(global_b, BLAS::get_layout_gmem_b(), blockIdx.x);

    auto [smem_tensor_a, smem_tensor_b] = cublasdx::shared_memory::slice<T, T>(
        smem, alignment::a, BLAS::get_layout_smem_a(), alignment::b, BLAS::get_layout_smem_b());

    cublasdx::copy<BLAS, alignment::a>(batch_a, smem_tensor_a);
    cublasdx::copy<BLAS, alignment::b>(batch_b, smem_tensor_b);
    cublasdx::copy_wait();

    // Solve A^H * X = B by passing the conjugate-transpose view of A.
    BLAS {}.execute(cublasdx::conj_transpose_view(smem_tensor_a), smem_tensor_b);
    __syncthreads();

    cublasdx::copy<BLAS, alignment::b>(smem_tensor_b, batch_b);
}

// -------------------------------------------------------------------------
// trsm_conj_transpose  - main example function, instantiated per SM arch
// -------------------------------------------------------------------------
template<unsigned int Arch>
int trsm_conj_transpose() {
    constexpr unsigned M = 32;
    constexpr unsigned N = 8;

    using T                  = cublasdx::complex<float>;
    constexpr auto Side      = cublasdx::side::left;
    constexpr auto StoreFill = cublasdx::fill_mode::lower; // A's storage fill mode
    constexpr auto ViewFill  = cublasdx::fill_mode::upper; // A^H is upper-triangular
    constexpr auto Diag      = cublasdx::diag::non_unit;
    constexpr auto ArrA      = cublasdx::col_major;
    constexpr auto ArrB      = cublasdx::col_major;

    constexpr unsigned num_batches = 512;
    constexpr unsigned dim_a       = M; // left-side: A is M x M

    constexpr unsigned BPB         = 1;
    constexpr unsigned num_threads = 64;

    // -----------------------------------------------------------------------
    // Compose the cuBLASDx TRSM descriptor.
    // The descriptor's FillMode describes the viewed matrix A^H (upper-triangular).
    // The conjugate transpose is applied at execute time via conj_transpose_view.
    // -----------------------------------------------------------------------
    using BLAS = decltype(
        cublasdx::Size<M, N>() +
        cublasdx::Precision<float>() +
        cublasdx::Type<cublasdx::type::complex>() +
        cublasdx::Function<cublasdx::function::TRSM>() +
        cublasdx::SM<Arch>() +
        cublasdx::Block() +
        cublasdx::BlockDim<num_threads>() +
        cublasdx::Side<Side>() +
        cublasdx::FillMode<ViewFill>() +
        cublasdx::Diag<Diag>() +
        cublasdx::Arrangement<ArrA, ArrB>() +
        cublasdx::BatchesPerBlock<BPB>());

    const unsigned padded_batches = cute::round_up(num_batches, BPB);

    // -----------------------------------------------------------------------
    // Host data: random complex matrices.
    // -----------------------------------------------------------------------
    const unsigned a_per_batch = dim_a * dim_a;
    const unsigned b_per_batch = M * N;

    auto gen_complex = [](size_t n, float lo, float hi) {
        std::vector<T>                        v(n);
        std::mt19937                          rng(42);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (auto& x : v) {
            x = T {dist(rng), dist(rng)};
        }
        return v;
    };

    auto h_A = gen_complex(padded_batches * a_per_batch, -4.f, 4.f);
    auto h_B = gen_complex(padded_batches * b_per_batch, -1.f, 1.f);

    // Make A diagonally dominant so the solve is well-conditioned.
    auto h_tensor_a = example::make_trsm_tensor<T, ArrA>(h_A.data(), dim_a, dim_a, padded_batches);
    example::make_diagonal_dominant(h_tensor_a);

    const auto h_B_orig = h_B;

    // -----------------------------------------------------------------------
    // Device buffers and global tensors
    // -----------------------------------------------------------------------
    example::device_vector<T> d_A(h_A);
    example::device_vector<T> d_B(h_B);

    auto global_a = cublasdx::make_gmem_tensor_batched<ArrA>(d_A.data(), dim_a, dim_a, padded_batches);
    auto global_b = cublasdx::make_gmem_tensor_batched<ArrB>(d_B.data(), M, N, padded_batches);

    using global_tensor_a_t = decltype(global_a);
    using global_tensor_b_t = decltype(global_b);

    // -----------------------------------------------------------------------
    // Kernel configuration
    // -----------------------------------------------------------------------
    const unsigned smem_bytes = cublasdx::make_shared_storage_calculator()
                                    .add(cublasdx::alignment_of<BLAS>::a, sizeof(T), BLAS::get_layout_smem_a())
                                    .add(cublasdx::alignment_of<BLAS>::b, sizeof(T), BLAS::get_layout_smem_b())
                                    .get();

    auto kernel = trsm_conj_transpose_kernel<BLAS, global_tensor_a_t, global_tensor_b_t>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    const unsigned num_blocks = padded_batches / BPB;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // -----------------------------------------------------------------------
    // Correctness run
    // -----------------------------------------------------------------------
    kernel<<<num_blocks, num_threads, smem_bytes, stream>>>(global_a, global_b);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<T> h_X(num_batches * b_per_batch);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(h_X.data(), d_B.data(), sizeof(T) * num_batches * b_per_batch, cudaMemcpyDeviceToHost));

    // cuBLAS reference: same storage, but op = CUBLAS_OP_C (conjugate transpose).
    constexpr bool is_col_a = (ArrA == cublasdx::col_major);
    constexpr bool is_col_b = (ArrB == cublasdx::col_major);
    constexpr bool is_left  = (Side == cublasdx::side::left);
    constexpr bool is_lower = (StoreFill == cublasdx::fill_mode::lower);
    constexpr bool is_unit  = (Diag == cublasdx::diag::unit);

    auto [h_B_ref, _] = example::reference_trsm<T>(
        h_A, h_B_orig, M, N, num_batches, is_left, is_lower, is_unit, is_col_a, is_col_b, CUBLAS_OP_C, 0, 0, stream);

    const double l2_err = example::calculate_error(h_X, h_B_ref);

    std::cout << "cuBLASDx TRSM Conjugate-Transpose Example" << std::endl;
    std::cout << "  Solving A^H * X = B" << std::endl;
    std::cout << "  M=" << M << "  N=" << N << "  Precision=complex<float>"
              << "  StoreFill=lower  ViewFill=upper  Diag=non_unit" << std::endl;
    std::cout << " =================================" << std::endl;
    std::cout << "L2 Error: " << std::fixed << std::setprecision(6) << l2_err << std::endl;
    std::cout << " =================================" << std::endl;

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return (l2_err < 0.001) ? 0 : 1;
}

struct trsm_conj_transpose_functor {
    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>, std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return trsm_conj_transpose<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(trsm_conj_transpose_functor {});
}
