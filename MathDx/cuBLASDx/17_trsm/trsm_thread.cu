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

// This is an example of a thread-level triangular solve (TRSM) using cuBLASDx.
//
// The operation solved is:
//
//   X * A = B
//
// where A is a lower-triangular, non-unit, double-precision matrix of size
// (N x N), and B / X are (M x N) matrices. All matrices use column-major
// layout. 300 independent instances are solved in parallel.
//
// Unlike the block-level API, the thread-level API assigns one CUDA thread per
// TRSM instance. Each thread reads A and B directly from global memory and
// writes X back. No shared memory is required.
//
// 3D batched global-memory tensors are created on the host from device pointers
// and passed directly to the kernel. The kernel extracts the per-thread tile
// using cublasdx::get_batch() and calls BLAS{}.execute() directly on those
// global-memory tensors.
//
// Correctness is verified against a cuBLAS reference solution.

#include "../common/trsm_common.hpp"

// Number of threads per block for the thread-level kernel.
constexpr unsigned THREADS_PER_BLOCK = 64;

// -------------------------------------------------------------------------
// trsm_thread_kernel
//
// One thread per TRSM instance.  GlobalTensorA / GlobalTensorB are 3D tensors
// (rows x cols x num_batches) created on the host and passed by value.
// Each thread extracts its own 2D tile and calls BLAS{}.execute() directly on
// the global-memory tensors - no shared memory required.
// -------------------------------------------------------------------------
template<class BLAS, class GlobalTensorA, class GlobalTensorB>
__global__ void trsm_thread_kernel(GlobalTensorA global_a, GlobalTensorB global_b) {
    const unsigned thread_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned total_items = cute::size<2>(global_a);
    if (thread_idx >= total_items) {
        return;
    }

    auto batch_a = cublasdx::get_batch(global_a, BLAS::get_layout_gmem_a(), thread_idx);
    auto batch_b = cublasdx::get_batch(global_b, BLAS::get_layout_gmem_b(), thread_idx);

    BLAS {}.execute(batch_a, batch_b);
}

// -------------------------------------------------------------------------
// simple_trsm_thread  - the main example function, instantiated per SM arch
// -------------------------------------------------------------------------
template<unsigned int Arch>
int simple_trsm_thread() {
    // -----------------------------------------------------------------------
    // TRSM problem definition
    // -----------------------------------------------------------------------
    constexpr unsigned M = 10; // rows of B
    constexpr unsigned N = 12; // columns of B (= size of A for right-side)

    using T             = double;
    constexpr auto Side = cublasdx::side::right;
    constexpr auto Fill = cublasdx::fill_mode::lower;
    constexpr auto Diag = cublasdx::diag::non_unit;
    constexpr auto ArrA = cublasdx::col_major;
    constexpr auto ArrB = cublasdx::col_major;

    constexpr unsigned num_batches = 300;

    constexpr bool     is_left = (Side == cublasdx::side::left);
    constexpr unsigned dim_a   = is_left ? M : N;

    // -----------------------------------------------------------------------
    // Compose the cuBLASDx TRSM descriptor (thread-level API).
    // -----------------------------------------------------------------------
    using BLAS = decltype(
        cublasdx::Size<M, N>() +
        cublasdx::Precision<T>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::TRSM>() +
        cublasdx::SM<Arch>() +
        cublasdx::Thread() +
        cublasdx::Side<Side>() +
        cublasdx::FillMode<Fill>() +
        cublasdx::Diag<Diag>() +
        cublasdx::Arrangement<ArrA, ArrB>());

    // -----------------------------------------------------------------------
    // Host data: A (triangular, dim_a x dim_a) and B (M x N), column-major.
    // -----------------------------------------------------------------------
    const unsigned a_per_batch = dim_a * dim_a;
    const unsigned b_per_batch = M * N;

    auto h_A = example::get_random_uniform_data<T>(num_batches * a_per_batch, -4.f, 4.f);
    auto h_B = example::get_random_uniform_data<T>(num_batches * b_per_batch, -1.f, 1.f);

    auto h_tensor_a = example::make_trsm_tensor<T, ArrA>(h_A.data(), dim_a, dim_a, num_batches);
    example::make_diagonal_dominant(h_tensor_a);

    // Keep an unmodified copy of B for the cuBLAS reference.
    const auto h_B_orig = h_B;

    // -----------------------------------------------------------------------
    // Device buffers
    // -----------------------------------------------------------------------
    example::device_vector<T> d_A(h_A);
    example::device_vector<T> d_B(h_B);

    // -----------------------------------------------------------------------
    // Build 3D global-memory tensors on the host and determine kernel types.
    //
    // Shape: (dim_a, dim_a, num_batches) for A
    //        (M, N, num_batches)         for B
    // -----------------------------------------------------------------------
    auto global_a = cublasdx::make_gmem_tensor_batched<ArrA>(d_A.data(), dim_a, dim_a, num_batches);
    auto global_b = cublasdx::make_gmem_tensor_batched<ArrB>(d_B.data(), M, N, num_batches);

    using global_tensor_a_t = decltype(global_a);
    using global_tensor_b_t = decltype(global_b);

    // -----------------------------------------------------------------------
    // Launch: one thread per TRSM instance.
    // -----------------------------------------------------------------------
    const unsigned num_blocks = (num_batches + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    trsm_thread_kernel<BLAS, global_tensor_a_t, global_tensor_b_t>
        <<<num_blocks, THREADS_PER_BLOCK>>>(global_a, global_b);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // -----------------------------------------------------------------------
    // Download results and verify correctness against cuBLAS.
    // -----------------------------------------------------------------------
    std::vector<T> h_X(num_batches * b_per_batch);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(h_X.data(), d_B.data(), sizeof(T) * num_batches * b_per_batch, cudaMemcpyDeviceToHost));

    auto [h_B_ref, cublas_ms] = example::reference_trsm<BLAS>(h_A, h_B_orig, num_batches);

    const double l2_err = example::calculate_error(h_X, h_B_ref);

    std::cout << "cuBLASDx TRSM Thread Example" << std::endl;
    std::cout << "  M=" << M << "  N=" << N << "  Side=" << (is_left ? "left" : "right")
              << "  Fill=" << (Fill == cublasdx::fill_mode::lower ? "lower" : "upper")
              << "  Diag=" << (Diag == cublasdx::diag::unit ? "unit" : "non_unit") << "  Precision=double"
              << "  Batches=" << num_batches << std::endl;
    std::cout << " =================================" << std::endl;
    std::cout << "L2 Error: " << l2_err << std::endl;
    std::cout << " =================================" << std::endl;
    return 0;
}

struct simple_trsm_thread_functor {
    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>, std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return simple_trsm_thread<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(simple_trsm_thread_functor {});
}
