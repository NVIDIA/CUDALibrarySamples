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

// This is an example of a block-level triangular solve (TRSM) performed in a
// single CUDA block using cuBLASDx.
//
// The operation solved is:
//
//   A * X = B
//
// where A is a lower-triangular, non-unit, single-precision matrix of size
// (M x M), and B / X are (M x N) matrices. All matrices use column-major
// layout.
//
// 3D batched global-memory tensors are created on the host from device pointers
// and passed directly to the kernel. The kernel extracts the per-block tile,
// copies A and B into shared memory, calls BLAS{}.execute(), and writes the
// solution X back to global memory.
//
// Correctness is verified against a cuBLAS reference solution.
//
// Performance is measured by timing kernel_repeats launches of the kernel with
// warm-up, following the pattern used in cuBLASDx example 11.

#include <cublasdx/database/trsm/trsm_db.hpp>

#include "../common/trsm_common.hpp"

// -------------------------------------------------------------------------
// trsm_block_kernel
//
// Each block processes BPB consecutive TRSM instances (BPB = BLAS::batches_per_block).
// GlobalTensorA / GlobalTensorB are 3D tensors (rows x cols x padded_batches)
// created on the host and passed by value.
// -------------------------------------------------------------------------
template<class BLAS, class GlobalTensorA, class GlobalTensorB>
__launch_bounds__(BLAS::max_threads_per_block) __global__
    void trsm_block_kernel(GlobalTensorA global_a, GlobalTensorB global_b) {
    extern __shared__ __align__(16) cublasdx::byte smem[];

    using T         = typename BLAS::a_value_type;
    using alignment = cublasdx::alignment_of<BLAS>;

    // Guard: skip blocks beyond the total block count.
    const unsigned total_blocks = cute::size<2>(global_a) / BLAS::batches_per_block;
    if (blockIdx.x >= total_blocks) {
        return;
    }

    // Extract per-block 2D (BPB==1) or 3D (BPB>1) tile from the global tensor.
    auto batch_a = cublasdx::get_batch(global_a, BLAS::get_layout_gmem_a(), blockIdx.x);
    auto batch_b = cublasdx::get_batch(global_b, BLAS::get_layout_gmem_b(), blockIdx.x);

    // Partition shared memory into A and B tensors.
    auto [smem_tensor_a, smem_tensor_b] = cublasdx::shared_memory::slice<T, T>(
        smem, alignment::a, BLAS::get_layout_smem_a(), alignment::b, BLAS::get_layout_smem_b());

    // Load A and B from global memory into shared memory.
    cublasdx::copy<BLAS, alignment::a>(batch_a, smem_tensor_a);
    cublasdx::copy<BLAS, alignment::b>(batch_b, smem_tensor_b);
    cublasdx::copy_wait();

    BLAS {}.execute(smem_tensor_a, smem_tensor_b);
    __syncthreads();

    // Write the solution (B overwritten by X) back to global memory.
    cublasdx::copy<BLAS, alignment::b>(smem_tensor_b, batch_b);
}

// -------------------------------------------------------------------------
// simple_trsm  - the main example function, instantiated per SM architecture
// -------------------------------------------------------------------------
template<unsigned int Arch>
int simple_trsm() {
    // -----------------------------------------------------------------------
    // TRSM problem definition
    // -----------------------------------------------------------------------
    constexpr unsigned M = 64; // rows of B (= size of A for left-side)
    constexpr unsigned N = 4;  // columns of B

    using T             = float;
    constexpr auto Side = cublasdx::side::left;
    constexpr auto Fill = cublasdx::fill_mode::lower;
    constexpr auto Diag = cublasdx::diag::non_unit;
    constexpr auto ArrA = cublasdx::col_major;
    constexpr auto ArrB = cublasdx::col_major;

    constexpr unsigned num_batches = 3855;

    constexpr bool     is_left = (Side == cublasdx::side::left);
    constexpr unsigned dim_a   = is_left ? M : N;

    constexpr unsigned BPB         = 1;
    constexpr unsigned num_threads = 64;

    // -----------------------------------------------------------------------
    // Compose the cuBLASDx TRSM descriptor.
    // -----------------------------------------------------------------------
    using BLAS = decltype(
        cublasdx::Size<M, N>() +
        cublasdx::Precision<T>() +
        cublasdx::Type<cublasdx::type::real>() +
        cublasdx::Function<cublasdx::function::TRSM>() +
        cublasdx::SM<Arch>() +
        cublasdx::Block() +
        cublasdx::BlockDim<num_threads>() +
        cublasdx::Side<Side>() +
        cublasdx::FillMode<Fill>() +
        cublasdx::Diag<Diag>() +
        cublasdx::Arrangement<ArrA, ArrB>() +
        cublasdx::BatchesPerBlock<BPB>());

    // Pad the batch count to a multiple of BPB so every block handles a full set.
    const unsigned padded_batches = cute::round_up(num_batches, BPB);

    // -----------------------------------------------------------------------
    // Host data: A (triangular, dim_a x dim_a) and B (M x N), column-major.
    // -----------------------------------------------------------------------
    const unsigned a_per_batch = dim_a * dim_a;
    const unsigned b_per_batch = M * N;

    auto h_A = example::get_random_uniform_data<T>(padded_batches * a_per_batch, -4.f, 4.f);
    auto h_B = example::get_random_uniform_data<T>(padded_batches * b_per_batch, -1.f, 1.f);

    auto h_tensor_a = example::make_trsm_tensor<T, ArrA>(h_A.data(), dim_a, dim_a, padded_batches);
    example::make_diagonal_dominant(h_tensor_a);

    // Keep an unmodified copy of B for the cuBLAS reference and for restoring
    // device memory before performance timing (TRSM overwrites B in-place).
    const auto h_B_orig = h_B;

    // -----------------------------------------------------------------------
    // Device buffers
    // -----------------------------------------------------------------------
    example::device_vector<T> d_A(h_A);
    example::device_vector<T> d_B(h_B);

    // -----------------------------------------------------------------------
    // Build 3D global-memory tensors on the host and determine kernel types.
    //
    // Shape: (dim_a, dim_a, padded_batches) for A
    //        (M, N, padded_batches)         for B
    // -----------------------------------------------------------------------
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

    auto kernel = trsm_block_kernel<BLAS, global_tensor_a_t, global_tensor_b_t>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    const unsigned num_blocks = padded_batches / BPB;
    const dim3     block_dim(num_threads);

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    auto run_kernel = [&](cudaStream_t s) { kernel<<<num_blocks, block_dim, smem_bytes, s>>>(global_a, global_b); };

    // -----------------------------------------------------------------------
    // Correctness run
    // -----------------------------------------------------------------------
    run_kernel(stream);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Download the result for the first num_batches entries.
    std::vector<T> h_X(num_batches * b_per_batch);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(h_X.data(), d_B.data(), sizeof(T) * num_batches * b_per_batch, cudaMemcpyDeviceToHost));

    static constexpr unsigned kernel_repeats         = 5;
    static constexpr unsigned kernel_warm_up_repeats = 5;

    // Compute cuBLAS reference solution and measure its performance in one call.
    auto [h_B_ref, cublas_time_ms] =
        example::reference_trsm<BLAS>(h_A, h_B_orig, num_batches, kernel_warm_up_repeats, kernel_repeats, stream);

    const double l2_err = example::calculate_error(h_X, h_B_ref);

    // -----------------------------------------------------------------------
    // cuBLASDx performance measurement.
    //
    // Restore d_B to the original input before timing so both warm-up and
    // measurement runs start from the same data.  global_b still points to
    // d_B.data() - no need to recreate the tensor.
    // -----------------------------------------------------------------------
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(d_B.data(), h_B_orig.data(), sizeof(T) * padded_batches * b_per_batch, cudaMemcpyHostToDevice));

    const double time_ms = example::measure::execution(run_kernel, kernel_warm_up_repeats, kernel_repeats, stream);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // TRSM FLOP count: left-side = M²*N, right-side = M*N².
    // same as MAGMA: https://github.com/CEED/MAGMA/blob/master/testing/flops.h
    const double flops_mul_single = (is_left ? (0.5 * (M + 1) * M * N) : (0.5 * M * (N + 1) * N));
    const double flops_add_single = (is_left ? (0.5 * (M - 1) * M * N) : (0.5 * M * (N - 1) * N));
    const double flops            = padded_batches * (flops_mul_single + flops_add_single);

    const double avg_time_ms   = time_ms / kernel_repeats;
    const double cublas_avg_ms = cublas_time_ms / kernel_repeats;

    std::cout << "cuBLASDx TRSM Block Example" << std::endl;
    std::cout << "  M=" << M << "  N=" << N << "  Side=" << (is_left ? "left" : "right")
              << "  Fill=" << (Fill == cublasdx::fill_mode::lower ? "lower" : "upper")
              << "  Diag=" << (Diag == cublasdx::diag::unit ? "unit" : "non_unit") << "  Precision=float" << std::endl;
    std::cout << "  BlockDim=" << num_threads << "  BPB=" << BPB << "  SharedMem=" << smem_bytes << " B" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  cuBLASDx block:" << std::endl;
    std::cout << "    Avg time [ms]: " << avg_time_ms << std::endl;
    std::cout << "    Performance [GFLOPS]: " << flops / (avg_time_ms * 1e6) << std::endl;
    std::cout << "  cuBLAS reference:" << std::endl;
    std::cout << "    Avg time [ms]: " << cublas_avg_ms << std::endl;
    std::cout << "    Performance [GFLOPS]: " << flops / (cublas_avg_ms * 1e6) << std::endl;
    std::cout << " =================================" << std::endl;
    std::cout << "L2 Error: " << l2_err << std::endl;
    std::cout << " =================================" << std::endl;

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    return 0;
}

struct simple_trsm_functor {
    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>, std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return simple_trsm<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(simple_trsm_functor {});
}
