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

// This example demonstrates batched GEMM using the regular cuBLASDx API
// (no pipelining, no K-dimension tiling).
//
//   C[b] = alpha * A[b] * B[b] + beta * C[b]   for b = 0 .. batch_count - 1
//
// Each per-batch matrix multiply has exactly the dimensions of the cuBLASDx
// BLAS descriptor (M x N x K), so a single block computes one full GEMM for
// one batch.  A grid-stride loop lets each block process multiple batches,
// allowing the grid size to be smaller than batch_count for better SM occupancy.
//
// Global memory is represented as rank-3 tensors via
// cublasdx::make_gmem_tensor_batched.  Per-batch 2D slices are extracted
// with cublasdx::get_batch, which produces tensors with static layouts
// compatible with cublasdx::copy.
//
// Compare with batched_gemm_pipeline.cu which uses the pipelining API and
// tiles larger matrices across multiple blocks.

#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

// -------------------------------------------------------------------------
// batched_gemm_kernel
//
// Blocks are mapped to batches via a grid-stride loop: each block processes
// batches blockIdx.x, blockIdx.x + gridDim.x, ... until all batches are done.
// This allows launching fewer blocks than batches for better occupancy control.
// -------------------------------------------------------------------------
template<class BLAS, class GlobalTensorA, class GlobalTensorB, class GlobalTensorC>
__launch_bounds__(BLAS::max_threads_per_block) __global__
    void batched_gemm_kernel(typename BLAS::c_value_type const alpha,
                             typename BLAS::c_value_type const beta,
                             GlobalTensorA                     global_a,
                             GlobalTensorB                     global_b,
                             GlobalTensorC                     global_c,
                             unsigned                          batch_count) {
    extern __shared__ __align__(16) cublasdx::byte smem[];

    using alignment = cublasdx::alignment_of<BLAS>;

    // Slice shared memory for A and B only (C lives in registers via accumulator)
    auto [a_shared, b_shared] =
        cublasdx::shared_memory::slice<typename BLAS::a_value_type, typename BLAS::b_value_type>(
            smem, alignment::a, BLAS::suggest_layout_smem_a(), alignment::b, BLAS::suggest_layout_smem_b());

    // Grid-stride loop: each block processes multiple batches
    for (unsigned batch = blockIdx.x; batch < batch_count; batch += gridDim.x) {
        // Extract per-batch 2D slices with static layouts from rank-3 tensors
        auto batch_a = cublasdx::get_batch(global_a, BLAS::get_layout_gmem_a(), batch);
        auto batch_b = cublasdx::get_batch(global_b, BLAS::get_layout_gmem_b(), batch);
        auto batch_c = cublasdx::get_batch(global_c, BLAS::get_layout_gmem_c(), batch);

        // Copy A, B from global to shared memory
        __syncthreads();
        cublasdx::copy<BLAS, alignment::a>(batch_a, a_shared);
        cublasdx::copy<BLAS, alignment::b>(batch_b, b_shared);
        cublasdx::copy_wait();

        // Execute GEMM: accumulator = A * B
        auto accumulator = BLAS().execute(a_shared, b_shared);

        // Epilogue: D = alpha * accumulator + beta * C (C loaded directly from gmem)
        accumulator.axpby(alpha, beta, batch_c);
    }
}

// -------------------------------------------------------------------------
// batched_gemm  - main example function, instantiated per SM arch
// -------------------------------------------------------------------------
template<unsigned int Arch>
int batched_gemm() {

    // Per-batch matrix dimensions (must match the BLAS descriptor exactly)
    constexpr unsigned m = 64;
    constexpr unsigned n = 64;
    constexpr unsigned k = 64;

    // Number of independent GEMMs (batches)
    const unsigned batch_count = 512;

    // GEMM descriptor: one block computes the full M x N x K multiply
    using BLAS =
        decltype(cublasdx::Size<m, n, k>() + cublasdx::Precision<float>() + cublasdx::Type<cublasdx::type::real>() +
                 cublasdx::Function<cublasdx::function::MM>() + cublasdx::Block() + cublasdx::SM<Arch>());

    using value_type = typename example::uniform_value_type_t<BLAS>;

    value_type alpha = value_type(1.1);
    value_type beta  = value_type(1.2);

    // ================================
    // Allocate and fill device memory
    // ================================

    // Per-batch element counts must match the batch stride used by
    // make_gmem_tensor_batched (= rows * cols, tight packing).
    // Do NOT use global_memory_size_of<BLAS> here -- it may include smem padding.
    constexpr auto a_per_batch = m * k;
    constexpr auto b_per_batch = k * n;
    constexpr auto c_per_batch = m * n;

    const size_t a_total = a_per_batch * batch_count;
    const size_t b_total = b_per_batch * batch_count;
    const size_t c_total = c_per_batch * batch_count;

    value_type* d_a = nullptr;
    value_type* d_b = nullptr;
    value_type* d_c = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_a, a_total * sizeof(value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_b, b_total * sizeof(value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_c, c_total * sizeof(value_type)));

    auto host_a = example::get_random_data<value_type>(a_total, 1);
    auto host_b = example::get_random_data<value_type>(b_total, 2);
    auto host_c = example::get_random_data<value_type>(c_total, 3);

    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_a, host_a.data(), a_total * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_b, host_b.data(), b_total * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_c, host_c.data(), c_total * sizeof(value_type), cudaMemcpyHostToDevice));

    // ================================================================
    // Build rank-3 global tensors using make_gmem_tensor_batched
    //   A: (M, K, batch_count)  -- batch stride = M * K
    //   B: (K, N, batch_count)  -- batch stride = K * N
    //   C: (M, N, batch_count)  -- batch stride = M * N
    // ================================================================

    // Arrangements must match the BLAS descriptor defaults: A=row_major, B=col_major, C=col_major
    auto global_a = cublasdx::make_gmem_tensor_batched<cublasdx::row_major>(d_a, m, k, batch_count);
    auto global_b = cublasdx::make_gmem_tensor_batched<cublasdx::col_major>(d_b, k, n, batch_count);
    auto global_c = cublasdx::make_gmem_tensor_batched<cublasdx::col_major>(d_c, m, n, batch_count);

    using global_tensor_a_t = decltype(global_a);
    using global_tensor_b_t = decltype(global_b);
    using global_tensor_c_t = decltype(global_c);

    // ================================
    // Launch kernel with grid-stride loop
    //
    // Grid size is capped at the number of SMs so each SM gets one block.
    // The grid-stride loop inside the kernel handles the remaining batches.
    // ================================

    auto shared_memory_size = cublasdx::get_shared_storage_size_ab<BLAS>();

    auto kernel = batched_gemm_kernel<BLAS, global_tensor_a_t, global_tensor_b_t, global_tensor_c_t>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    int num_sms = 0;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    const unsigned grid_size = std::min(batch_count, static_cast<unsigned>(num_sms));

    kernel<<<grid_size, BLAS::block_dim, shared_memory_size>>>(alpha, beta, global_a, global_b, global_c, batch_count);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // ================================
    // Copy results back and verify
    // ================================

    std::vector<value_type> host_c_result(c_total);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_c_result.data(), d_c, c_total * sizeof(value_type), cudaMemcpyDeviceToHost));

    bool all_correct = true;
    for (unsigned b = 0; b < batch_count; ++b) {
        std::vector<value_type> batch_a(host_a.begin() + b * a_per_batch, host_a.begin() + (b + 1) * a_per_batch);
        std::vector<value_type> batch_b(host_b.begin() + b * b_per_batch, host_b.begin() + (b + 1) * b_per_batch);
        std::vector<value_type> batch_c(host_c.begin() + b * c_per_batch, host_c.begin() + (b + 1) * c_per_batch);

        auto reference = example::reference_gemm<BLAS>(alpha, batch_a, batch_b, beta, batch_c);

        std::vector<value_type> batch_result(host_c_result.begin() + b * c_per_batch,
                                             host_c_result.begin() + (b + 1) * c_per_batch);

        if (not example::check_error<BLAS>(batch_result, reference)) {
            std::cerr << "Batch " << b << ": FAILED" << std::endl;
            all_correct = false;
        }
    }

    CUDA_CHECK_AND_EXIT(cudaFree(d_a));
    CUDA_CHECK_AND_EXIT(cudaFree(d_b));
    CUDA_CHECK_AND_EXIT(cudaFree(d_c));

    if (all_correct) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    return 1;
}

struct batched_gemm_functor {
    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>, std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return batched_gemm<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(batched_gemm_functor {});
}
