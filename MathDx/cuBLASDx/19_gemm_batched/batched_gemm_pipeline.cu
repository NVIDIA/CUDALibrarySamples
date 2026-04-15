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

#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

// Batched GEMM kernel using the pipelining API with a grid-stride loop
// over the batch dimension.
//
// Each CUDA block processes one (row_tile, col_tile) spatial position and
// iterates over batches: blockIdx.z, blockIdx.z + gridDim.z, ...
//
//   blockIdx.x = row tile index  (0 .. M/tile_m - 1)
//   blockIdx.y = col tile index  (0 .. N/tile_n - 1)
//   blockIdx.z = initial batch index
//
// A and B are rank-3 tensors (M, K, batch_count) and (K, N, batch_count).
// The pipeline's get_tile() accepts rank-2 coordinates per operand:
//   coord_a = (row_tile, batch)  -- selects the A tile for this block
//   coord_b = (col_tile, batch)  -- selects the B tile for this block
//
// After the first batch, reset_tile() re-points the pipeline at the next
// batch without reallocating barriers or shared memory.
//
// C is passed as a flat pointer. Inside the kernel, cublasdx::make_gmem_tensor
// reconstructs a 2D view for the current batch (offset = batch * m * n),
// then cublasdx::get_tile selects the per-block output tile.
// ArrC is a template parameter so make_gmem_tensor<ArrC> can be called in device code.
template<class BLAS, cublasdx::arrangement ArrC, class Alpha, class Beta, class CValueType, class DevicePipeline>
__launch_bounds__(DevicePipeline::max_threads_per_block, 1) __global__
    void batched_gemm_kernel(Alpha const                            alpha,
                             Beta const                             beta,
                             CValueType*                            c_ptr,
                             unsigned                               m,
                             unsigned                               n,
                             unsigned                               ldc,
                             unsigned                               batch_count,
                             __grid_constant__ DevicePipeline const device_pipeline) {
#ifdef __CUDA_ARCH__
    if constexpr (cublasdx::sm_of_v<BLAS> == __CUDA_ARCH__) {

        extern __shared__ __align__(device_pipeline.buffer_alignment()) cublasdx::byte smem[];

        // First batch: create the tile pipeline
        auto tile_pipeline =
            device_pipeline.get_tile(smem,
                                     cublasdx::make_coord(blockIdx.x, blockIdx.z),  // A: (row_tile, batch)
                                     cublasdx::make_coord(blockIdx.y, blockIdx.z)); // B: (col_tile, batch)

        // Grid-stride loop over the batch dimension
        for (unsigned batch = blockIdx.z; batch < batch_count;) {
            // Build a 2D C view for the current batch
            auto tile_gmem_c = cublasdx::get_tile(cublasdx::make_gmem_tensor<ArrC>(c_ptr + batch * m * n, m, n, ldc),
                                                  BLAS::c_shape,
                                                  blockIdx.x,
                                                  blockIdx.y);

            // Epilogue: D = alpha * A * B + beta * C
            auto epilogue_functor = [&](auto& accumulator) {
                auto d_fragment = accumulator.make_partition_and_copy(tile_gmem_c);
                cublasdx::axpby(alpha, accumulator.get_results(), beta, d_fragment);
                accumulator.partition_and_copy(d_fragment, tile_gmem_c);
            };

            tile_pipeline.execute(epilogue_functor);

            // Re-point the pipeline at the next batch for the next iteration
            batch += gridDim.z;
            device_pipeline.reset_tile(tile_pipeline,
                                       cublasdx::make_coord(blockIdx.x, batch),  // A: (row_tile, batch)
                                       cublasdx::make_coord(blockIdx.y, batch)); // B: (col_tile, batch)
        }
    }
#endif
}

// This is an example of batched GEMM using cuBLASDx pipelining API,
// where batching is represented as the third dimension of global memory tensors.
//
//              C[b] = alpha * A[b] * B[b] + beta * C[b]   for b = 0 .. batch_count - 1
//
// A[b], B[b], C[b] are M x K, K x N, M x N matrices stored contiguously per batch:
//   A: (M, K, batch_count)  -- row-major
//   B: (K, N, batch_count)  -- col-major
//   C: (M, N, batch_count)  -- row-major
//
// The pipelining API handles the batch dimension transparently through the rank-3
// tensor layout. Internally, suggest_device_pipeline() detects rank-3 tensors and
// adjusts the tiler to include a unit batch dimension (tile_m, tile_k, 1) / (tile_k, tile_n, 1).
//
// A grid-stride loop over the batch dimension lets each spatial (row_tile, col_tile)
// block process multiple batches.  After the first batch, reset_tile() re-points the
// pipeline at the next batch without reinitializing barriers or shared memory.
//
// Grid: (M/tile_m, N/tile_n, min(batch_count, num_sms / spatial_tiles))
//   blockIdx.z is the initial batch index; the grid-stride loop handles the rest.
//
// M, N, K must be divisible by tile_m, tile_n, tile_k respectively.
// batch_count can be any positive integer.
template<unsigned int Arch, cublasdx::sm_modifier Modifier>
int batched_gemm_pipeline() {

    // ===================================
    // Configurable Global GEMM properties
    // ===================================

    // Per-batch matrix dimensions
    const unsigned m = 1024; // A: M x K, C: M x N
    const unsigned n = 1024; // B: K x N, C: M x N
    const unsigned k = 512;  // Contraction dimension

    // Number of batches -- can be any positive integer
    const unsigned batch_count = 4;

    // Global data arrangement
    constexpr auto global_arrangement_a = cublasdx::row_major;
    constexpr auto global_arrangement_b = cublasdx::col_major;
    constexpr auto global_arrangement_c = cublasdx::row_major;

    // Leading dimensions (tight packing, no extra padding)
    const unsigned global_lda = (global_arrangement_a == cublasdx::col_major) ? m : k;
    const unsigned global_ldb = (global_arrangement_b == cublasdx::col_major) ? k : n;
    const unsigned global_ldc = (global_arrangement_c == cublasdx::col_major) ? m : n;

    // ======================================
    // Configurable cuBLASDx tile properties
    // ======================================

    using a_compute_precision = __half;
    using b_compute_precision = __half;
    using c_compute_precision = float;

    constexpr auto type = cublasdx::type::real;

    using a_value_type =
        std::conditional_t<type == cublasdx::type::real, a_compute_precision, cublasdx::complex<a_compute_precision>>;
    using b_value_type =
        std::conditional_t<type == cublasdx::type::real, b_compute_precision, cublasdx::complex<b_compute_precision>>;
    using c_value_type =
        std::conditional_t<type == cublasdx::type::real, c_compute_precision, cublasdx::complex<c_compute_precision>>;

    c_value_type alpha = example::make_value<c_value_type>(1.1);
    c_value_type beta  = example::make_value<c_value_type>(1.2);

    constexpr unsigned tile_m       = 128;
    constexpr unsigned tile_n       = 128;
    constexpr unsigned tile_k       = 32;
    constexpr int      tile_threads = 128;

    constexpr auto tile_arr_a = global_arrangement_a;
    constexpr auto tile_arr_b = global_arrangement_b;
    constexpr auto tile_arr_c = global_arrangement_c;

    constexpr unsigned int cublasdx_alignment = 16;

    // If manual_pipeline_depth is 0, the maximum depth for the target architecture is used.
    constexpr unsigned manual_pipeline_depth   = 0;
    constexpr bool     override_pipeline_depth = (manual_pipeline_depth != 0);

    // ================================
    // Validate configuration
    // ================================

    constexpr unsigned stage_shared_req = tile_m * tile_k * sizeof(a_value_type) +
                                          tile_k * tile_n * sizeof(b_value_type) +
                                          sizeof(cublasdx::pipeline_stage_scratch_t);

    constexpr unsigned available_shared_memory = commondx::device_info<Arch>::shared_memory();
    constexpr unsigned maximal_pipeline_depth  = std::min(k / tile_k, std::min(16u, (available_shared_memory - 32u) / stage_shared_req));
    constexpr unsigned pipeline_depth = override_pipeline_depth ? manual_pipeline_depth : maximal_pipeline_depth;

    static_assert(pipeline_depth <= maximal_pipeline_depth,
                  "Chosen pipeline depth requires more shared memory than available on target architecture");

    if (m % tile_m != 0 || n % tile_n != 0 || k % tile_k != 0) {
        std::cerr << "M, N, K must be divisible by tile_m, tile_n, tile_k" << std::endl;
        return 1;
    }

    if (k / tile_k < pipeline_depth) {
        std::cerr << "K stages (" << k / tile_k << ") must be >= pipeline_depth (" << pipeline_depth << ")"
                  << std::endl;
        return 1;
    }

    // ================================
    // Build BLAS descriptor
    // ================================

    // Same descriptor as example 11 -- batching is purely a grid/tensor concern,
    // not part of the per-block BLAS descriptor.
    using BLAS = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                          cublasdx::Precision<a_compute_precision, b_compute_precision, c_compute_precision>() +
                          cublasdx::Type<type>() + cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<tile_arr_a, tile_arr_b, tile_arr_c>() + cublasdx::Block() +
                          cublasdx::BlockDim<tile_threads>() + cublasdx::StaticBlockDim() +
                          cublasdx::Alignment<cublasdx_alignment, cublasdx_alignment, cublasdx_alignment>() +
                          cublasdx::EnableInputStreaming() + cublasdx::WithPipeline() + cublasdx::SM<Arch, Modifier>());

    // ================================
    // Allocate and fill device memory
    // ================================

    const size_t a_total = static_cast<size_t>(m) * k * batch_count;
    const size_t b_total = static_cast<size_t>(k) * n * batch_count;
    const size_t c_total = static_cast<size_t>(m) * n * batch_count;

    a_value_type* d_a = nullptr;
    b_value_type* d_b = nullptr;
    c_value_type* d_c = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_a, a_total * sizeof(a_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_b, b_total * sizeof(b_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&d_c, c_total * sizeof(c_value_type)));

    auto host_a = example::get_random_data<a_value_type>(a_total, 1);
    auto host_b = example::get_random_data<b_value_type>(b_total, 2);
    auto host_c = example::get_random_data<c_value_type>(c_total, 3);

    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_a, host_a.data(), a_total * sizeof(a_value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_b, host_b.data(), b_total * sizeof(b_value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d_c, host_c.data(), c_total * sizeof(c_value_type), cudaMemcpyHostToDevice));

    // ================================================================
    // Build rank-3 global tensors -- batching hidden as 3rd dimension
    //   A: (M, K, batch_count)   row-major, batch stride = M * K
    //   B: (K, N, batch_count)   col-major, batch stride = K * N
    // C is passed as a flat pointer; the kernel reconstructs a per-batch
    // 2D view via cublasdx::make_gmem_tensor (batch stride = M * N).
    // ================================================================

    auto global_a = cublasdx::make_gmem_tensor_batched<global_arrangement_a>(d_a, m, k, batch_count);
    auto global_b = cublasdx::make_gmem_tensor_batched<global_arrangement_b>(d_b, k, n, batch_count);

    // ================================================================
    // Build device pipeline from rank-3 A/B tensors.
    // suggest_device_pipeline detects rank-3 tensors via pipeline_helper::is_3d
    // and adjusts internal tilers to (tile_m, tile_k, 1) / (tile_k, tile_n, 1).
    // ================================================================

    auto opt_device_pipeline = cublasdx::suggest_device_pipeline<pipeline_depth, BLAS>(global_a, global_b);

    if (not opt_device_pipeline) {
        std::cerr << "Pipeline configuration invalid: ensure M/N/K are divisible by tile sizes and K >= pipeline depth"
                  << std::endl;
        return 1;
    }

    auto device_pipeline = opt_device_pipeline.value();

    auto shared_memory_size = cublasdx::make_shared_storage_calculator()
                                  .add(device_pipeline.buffer_alignment(), device_pipeline.buffer_size())
                                  .get();

    // ================================================================
    // Launch kernel with grid-stride loop over the batch dimension
    //
    // Grid: (M/tile_m, N/tile_n, min(batch_count, num_sms / spatial_tiles))
    //   blockIdx.x = row tile index
    //   blockIdx.y = col tile index
    //   blockIdx.z = initial batch index (grid-stride loop handles the rest)
    // ================================================================

    int num_sms = 0;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    const unsigned spatial_tiles = (m / tile_m) * (n / tile_n);
    const unsigned batch_grid    = std::min(batch_count, std::max(1u, static_cast<unsigned>(num_sms) / spatial_tiles));

    dim3 grid_dim = {m / tile_m, n / tile_n, batch_grid};

    auto kernel = batched_gemm_kernel<BLAS,
                                      global_arrangement_c,
                                      c_value_type,
                                      c_value_type,
                                      c_value_type,
                                      decltype(device_pipeline)>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    kernel<<<grid_dim, device_pipeline.get_block_dim(), shared_memory_size>>>(
        alpha, beta, d_c, m, n, global_ldc, batch_count, device_pipeline);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // ================================
    // Copy results back and verify
    // ================================

    std::vector<c_value_type> host_c_result(c_total);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_c_result.data(), d_c, c_total * sizeof(c_value_type), cudaMemcpyDeviceToHost));

    const auto gemm_shape = std::make_tuple(m, n, k);
    const auto gemm_ld    = std::make_tuple(global_lda, global_ldb, global_ldc);

    bool all_correct = true;
    for (unsigned b = 0; b < batch_count; ++b) {
        std::vector<a_value_type> batch_a(host_a.begin() + b * m * k, host_a.begin() + (b + 1) * m * k);
        std::vector<b_value_type> batch_b(host_b.begin() + b * k * n, host_b.begin() + (b + 1) * k * n);
        std::vector<c_value_type> batch_c(host_c.begin() + b * m * n, host_c.begin() + (b + 1) * m * n);

        auto reference = example::reference_gemm<BLAS>(gemm_shape, gemm_ld, alpha, batch_a, batch_b, beta, batch_c);

        std::vector<c_value_type> batch_result(host_c_result.begin() + b * m * n,
                                               host_c_result.begin() + (b + 1) * m * n);

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

struct batched_gemm_pipeline_functor {
    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>, std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return batched_gemm_pipeline<Arch, Modifier>();
    }
};

int main(int, char**) {
    return example::sm_runner(batched_gemm_pipeline_functor {});
}
