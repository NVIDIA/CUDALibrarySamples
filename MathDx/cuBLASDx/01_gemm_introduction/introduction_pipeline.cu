/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <array>
#include <iostream>
#include <system_error>
#include <iomanip>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"

template<class BLAS, class Alpha, class Beta, class CTensor, class DevicePipeline>
__launch_bounds__(DevicePipeline::max_threads_per_block, 1) __global__
    void gemm_kernel(Alpha const                            alpha,
                     Beta const                             beta,
                     CTensor                                global_c,
                     // IMPORTANT: Notice __grid_constant__ is used for device_pipeline argument
                     __grid_constant__ DevicePipeline const device_pipeline) {
#ifdef __CUDA_ARCH__
    if constexpr (cublasdx::sm_of_v<BLAS> == __CUDA_ARCH__) {

        // Use device_pipeline traits to properly align dynamic shared memory
        extern __shared__ __align__(device_pipeline.buffer_alignment()) char smem[];

        // Instantiate tile_pipeline for trivial tile dictated by block coordinates
        auto tile_pipeline = device_pipeline.get_tile(smem, blockIdx.x, blockIdx.y);

        // Define epilogue to be opaquely executed by BLAS::block_dim threads
        auto epilogue_functor = [&](auto& accumulator) {
            // Partition logical C tile of input/output data
            auto tile_gmem_c   = cublasdx::get_tile(global_c, BLAS::c_shape, blockIdx.x, blockIdx.y);
            // Create fragment and copy appropriate per-thread partition of data into it
            auto d_fragment = accumulator.make_partition_and_copy(tile_gmem_c);
            // D = alpha * A * B + beta * C
            cublasdx::axpby(alpha, accumulator.get_results(), beta, d_fragment);
            // Store results back to global memory
            accumulator.partition_and_copy(d_fragment, tile_gmem_c);
        };

        // Execute the pipeline, accumulating over entire K dimension the chosen per-CTA tile
        tile_pipeline.execute(epilogue_functor);
    }
#endif
}

// This is an example of testing performance of cuBLASDx as tile provider, executing a general matrix multiply (GEMM)
// on the entire GPU and distributing work across all SMs, each of them running a small GEMM tile with cuBLASDx.
//
//              C = alpha * A * B + beta * C
//
// A, B, and C are matrices. Mixed precisions are supported, decoupled precisions are supported.
//

template<unsigned int Arch, cublasdx::sm_modifier Modifier>
int introduction_pipeline() {

    // ===================================
    // Configurable Global GEMM properties
    // ===================================

    // Global GEMM Size --> MNK, where:
    // - A matrix is M x K
    // - B matrix is K x N
    // - C matrix is M x N

    // This size can be set dynamically from command line
    const auto m = 8192; // Global M GEMM Size
    const auto n = 8192; // Global N GEMM Size
    const auto k = 8192; // Global K GEMM Size

    // Global GEMM Arrangement:
    // - cubladsx::row_major, row major data arrangement
    // - cubladsx::col_major, col major data arrangement
    // Note: these values need to be constexpr
    constexpr auto global_arrangement_a = cublasdx::row_major;
    constexpr auto global_arrangement_b = cublasdx::col_major;
    constexpr auto global_arrangement_c = cublasdx::row_major;

    // Leading Dimensions to be used for global data
    // Note: for matrix of size X x Y, the LD must be:
    // - greater or equal than X if matrix is col-major
    // - greater or equal than Y if matrix is row-major
    // Note: these values can be dynamic
    const auto global_lda = (global_arrangement_a == cublasdx::col_major) ? m : k;
    const auto global_ldb = (global_arrangement_b == cublasdx::col_major) ? k : n;
    const auto global_ldc = (global_arrangement_c == cublasdx::col_major) ? m : n;

    // Compute precision (use Tensor Cores of this precision)
    // and cuBLAS input precision
    using a_compute_precision = __half;
    using b_compute_precision = __half;
    using c_compute_precision = float;

    // Number type, either real or complex
    constexpr auto type = cublasdx::type::real;

    // Create data type, based on:
    // - precision
    // - type (real / complex)
    // this will be either precision or cublasdx::complex<precision>
    using a_compute_value_type = cute::conditional_t<type == cublasdx::type::real, a_compute_precision, cublasdx::complex<a_compute_precision>>;
    using b_compute_value_type = cute::conditional_t<type == cublasdx::type::real, b_compute_precision, cublasdx::complex<b_compute_precision>>;
    using c_compute_value_type = cute::conditional_t<type == cublasdx::type::real, c_compute_precision, cublasdx::complex<c_compute_precision>>;

    // Scalar multipliers
    // C = alpha * A * B + beta * C
    c_compute_value_type alpha = example::make_value<c_compute_value_type>(1.1);
    c_compute_value_type beta  = example::make_value<c_compute_value_type>(1.2);

    // ======================================
    // Configurable cuBLASDx tile properties
    // ======================================

    // tile size, this describes smaller GEMM,
    // which will be computed by each threadblock
    // using cuBLASDx
    constexpr unsigned tile_m = 128;
    constexpr unsigned tile_n = 128;
    constexpr unsigned tile_k = 32;

    // Number of threads to compute the tile described above
    constexpr int tile_threads = 128;

    // If pipeline_depth is left at 0, an automatically generated depth
    // will be used, equal to the maximal possible value based on shared memory
    // size of chosen device architecture
    constexpr unsigned pipeline_depth   = 2;

    // Arrangement of data in a per-threadblock tile of data
    constexpr auto tile_arr_a = global_arrangement_a;
    constexpr auto tile_arr_b = global_arrangement_b;
    constexpr auto tile_arr_c = global_arrangement_c;

    // Maximal alignment to be used for shared memory data.
    // Effectively this limits maximal vectorization level
    // for loads and stores.
    constexpr unsigned int maximal_alignment  = 16;
    constexpr unsigned int cublasdx_alignment = maximal_alignment;

    // ================================
    // Verify configuration correctness
    // ================================

    auto k_stages = k / tile_k;
    if (k_stages < pipeline_depth) {
        std::cerr << "PipelineDepth must be less or equal to GEMM k stages, please adjust pipeline_depth"
                  << std::endl;
        return 1;
    }

    const bool divisible = (m % tile_m == 0 and n % tile_n == 0 and k % tile_k == 0);
    if (not divisible) {
        std::cerr << "M, N, K dimensions must be divisible by tile_m, tile_n, tile_k" << std::endl;
        return 1;
    }

    // ================================
    // Prepare inputs
    // ================================

    auto host_a_io = example::get_random_data<a_compute_value_type>(m * k, 1);
    auto host_b_io = example::get_random_data<b_compute_value_type>(k * n, 2);
    auto host_c_io = example::get_random_data<c_compute_value_type>(m * n, 3);

    a_compute_value_type* a_data = nullptr;
    b_compute_value_type* b_data = nullptr;
    c_compute_value_type* c_data = nullptr;

    CUDA_CHECK_AND_EXIT(
        cudaMalloc(&a_data, m * k * sizeof(a_compute_value_type)));
    CUDA_CHECK_AND_EXIT(
        cudaMalloc(&b_data, k * n * sizeof(b_compute_value_type)));
    CUDA_CHECK_AND_EXIT(
        cudaMalloc(&c_data, m * n * sizeof(c_compute_value_type)));

    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(a_data, host_a_io.data(), m * k * sizeof(a_compute_value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(b_data, host_b_io.data(), k * n * sizeof(b_compute_value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(c_data, host_c_io.data(), m * n * sizeof(c_compute_value_type), cudaMemcpyHostToDevice));

    // Create tensors for global A / B / C corresponding to set MNK, arrangement and LDs
    auto global_a = cublasdx::make_gmem_tensor<global_arrangement_a>(a_data, m, k, global_lda);
    auto global_b = cublasdx::make_gmem_tensor<global_arrangement_b>(b_data, k, n, global_ldb);
    auto global_c = cublasdx::make_gmem_tensor<global_arrangement_c>(c_data, m, n, global_ldc);

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))

    // cuBLASDx type creation
    using BLAS = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                          cublasdx::Precision<a_compute_precision, b_compute_precision, c_compute_precision>() +
                          cublasdx::Type<type>() + cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<tile_arr_a, tile_arr_b, tile_arr_c>() + cublasdx::Block() +
                          cublasdx::BlockDim<tile_threads>() +
                          cublasdx::Alignment<cublasdx_alignment, cublasdx_alignment, cublasdx_alignment>() +
                          cublasdx::WithPipeline() + cublasdx::SM<Arch, Modifier>());

    // =============================
    // Setup cuBLASDx pipeline
    // =============================

    // Note that if there is enough smem and register pressure is low then it might be possible to execute
    // multiple persistent CTAs per SM
    dim3 grid_dim = dim3 {m / tile_m, n / tile_n, 1};

    // Attempt to create cuBLASDx pipeline object
    auto opt_device_pipeline = cublasdx::suggest_device_pipeline<pipeline_depth, BLAS>(global_a, global_b);

    // Check if object is valid
    if (not opt_device_pipeline) {
        std::cout << "Incorrect pipeline configuration, please ensure global tensors are divisible by tile"
                    << std::endl;
        exit(1);
    }

    // Get device_pipeline, used to instantiate tile_pipelines later
    auto device_pipeline = opt_device_pipeline.value();

    // Use device_pipeline traits to get necessary shared memory allocation size
    auto shared_memory_size = cublasdx::make_shared_storage_calculator()
                                    .add(device_pipeline.buffer_alignment(), device_pipeline.buffer_size())
                                    .get();

    using alpha_t = c_compute_value_type;
    using beta_t = c_compute_value_type;
    auto kernel = gemm_kernel<BLAS, alpha_t, beta_t, decltype(global_c), decltype(device_pipeline)>;
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    // Pass device_pipeline to kernel as regular by value argument
    kernel<<<grid_dim, device_pipeline.get_block_dim(), shared_memory_size, stream>>>(
        alpha, beta, global_c, device_pipeline);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy back data
    std::vector<c_compute_value_type> results(m * n);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(), c_data, results.size() * sizeof(c_compute_value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free resources.
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(a_data));
    CUDA_CHECK_AND_EXIT(cudaFree(b_data));
    CUDA_CHECK_AND_EXIT(cudaFree(c_data));

    printf("Success! Please refer to example_11 (device_gemm_performance) for an advanced measurement\n");

    return 0;
}

struct introduction_pipeline_functor {

    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>,
                   std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return introduction_pipeline<Arch, Modifier>();
    }
};

int main(int , char** ) {
    return example::sm_runner(introduction_pipeline_functor {});
}
