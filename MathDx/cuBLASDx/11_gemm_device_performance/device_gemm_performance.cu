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
#include "../reference/reference.hpp"

template<class BLAS, class Alpha, class Beta, class CTensor, class DevicePipeline>
__launch_bounds__(DevicePipeline::max_threads_per_block, 1) __global__
    void gemm_kernel(Alpha const                            alpha,
                     Beta const                             beta,
                     CTensor                                global_c,
                     __grid_constant__ DevicePipeline const device_pipeline) {
#ifdef __CUDA_ARCH__
    if constexpr (cublasdx::sm_of_v<BLAS> == __CUDA_ARCH__) {

        extern __shared__ __align__(device_pipeline.buffer_alignment()) char smem[];

        auto tile_pipeline = device_pipeline.get_tile(smem, blockIdx.x, blockIdx.y);
        auto tile_gmem_c   = cublasdx::get_tile(global_c, BLAS::c_shape, blockIdx.x, blockIdx.y);

        auto epilogue_functor = [&](auto& accumulator) {
            auto d_fragment = accumulator.make_partition_and_copy(tile_gmem_c);
            cublasdx::axpby(alpha, accumulator.get_results(), beta, d_fragment);
            accumulator.partition_and_copy(d_fragment, tile_gmem_c);
        };

        tile_pipeline.execute(epilogue_functor);
    }
#endif
}

template<class BLAS,
         int PipelineDepth,
         class GEMMShape,
         class GEMMArr,
         class GEMMLD,
         class Alpha,
         class AValueType,
         class BValueType,
         class Beta,
         class CValueType>
auto measure_cublasdx(GEMMShape         gemm_shape,
                      GEMMArr           gemm_arr,
                      GEMMLD            gemm_ld,
                      const Alpha       alpha,
                      const AValueType* a,
                      const BValueType* b,
                      const Beta        beta,
                      CValueType*       c,
                      unsigned          kernel_warm_up_repeats,
                      unsigned          kernel_repeats,
                      cudaStream_t      stream) {
    // Grid size configuration
    const auto [m, n, k]       = gemm_shape;
    const auto [lda, ldb, ldc] = gemm_ld;
    constexpr auto tile_m      = cublasdx::size_of_v_m<BLAS>;
    constexpr auto tile_n      = cublasdx::size_of_v_n<BLAS>;

    // Create tensors for global A / B / C corresponding to set MNK, arrangement and LDs
    auto global_a = cublasdx::make_gmem_tensor<cute::get<0>(gemm_arr)>(a, m, k, lda);
    auto global_b = cublasdx::make_gmem_tensor<cute::get<1>(gemm_arr)>(b, k, n, ldb);
    auto global_c = cublasdx::make_gmem_tensor<cute::get<2>(gemm_arr)>(c, m, n, ldc);

    int num_sms = 0;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));

    // Note that if there is enough smem and register pressure is low then it might be possible to execute
    // multiple persistent CTAs per SM
    dim3 grid_dim = dim3 {m / tile_m, n / tile_n, 1};

    auto run_cublasdx_gemm = [&](cudaStream_t str) {
        auto opt_device_pipeline = cublasdx::suggest_device_pipeline<PipelineDepth, BLAS>(global_a, global_b);

        if (not opt_device_pipeline) {
            std::cout << "Incorrect pipeline configuration, please ensure global tensors are divisible by tile"
                      << std::endl;
            exit(1);
        }

        auto device_pipeline = opt_device_pipeline.value();

        // Increase max dynamic shared memory for the kernel if needed.
        auto shared_memory_size = cublasdx::make_shared_storage_calculator()
                                      .add(device_pipeline.buffer_alignment(), device_pipeline.buffer_size())
                                      .get();

        auto kernel = gemm_kernel<BLAS, Alpha, Beta, decltype(global_c), decltype(device_pipeline)>;
        CUDA_CHECK_AND_EXIT(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
        kernel<<<grid_dim, device_pipeline.get_block_dim(), shared_memory_size, str>>>(
            alpha, beta, global_c, device_pipeline);
    };

    // First run for correctness check
    run_cublasdx_gemm(stream);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy back data
    std::vector<CValueType> results(m * n);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(), c, results.size() * sizeof(CValueType), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Execute kernel.
    double time = example::measure::execution(run_cublasdx_gemm, kernel_warm_up_repeats, kernel_repeats, stream);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    return std::make_tuple(time, results);
}

// This is an example of testing performance of cuBLASDx as tile provider, executing a general matrix multiply (GEMM)
// on the entire GPU and distributing work across all SMs, each of them running a small GEMM tile with cuBLASDx.
//
//              C = alpha * A * B + beta * C
//
// A, B, and C are matrices. Mixed precisions are supported, decoupled precisions are supported.
// cuBLASLt is the comparison point for this example.
//
// This example provides a CLI interface for changing the dynamic size of the GEMM
// make device_gemm_performance <m> <n> <k>
//
// Please refer to the documentation for more details.

template<unsigned int Arch, cublasdx::sm_modifier Modifier, class GlobalShape>
int device_gemm_performance(GlobalShape global_shape) {

    // ==========================================================
    // Tile chosen for big SGEMMs on B200 (fp32), for best
    // results for other precisions please see our recommended
    // configurations below, or attempt a thorough
    // parameter scan over: (tile_m / tile_n / tile_k / threads)
    // and staging C loads / stores through shared memory

    // ===================================
    // Configurable Global GEMM properties
    // ===================================

    // Global GEMM Size --> MNK, where:
    // - A matrix is M x K
    // - B matrix is K x N
    // - C matrix is M x N

    // This size can be set dynamically from command line
    const auto m = cute::get<0>(global_shape); // Global M GEMM Size
    const auto n = cute::get<1>(global_shape); // Global N GEMM Size
    const auto k = cute::get<2>(global_shape); // Global K GEMM Size

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

    // If manual_pipeline_depth is left at 0, an automatically generated depth
    // will be used, equal to the maximal possible value based on shared memory
    // size of chosen device architecture
    constexpr unsigned manual_pipeline_depth   = 0;
    constexpr bool     override_pipeline_depth = (manual_pipeline_depth != 0);

    // GeForce tiles to try
    // fp8  | TN | (4096 to 8192)  | --> V:64,64,64   T:128
    // fp8  | TN | (over 8192)     | --> V:128,128,64 T:128
    // int8 | TN | (4096 to 8192)  | --> V:64,64,64   T:128
    // int8 | TN | (over 8192)     | --> V:128,128,64 T:128
    // fp16 | TN | (4096 to 8192)  | --> V:64,64,32   T:128
    // fp16 | TN | (over 8192)     | --> V:128,128,32 T:128
    // tf32 | TN | (4096 to ~6000) | --> V:64,64,16   T:128
    // tf32 | TN | (~6000 to 8192) | --> V:64,64,32   T:128
    // tf32 | TN | (over 8192)     | --> V:128,128,16 T:128
    // fp32 | TN | (col-major)     | --> V:256,128,32 T:256
    // fp32 | TT | (row-major)     | --> V:64,128,16  T:128
    // fp64 | TN | (all sizes)     | --> V:64,64,32   T:128

    // Datacenter tiles to try
    // fp32 | (big)   |            | --> V:128,128,32, T:128
    // fp32 | (small) |            | --> V:128,64,16,  T:128


    // Arrangement of data in a per-threadblock tile of data
    constexpr auto tile_arr_a = global_arrangement_a;
    constexpr auto tile_arr_b = global_arrangement_b;
    constexpr auto tile_arr_c = global_arrangement_c;

    // Input used to be converted to the final compute precision,
    // this can be used to simulate either in-flight quantization
    // or flexible upcasting of data to save on bandwidth
    // NOTE: either these types must be implicitly convertible to
    // compute types, or converters should be provided in appropriate places.
    // please refer to simple_gemm_fp32_decoupled.cu example for more details.
    using a_io_value_type = a_compute_value_type;
    using b_io_value_type = b_compute_value_type;
    using c_io_value_type = c_compute_value_type;

    // Maximal alignment to be used for shared memory data.
    // Effectively this limits maximal vectorization level
    // for loads and stores.
    constexpr unsigned int maximal_alignment  = 16;
    constexpr unsigned int cublasdx_alignment = maximal_alignment;

    // ================================
    // Verify configuration correctness
    // ================================

    constexpr unsigned stage_shared_req = tile_m * tile_k * sizeof(a_compute_value_type) +
                                          tile_k * tile_n * sizeof(b_compute_value_type) +
                                          sizeof(cublasdx::pipeline_stage_scratch_t);

    constexpr unsigned available_shared_memory = commondx::device_info<Arch>::shared_memory();
    constexpr unsigned maximal_pipeline_depth  = cute::min(16, (available_shared_memory - 32) / stage_shared_req);
    constexpr unsigned pipeline_depth = override_pipeline_depth ? manual_pipeline_depth : maximal_pipeline_depth;
    static_assert(pipeline_depth <= maximal_pipeline_depth, "The chosen pipeline depth requires more shared memory than is available for the target architecture");

    auto k_stages = k / tile_k;
    if (k_stages < pipeline_depth) {
        std::cerr << "PipelineDepth must be less or equal to GEMM k stages, please adjust manual_pipeline_depth"
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

    // Use tuples to avoid passing 20 arguments to a function
    constexpr auto global_arrangement = cute::make_tuple(
        // These must be passed as integral constants to properly dispatch static striding
        std::integral_constant<cublasdx::arrangement, global_arrangement_a> {},
        std::integral_constant<cublasdx::arrangement, global_arrangement_b> {},
        std::integral_constant<cublasdx::arrangement, global_arrangement_c> {});

    const auto global_ld = cute::make_tuple(global_lda, global_ldb, global_ldc);

    // Performance comparison parameters
    const unsigned int kernel_repeats         = 5;
    const unsigned int kernel_warm_up_repeats = 15;

    constexpr bool compare_to_fp64_cublas = false;
    using a_cublas_value_type             = cute::conditional_t<compare_to_fp64_cublas, double, a_compute_value_type>;
    using b_cublas_value_type             = cute::conditional_t<compare_to_fp64_cublas, double, b_compute_value_type>;
    using c_cublas_value_type             = cute::conditional_t<compare_to_fp64_cublas, double, c_compute_value_type>;

    // Test implementation
    a_io_value_type* a_cublasdx = nullptr;
    b_io_value_type* b_cublasdx = nullptr;
    c_io_value_type* c_cublasdx = nullptr;

    a_cublas_value_type* a_cublas = nullptr;
    b_cublas_value_type* b_cublas = nullptr;
    c_cublas_value_type* c_cublas = nullptr;

    // Fill the A, B, C matrices with random values.
    {
        // Use nullptr tensors to make it easier to calculate memory requirements
        auto dummy_global_a =
            cublasdx::make_gmem_tensor<cute::get<0>(global_arrangement)>(a_cublasdx, m, k, global_lda);
        auto dummy_global_b =
            cublasdx::make_gmem_tensor<cute::get<1>(global_arrangement)>(b_cublasdx, k, n, global_ldb);
        auto dummy_global_c =
            cublasdx::make_gmem_tensor<cute::get<2>(global_arrangement)>(c_cublasdx, m, n, global_ldc);

        CUDA_CHECK_AND_EXIT(
            cudaMalloc(&a_cublasdx, cublasdx::cosize(dummy_global_a.layout()) * sizeof(a_io_value_type)));
        CUDA_CHECK_AND_EXIT(
            cudaMalloc(&a_cublas, cublasdx::cosize(dummy_global_a.layout()) * sizeof(a_cublas_value_type)));

        CUDA_CHECK_AND_EXIT(
            cudaMalloc(&b_cublasdx, cublasdx::cosize(dummy_global_b.layout()) * sizeof(b_io_value_type)));
        CUDA_CHECK_AND_EXIT(
            cudaMalloc(&b_cublas, cublasdx::cosize(dummy_global_b.layout()) * sizeof(b_cublas_value_type)));

        CUDA_CHECK_AND_EXIT(
            cudaMalloc(&c_cublasdx, cublasdx::cosize(dummy_global_c.layout()) * sizeof(c_io_value_type)));
        CUDA_CHECK_AND_EXIT(
            cudaMalloc(&c_cublas, cublasdx::cosize(dummy_global_c.layout()) * sizeof(c_cublas_value_type)));

        auto host_a_io = example::get_random_data<a_io_value_type>(m * k, 1);
        auto host_b_io = example::get_random_data<b_io_value_type>(k * n, 2);
        auto host_c_io = example::get_random_data<c_io_value_type>(m * n, 3);

        // Create A cuBLASLt input
        using a_converter_t      = example::converter<a_cublas_value_type>;
        using a_functor_output_t = cublasdx::detail::res_t<a_converter_t, a_io_value_type>;
        static_assert(std::is_convertible_v<a_functor_output_t, a_cublas_value_type>,
                      "Input type must be convertible to compute type");

        auto host_a_cublas = std::vector<a_cublas_value_type>(host_a_io.size());
        std::transform(host_a_io.cbegin(), host_a_io.cend(), host_a_cublas.begin(), a_converter_t {});

        // Create B cuBLASLt input
        using b_converter_t      = example::converter<b_cublas_value_type>;
        using b_functor_output_t = cublasdx::detail::res_t<b_converter_t, b_io_value_type>;
        static_assert(std::is_convertible_v<b_functor_output_t, b_cublas_value_type>,
                      "Input type must be convertible to compute type");

        auto host_b_cublas = std::vector<b_cublas_value_type>(host_b_io.size());
        std::transform(host_b_io.cbegin(), host_b_io.cend(), host_b_cublas.begin(), b_converter_t {});

        // Create C cuBLASLt input
        using c_converter_t      = example::converter<c_cublas_value_type>;
        using c_functor_output_t = cublasdx::detail::res_t<c_converter_t, c_io_value_type>;
        static_assert(std::is_convertible_v<c_functor_output_t, c_cublas_value_type>,
                      "Input type must be convertible to compute type");

        auto host_c_cublas = std::vector<c_cublas_value_type>(host_c_io.size());
        std::transform(host_c_io.cbegin(), host_c_io.cend(), host_c_cublas.begin(), c_converter_t {});

        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(a_cublasdx, host_a_io.data(), m * k * sizeof(a_io_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(b_cublasdx, host_b_io.data(), k * n * sizeof(b_io_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(c_cublasdx, host_c_io.data(), m * n * sizeof(c_io_value_type), cudaMemcpyHostToDevice));

        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(a_cublas, host_a_cublas.data(), m * k * sizeof(a_cublas_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(b_cublas, host_b_cublas.data(), k * n * sizeof(b_cublas_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(c_cublas, host_c_cublas.data(), m * n * sizeof(c_cublas_value_type), cudaMemcpyHostToDevice));
        // destroy host vectors
    }

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))

    // cuBLASDx type creation
    using BLAS = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                          cublasdx::Precision<a_compute_precision, b_compute_precision, c_compute_precision>() +
                          cublasdx::Type<type>() + cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<tile_arr_a, tile_arr_b, tile_arr_c>() + cublasdx::Block() +
                          cublasdx::BlockDim<tile_threads>() + cublasdx::StaticBlockDim() +
                          cublasdx::Alignment<cublasdx_alignment, cublasdx_alignment, cublasdx_alignment>() +
                          cublasdx::EnableInputStreaming() + cublasdx::WithPipeline() + cublasdx::SM<Arch, Modifier>());

    // =============================
    // Execute cuBLASDx and cuBLASLt
    // =============================

    auto [time_cublasdx, host_dx_results] = measure_cublasdx<BLAS, pipeline_depth>(global_shape,
                                                                                   global_arrangement,
                                                                                   global_ld,
                                                                                   alpha,
                                                                                   a_cublasdx,
                                                                                   b_cublasdx,
                                                                                   beta,
                                                                                   c_cublasdx,
                                                                                   kernel_warm_up_repeats,
                                                                                   kernel_repeats,
                                                                                   stream);

    // Measure cuBLAS performance.
    auto [time_cublas, host_blas_results] =
        example::cublaslt_runner<a_cublas_value_type, b_cublas_value_type, c_cublas_value_type>(
            global_shape, global_arrangement, global_ld)
            .execute_with_time_and_results(
                alpha, a_cublas, b_cublas, beta, c_cublas, kernel_warm_up_repeats, kernel_repeats, stream);

    // Write performance data.
    using cublasdx::size_of;
    std::cout << "m, n, k: " << m << ", " << n << ", " << k << std::endl;
    std::cout << "Compute Type A: " << example::type_string<a_compute_value_type>() << std::endl;
    std::cout << "Compute Type B: " << example::type_string<b_compute_value_type>() << std::endl;
    std::cout << "Compute Type C: " << example::type_string<c_compute_value_type>() << std::endl;
    std::cout << "Dx Input Precision A: " << example::precision_string<a_io_value_type>() << std::endl;
    std::cout << "Dx Input Precision B: " << example::precision_string<b_io_value_type>() << std::endl;
    std::cout << "Dx Input Precision C: " << example::precision_string<c_io_value_type>() << std::endl;

    const double avg_time_dx = time_cublasdx / kernel_repeats;
    const double dx_gflops =
        example::gemm_flops<a_compute_value_type, b_compute_value_type, c_compute_value_type>(m, n, k) /
        (avg_time_dx * 1e6);

    std::cout << "\ncuBLASDx\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Avg time [ms]  = " << avg_time_dx << "\n";
    std::cout << "Avg GFLOP/s  = " << dx_gflops << "\n";

    const double avg_time_cublas = time_cublas / kernel_repeats;
    double cublas_gflops = example::gemm_flops<a_cublas_value_type, b_cublas_value_type, c_cublas_value_type>(m, n, k) /
                           (avg_time_cublas * 1e6);

    std::cout << "\ncuBLASLt (not including heuristic)\n";
    std::cout << "Avg time [ms]  = " << avg_time_cublas << "\n";
    std::cout << "Avg GFLOP/s  = " << cublas_gflops << "\n";

    constexpr bool verbose_knob = false;
    constexpr bool print_knob   = true;

    auto error = example::calculate_error(host_dx_results, host_blas_results, verbose_knob, print_knob);
    std::cout << std::fixed << std::setprecision(10) << "Error = " << error << "\n";

    std::cout << std::fixed << std::setprecision(2) << "cuBLAS / cuBLASDx timings = " << time_cublas / time_cublasdx
              << "\n";

    // Free resources.
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(a_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(b_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(c_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(a_cublas));
    CUDA_CHECK_AND_EXIT(cudaFree(b_cublas));
    CUDA_CHECK_AND_EXIT(cudaFree(c_cublas));

    return 0;
}

struct device_gemm_performance_functor {

    template<int Arch, cublasdx::sm_modifier Modifier, class GlobalShape>
    int operator()(std::integral_constant<int, Arch>,
                   std::integral_constant<cublasdx::sm_modifier, Modifier>,
                   GlobalShape global_shape) {
        return device_gemm_performance<Arch, Modifier>(global_shape);
    }
};

int main(int argc, char** argv) {
    std::array<unsigned int, 3> mnk = {8192, 8192, 8192};
    auto usage = []() { std::cerr << "Incorrect usage: ./device_gemm_performance [m n k]" << std::endl; };

    if (argc == 4) {
        std::cout << "Tile optimized for big SGEMMs on B200 (fp32), for best "
                     "results for other precisions please see our recommended "
                     "configurations in device_gemm_performance.cu, or attempt a thorough "
                     "parameter scan over: (tile_m / tile_n / tile_k / threads) "
                     "and staging C loads / stores through shared memory"
                  << std::endl;

        try {
            std::transform(argv + 1, argv + argc, mnk.begin(), [&](char* dim_input) { return std::stoul(dim_input); });
        } catch (...) {
            usage();
            return 1;
        }
    } else if (argc != 1) {
        usage();
        return 1;
    }

    return example::sm_runner(device_gemm_performance_functor {}, cute::make_shape(mnk[0], mnk[1], mnk[2]));
}
