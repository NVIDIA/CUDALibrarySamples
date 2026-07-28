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

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

namespace {
    template<class BLAS, unsigned SwizzleM, class Alpha, class Beta, class CTensor, class Pipeline>
    __launch_bounds__(Pipeline::max_threads_per_block, 1) __global__
    void pipeline_kernel(Alpha const alpha,
                         Beta const beta,
                         CTensor global_c,
                         __grid_constant__ Pipeline const device_pipeline) {
        CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(BLAS);
        extern __shared__ __align__(128) cublasdx::byte smem[];
        const auto [tile_coord_m, tile_coord_n] = example::get_threadblock_swizzled_tile_coord<SwizzleM>();

        auto tile_c = cublasdx::get_tile(global_c, BLAS::c_shape, tile_coord_m, tile_coord_n);
        auto tile_pipeline = device_pipeline.get_tile(smem, tile_coord_m, tile_coord_n);
        auto accumulator = tile_pipeline.get_accumulator();

        tile_pipeline.execute(accumulator);

        if (accumulator.is_thread_active()) {
            auto c_fragment = accumulator.get_results();
            auto d_fragment = accumulator.make_partition_and_copy(tile_c);
            cublasdx::axpby(alpha, c_fragment, beta, d_fragment);
            accumulator.partition_and_copy(d_fragment, tile_c);
        }
    }

    template<class BLAS, unsigned SwizzleM, class Alpha, class Beta, class ATensor, class BTensor, class CTensor>
    auto measure_cublasdx_emulated(Alpha const alpha,
                                   Beta const beta,
                                   ATensor const& global_a,
                                   BTensor const& global_b,
                                   CTensor global_c,
                                   unsigned const kernel_warm_up_repeats,
                                   unsigned const kernel_repeats,
                                   cudaStream_t const stream) {
        constexpr auto tile_m = cublasdx::size_of_v_m<BLAS>;
        constexpr auto tile_n = cublasdx::size_of_v_n<BLAS>;
        const auto m = cute::shape<0>(global_c);
        const auto n = cute::shape<1>(global_c);

        auto run_cublasdx_gemm = [&](cudaStream_t const str) {
            // cublasdx::suggest_pipeline returns a result that needs to be checked
            auto pipeline =
                cublasdx::suggest_pipeline<BLAS, cublasdx::reusable_accumulator>(global_a, global_b, str);

            if (!pipeline) {
                auto const error = pipeline.error();
                std::cout << "Failed to create device pipeline";
                if (error.code != cublasdx::pipeline_error_code::none) {
                    std::cout << ": " << cublasdx::pipeline_error_string(error.code);
                }
                if (error.get_cuda_error() != cudaSuccess) {
                    std::cout << " (" << cudaGetErrorString(error.get_cuda_error()) << ")";
                }
                std::cout << std::endl;
                exit(1);
            }

            auto shared_memory_size = pipeline->buffer_size();

            auto kernel = pipeline_kernel<BLAS, SwizzleM, Alpha, Beta, CTensor, decltype(pipeline->get_device_handle())>;
            CUDA_CHECK_AND_EXIT(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

            kernel<<<dim3(m / tile_m, n / tile_n, 1), pipeline->get_block_dim(), shared_memory_size, str>>>(
                alpha, beta, global_c, pipeline->get_device_handle());
        };

        run_cublasdx_gemm(stream);
        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        using c_value_type = typename CTensor::value_type;
        std::vector<c_value_type> results(m * n);
        auto const c_ptr = cute::raw_pointer_cast(global_c.data());
        CUDA_CHECK_AND_EXIT(
            cudaMemcpy(results.data(), c_ptr, results.size() * sizeof(c_value_type), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        auto time = example::measure::execution(run_cublasdx_gemm, kernel_warm_up_repeats, kernel_repeats, stream);

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        return std::make_tuple(time, results);
    }
}

template<unsigned Arch, cublasdx::sm_modifier Modifier, class GlobalShape>
int sgemm_emulation_pipeline(GlobalShape const global_shape) {
    using value_type = float;

    const auto m = cute::get<0>(global_shape);
    const auto n = cute::get<1>(global_shape);
    const auto k = cute::get<2>(global_shape);

    constexpr auto global_arrangement_a = cublasdx::row_major;
    constexpr auto global_arrangement_b = cublasdx::col_major;
    constexpr auto global_arrangement_c = cublasdx::row_major;

    const auto global_lda = (global_arrangement_a == cublasdx::col_major) ? m : k;
    const auto global_ldb = (global_arrangement_b == cublasdx::col_major) ? k : n;
    const auto global_ldc = (global_arrangement_c == cublasdx::col_major) ? m : n;

    // Default 128x128x128 tile; 
    // increase tile_n to 256 for B200 and later
    constexpr unsigned tile_m = 128;
    constexpr unsigned tile_n = 128;
    constexpr unsigned tile_k = 128;
    constexpr unsigned threads = 128;
    constexpr unsigned swizzle_m = 16;
    constexpr unsigned required_mantissa_bits = 23;
    constexpr unsigned kernel_warm_up_repeats = 1;
    constexpr unsigned kernel_repeats = 10;

    constexpr value_type alpha = 1.1f;
    constexpr value_type beta = 1.2f;

    using emulated_blas = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                                   cublasdx::Precision<value_type, value_type, value_type>() +
                                   cublasdx::Type<cublasdx::type::real>() +
                                   cublasdx::Function<cublasdx::function::MM>() +
                                   cublasdx::Arrangement<global_arrangement_a,
                                                         global_arrangement_b,
                                                         global_arrangement_c>() +
                                   cublasdx::Block() +
                                   cublasdx::BlockDim<threads>() +
                                   cublasdx::StaticBlockDim() +
                                   cublasdx::Alignment<16, 16, 16>() +
                                   cublasdx::EnableInputStreaming() +
                                   cublasdx::WithPipeline() +
                                   cublasdx::RequiredMantissaBits<required_mantissa_bits>() +
                                   cublasdx::SM<Arch, Modifier>());

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    auto host_a = example::get_random_data<value_type>(std::size_t(m) * k, 1);
    auto host_b = example::get_random_data<value_type>(std::size_t(k) * n, 2);
    auto host_c = example::get_random_data<value_type>(std::size_t(m) * n, 3);

    example::device_vector<value_type> device_a = host_a;
    example::device_vector<value_type> device_b = host_b;
    example::device_vector<value_type> device_emulated_c = host_c;
    example::device_vector<value_type> device_cublas_c = host_c;

    auto gemm_shape = cute::make_shape(m, n, k);
    constexpr auto global_arrangement = cute::make_tuple(
        std::integral_constant<cublasdx::arrangement, cublasdx::row_major> {},
        std::integral_constant<cublasdx::arrangement, cublasdx::col_major> {},
        std::integral_constant<cublasdx::arrangement, cublasdx::row_major> {});
    auto global_ld = cute::make_tuple(global_lda, global_ldb, global_ldc);

    auto global_a = cublasdx::make_gmem_tensor<global_arrangement_a>(device_a.data(), m, k, global_lda);
    auto global_b = cublasdx::make_gmem_tensor<global_arrangement_b>(device_b.data(), k, n, global_ldb);
    auto emulated_c = cublasdx::make_gmem_tensor<global_arrangement_c>(device_emulated_c.data(), m, n, global_ldc);

    auto [time_cublasdx, host_dx_results] =
        measure_cublasdx_emulated<emulated_blas, swizzle_m>(alpha,
                                                            beta,
                                                            global_a,
                                                            global_b,
                                                            emulated_c,
                                                            kernel_warm_up_repeats,
                                                            kernel_repeats,
                                                            stream);

    auto [time_cublas, host_blas_results] =
        example::cublaslt_runner<value_type, value_type, value_type>(gemm_shape, global_arrangement, global_ld)
            .execute_with_time_and_results(alpha,
                                           device_a.data(),
                                           device_b.data(),
                                           beta,
                                           device_cublas_c.data(),
                                           kernel_warm_up_repeats,
                                           kernel_repeats,
                                           stream);

    const double flops = example::gemm_flops<value_type, value_type, value_type>(m, n, k);
    const double avg_time_dx = time_cublasdx.event_ms / kernel_repeats;
    const double avg_host_time_dx = time_cublasdx.host_ms / kernel_repeats;
    const double dx_gflops = flops / (avg_time_dx * 1e6);
    const double avg_time_cublas = time_cublas.event_ms / kernel_repeats;
    const double avg_host_time_cublas = time_cublas.host_ms / kernel_repeats;
    const double cublas_gflops = flops / (avg_time_cublas * 1e6);

    std::cout << "m, n, k: " << m << ", " << n << ", " << k << std::endl;
    std::cout << "Compute Type A: " << example::type_string<value_type>() << std::endl;
    std::cout << "Compute Type B: " << example::type_string<value_type>() << std::endl;
    std::cout << "Compute Type C: " << example::type_string<value_type>() << std::endl;
    std::cout << "Dx Input Precision A: " << example::precision_string<value_type>() << std::endl;
    std::cout << "Dx Input Precision B: " << example::precision_string<value_type>() << std::endl;
    std::cout << "Dx Input Precision C: " << example::precision_string<value_type>() << std::endl;
    std::cout << "Tile m, n, k: " << tile_m << ", " << tile_n << ", " << tile_k << std::endl;

    std::cout << "\ncuBLASDx\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Avg time [ms]  = " << avg_time_dx << "\n";
    std::cout << "Avg host time [ms]  = " << avg_host_time_dx << "\n";
    std::cout << "Avg GFLOP/s  = " << dx_gflops << "\n";

    std::cout << "\ncuBLASLt native FP32 (not including heuristic)\n";
    std::cout << "Avg time [ms]  = " << avg_time_cublas << "\n";
    std::cout << "Avg host time [ms]  = " << avg_host_time_cublas << "\n";
    std::cout << "Avg GFLOP/s  = " << cublas_gflops << "\n";

    constexpr bool verbose_knob = false;
    constexpr bool print_knob = true;
    auto error = example::calculate_error(host_dx_results, host_blas_results, verbose_knob, print_knob);
    std::cout << std::fixed << std::setprecision(10) << "Error = " << error << "\n";
    const bool is_correct = example::is_error_acceptable<value_type, value_type, value_type>(error);
    std::cout << (is_correct ? "Success!" : "Failure!") << "\n";

    std::cout << std::fixed << std::setprecision(2)
              << "cuBLAS / cuBLASDx timings = " << time_cublas.event_ms / time_cublasdx.event_ms << "\n";
    std::cout << "cuBLAS / cuBLASDx host timings = " << time_cublas.host_ms / time_cublasdx.host_ms << "\n";

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    return is_correct ? 0 : 1;
}

struct sgemm_emulation_pipeline_functor {
    template<int Arch, cublasdx::sm_modifier Modifier, class GlobalShape>
    int operator()(std::integral_constant<int, Arch>,
                   std::integral_constant<cublasdx::sm_modifier, Modifier>,
                   GlobalShape const global_shape) const {
        return sgemm_emulation_pipeline<Arch, Modifier>(global_shape);
    }
};

int main(int const argc, char** argv) {
    std::array<unsigned int, 3> mnk = {8192, 8192, 8192};
    auto usage = []() { std::cerr << "Incorrect usage: ./sgemm_emulation_pipeline [m n k]" << std::endl; };

    if (argc == 4) {
        try {
            std::transform(argv + 1, argv + argc, mnk.begin(), [&](char const* const dim_input) { return std::stoul(dim_input); });
        } catch (...) {
            usage();
            return 1;
        }
    } else if (argc != 1) {
        usage();
        return 1;
    }

    return example::sm_runner(sgemm_emulation_pipeline_functor {}, cute::make_shape(mnk[0], mnk[1], mnk[2]));
}
