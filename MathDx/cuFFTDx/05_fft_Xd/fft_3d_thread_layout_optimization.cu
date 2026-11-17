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

#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cufftdx.hpp>

#include "../common/block_io_generic_strided.hpp"
#include "../common/common.hpp"
#include "../common/random.hpp"

inline constexpr unsigned int cufftdx_example_warm_up_runs     = 5;
inline constexpr unsigned int cufftdx_example_performance_runs = 20;

template<class FFT,
         example::dimension_description Dimension,
         unsigned int                   SizeX,
         unsigned int                   SizeY,
         unsigned int                   SizeZ,
         class ComplexType = typename FFT::value_type>
__launch_bounds__(FFT::max_threads_per_block) __global__ void fft_3d_thread_layout_kernel(
    const ComplexType*           input,
    ComplexType*                 output,
    typename FFT::workspace_type workspace) {
    CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(FFT) {

        using io_type = example::io_generic_strided<FFT>;

        ComplexType thread_data[FFT::storage_size];

        // io_generic_strided uses logical working-group IDs, so it is agnostic to
        // the physical CUDA-thread order selected by ThreadLayout.
        const auto batch_id = FFT::working_group::batchid();
        io_type::load_strided<Dimension, SizeX, SizeY, SizeZ>(input, thread_data, batch_id);

        extern __shared__ __align__(alignof(float4)) cufftdx::byte shared_memory[];
        FFT().execute(thread_data, shared_memory, workspace);

        io_type::store_strided<Dimension, SizeX, SizeY, SizeZ>(thread_data, output, batch_id);
    }
}

template<class T>
example::fft_results<T> cufft_fft_3d(unsigned int fft_size_x,
                                     unsigned int fft_size_y,
                                     unsigned int fft_size_z,
                                     T*           input,
                                     T*           output,
                                     cudaStream_t stream) {
    using complex_type = cufftComplex;
    static_assert(sizeof(T) == sizeof(complex_type), "");
    static_assert(alignof(T) == alignof(complex_type), "");

    auto* cufft_input  = reinterpret_cast<complex_type*>(input);
    auto* cufft_output = reinterpret_cast<complex_type*>(output);

    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan3d(&plan, fft_size_x, fft_size_y, fft_size_z, CUFFT_C2C));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    const auto cufft_execution = [&](cudaStream_t /* execution_stream */) {
        CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, cufft_input, cufft_output, CUFFT_FORWARD));
    };

    cufft_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const size_t   flat_fft_size       = fft_size_x * fft_size_y * fft_size_z;
    const size_t   flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
    std::vector<T> output_host(flat_fft_size,
                               {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const auto time = example::measure_execution_ms(
        cufft_execution, cufftdx_example_warm_up_runs, cufftdx_example_performance_runs, stream);

    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
    return {output_host, time / cufftdx_example_performance_runs};
}

template<class FFTX, class FFTY, class FFTZ, class T>
example::fft_results<T> cufftdx_fft_3d_thread_layout(T* input, T* output, cudaStream_t stream) {
    using dim_desc_type                      = example::dimension_description;
    using complex_type                       = typename FFTX::value_type;
    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int fft_size_z = cufftdx::size_of<FFTZ>::value;

    static_assert(std::is_same_v<cufftdx::precision_of_t<FFTX>, cufftdx::precision_of_t<FFTY>>,
                  "FFTX and FFTY must have the same precision");
    static_assert(std::is_same_v<cufftdx::precision_of_t<FFTY>, cufftdx::precision_of_t<FFTZ>>,
                  "FFTY and FFTZ must have the same precision");
    static_assert(std::is_same_v<typename FFTX::value_type, typename FFTY::value_type>,
                  "FFTX and FFTY must operate on the same type");
    static_assert(std::is_same_v<typename FFTY::value_type, typename FFTZ::value_type>,
                  "FFTY and FFTZ must operate on the same type");
    static_assert(sizeof(T) == sizeof(complex_type), "");
    static_assert(alignof(T) == alignof(complex_type), "");
    static_assert((fft_size_x * fft_size_y) % FFTZ::ffts_per_block == 0,
                  "FFTsPerBlock for FFTZ must divide X * Y");
    static_assert((fft_size_x * fft_size_z) % FFTY::ffts_per_block == 0,
                  "FFTsPerBlock for FFTY must divide X * Z");
    static_assert((fft_size_y * fft_size_z) % FFTX::ffts_per_block == 0,
                  "FFTsPerBlock for FFTX must divide Y * Z");

    auto* cufftdx_input  = reinterpret_cast<complex_type*>(input);
    auto* cufftdx_output = reinterpret_cast<complex_type*>(output);

    auto error_code = cudaFuncSetAttribute(
        fft_3d_thread_layout_kernel<FFTZ, dim_desc_type::Z, fft_size_x, fft_size_y, fft_size_z, complex_type>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFTZ::shared_memory_size);
    CUDA_CHECK_AND_EXIT(error_code);
    error_code = cudaFuncSetAttribute(
        fft_3d_thread_layout_kernel<FFTY, dim_desc_type::Y, fft_size_x, fft_size_y, fft_size_z, complex_type>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFTY::shared_memory_size);
    CUDA_CHECK_AND_EXIT(error_code);
    error_code = cudaFuncSetAttribute(
        fft_3d_thread_layout_kernel<FFTX, dim_desc_type::X, fft_size_x, fft_size_y, fft_size_z, complex_type>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFTX::shared_memory_size);
    CUDA_CHECK_AND_EXIT(error_code);

    auto workspace_z = cufftdx::make_workspace<FFTZ>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_y = cufftdx::make_workspace<FFTY>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_x = cufftdx::make_workspace<FFTX>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);

    const auto grid_fft_size_z   = example::div_up(fft_size_y * fft_size_x, FFTZ::ffts_per_block);
    const auto grid_fft_size_y   = example::div_up(fft_size_z * fft_size_x, FFTY::ffts_per_block);
    const auto grid_fft_size_x   = example::div_up(fft_size_y * fft_size_z, FFTX::ffts_per_block);
    const auto cufftdx_execution = [&](cudaStream_t execution_stream) {
        fft_3d_thread_layout_kernel<FFTZ, dim_desc_type::Z, fft_size_x, fft_size_y, fft_size_z, complex_type>
            <<<grid_fft_size_z, FFTZ::block_dim, FFTZ::shared_memory_size, execution_stream>>>(
                cufftdx_input, cufftdx_output, workspace_z);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        fft_3d_thread_layout_kernel<FFTY, dim_desc_type::Y, fft_size_x, fft_size_y, fft_size_z, complex_type>
            <<<grid_fft_size_y, FFTY::block_dim, FFTY::shared_memory_size, execution_stream>>>(
                cufftdx_output, cufftdx_output, workspace_y);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
        fft_3d_thread_layout_kernel<FFTX, dim_desc_type::X, fft_size_x, fft_size_y, fft_size_z, complex_type>
            <<<grid_fft_size_x, FFTX::block_dim, FFTX::shared_memory_size, execution_stream>>>(
                cufftdx_output, cufftdx_output, workspace_x);
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
    };

    cufftdx_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    static constexpr size_t   flat_fft_size       = fft_size_x * fft_size_y * fft_size_z;
    static constexpr size_t   flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
    std::vector<complex_type> output_host(
        flat_fft_size, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const auto time = example::measure_execution_ms(
        cufftdx_execution, cufftdx_example_warm_up_runs, cufftdx_example_performance_runs, stream);
    return {output_host, time / cufftdx_example_performance_runs};
}

// This example performs a 3D FFT in three kernels. Z accesses contiguous data
// and uses the regular ThreadLayout. Y and X access strided data and use the
// transposed ThreadLayout, which makes direct global-memory I/O coalesced
// without external shared-memory staging.
template<unsigned int Arch>
int fft_3d_thread_layout_optimization() {
    using precision_type = float;
    using complex_type   = cufftdx::complex<precision_type>;

    static constexpr unsigned int fft_size_z = 32;
    static constexpr unsigned int fft_size_y = 1024;
    static constexpr unsigned int fft_size_x = 1024;
    static constexpr unsigned int ept_z      = 16;
    static constexpr unsigned int fpb_z      = 4;
    static constexpr unsigned int ept_y      = 32;
    static constexpr unsigned int fpb_y      = 2;
    static constexpr unsigned int ept_x      = 32;
    static constexpr unsigned int fpb_x      = 2;

    using namespace cufftdx;
    using fft_base = decltype(Block() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                              Precision<precision_type>() + SM<Arch>());
    using fft_z    = decltype(fft_base() + Size<fft_size_z>() + ElementsPerThread<ept_z>() + FFTsPerBlock<fpb_z>() +
                           experimental::ThreadLayout<experimental::thread_layout::regular>());
    using fft_y    = decltype(fft_base() + Size<fft_size_y>() + ElementsPerThread<ept_y>() + FFTsPerBlock<fpb_y>() +
                           experimental::ThreadLayout<experimental::thread_layout::transposed>());
    using fft_x    = decltype(fft_base() + Size<fft_size_x>() + ElementsPerThread<ept_x>() + FFTsPerBlock<fpb_x>() +
                           experimental::ThreadLayout<experimental::thread_layout::transposed>());

    static constexpr size_t flat_fft_size       = fft_size_x * fft_size_y * fft_size_z;
    static constexpr size_t flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
    const auto              input_host          = example::get_random_complex_data<precision_type>(flat_fft_size, -1, 1);

    complex_type* input  = nullptr;
    complex_type* output = nullptr;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input, flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, flat_fft_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(input, input_host.data(), flat_fft_size_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    const auto cufftdx_results = cufftdx_fft_3d_thread_layout<fft_x, fft_y, fft_z>(input, output, stream);
    const auto cufft_results   = cufft_fft_3d(fft_size_x, fft_size_y, fft_size_z, input, output, stream);

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(input));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    const auto fft_error = example::fft_signal_error::calculate_for_complex_values(
        cufftdx_results.output, cufft_results.output);
    const bool success = fft_error.l2_relative_error < 0.001;

    std::cout << "FFT: (" << fft_size_x << ", " << fft_size_y << ", " << fft_size_z << ")\n";
    std::cout << "Correctness results:\n";
    std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
    std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
    if (success) {
        std::cout << "\nPerformance results:\n";
        std::cout << std::setw(10) << "cuFFTDx: " << cufftdx_results.avg_time_in_ms << " [ms]\n";
        std::cout << std::setw(10) << "cuFFT: " << cufft_results.avg_time_in_ms << " [ms]\n";
        std::cout << "Success\n";
        return 0;
    }

    std::cout << "Failure\n";
    return 1;
}

template<unsigned int Arch>
struct fft_3d_thread_layout_optimization_functor {
    int operator()() { return fft_3d_thread_layout_optimization<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<fft_3d_thread_layout_optimization_functor>();
}
