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

#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../common/common.hpp"
#include "../common/random.hpp"

#include "io_strided_conv_smem.hpp"
#include "kernels.hpp"
#include "reference.hpp"

// Used for getting stable performance results
inline constexpr unsigned int cufftdx_example_warm_up_runs     = 5;
inline constexpr unsigned int cufftdx_example_performance_runs = 15;

using namespace example;

// This function performs fused 3D FFT convolution with pre- and post-processing, ie. it executes:
// pre-processing, front 3D FFT, filter element-wise function (a.k.a. kernel), back 3D FFT.
// Front and back FFTs can either be inverse and forward FFTs, or more commonly forward and inverse.
template<int Batches, class FFTXPartial, class FFTYPartial, class FFTZPartial, class LoadFunctor, class FilterFunctor, class StoreFunctor, typename InputType>
auto cufftdx_3d_convolution_r2c(InputType* input, InputType* output, cudaStream_t stream) {
    using namespace cufftdx;

    using id_op = example::identity;


    // Create and configure kernel for the X dimension (strided)

    constexpr auto size_z          = cufftdx::size_of<FFTZPartial>::value;
    constexpr bool is_z_power_of_2 = is_power_of_2(size_z);

    constexpr auto z_real_mode = is_z_power_of_2 ? cufftdx::real_mode::folded : cufftdx::real_mode::normal;

    using FFTX  = decltype(FFTXPartial() + Direction<fft_direction::forward>());
    using IFFTX = decltype(FFTXPartial() + Direction<fft_direction::inverse>());

    using FFTY  = decltype(FFTYPartial() + Direction<fft_direction::forward>());
    using IFFTY = decltype(FFTYPartial() + Direction<fft_direction::inverse>());

    using FFTZ  = decltype(FFTZPartial() + Type<fft_type::r2c>() + RealFFTOptions<complex_layout::natural, z_real_mode>());
    using IFFTZ = decltype(FFTZPartial() + Type<fft_type::c2r>() + RealFFTOptions<complex_layout::natural, z_real_mode>());

    // Retrieve precision information from description type
    using input_type         = InputType;
    using value_type         = typename decltype(FFTZ())::value_type;
    using precision          = cufftdx::precision_of_t<FFTXPartial>;
    constexpr bool is_double = std::is_same_v<precision, double>;
    using vector_type        = std::conditional_t<is_double, double2, float2>;

    // Retrieve size information from description types
    static constexpr unsigned int x_input_length  = FFTX::input_length;
    static constexpr unsigned int y_input_length  = FFTY::input_length;
    static constexpr unsigned int z_output_length = FFTZ::output_length;

    static constexpr unsigned int flat_input_size     = x_input_length * y_input_length * size_z;
    static constexpr unsigned int total_complex_elems = x_input_length * y_input_length * z_output_length;

    // This buffer needs to be created for intermediate FFT results, since it cannot
    // trivially happen in-place when being padded with zeros.
    value_type* inter;
    const auto  flat_inter_size_bytes = total_complex_elems * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&inter, Batches * flat_inter_size_bytes));

    static constexpr auto x_fpb = FFTX::ffts_per_block;
    static constexpr auto y_fpb = FFTY::ffts_per_block;
    static constexpr auto z_fpb = FFTZ::ffts_per_block;

    static constexpr unsigned int x_batches = total_complex_elems / x_input_length;
    static constexpr unsigned int y_batches = total_complex_elems / y_input_length;
    static constexpr unsigned int z_batches = total_complex_elems / z_output_length;

    // X is always the convolution dimension
    using io_x_front = io_strided_conv_smem<dimension::x, true, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;
    using io_x_back  = io_strided_conv_smem<dimension::x, false, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;

    using io_y_front = io_strided_conv_smem<dimension::y, true, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;
    using io_y_back  = io_strided_conv_smem<dimension::y, false, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;

    using io_z_front = io_strided_conv_smem<dimension::z, true, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;
    using io_z_back  = io_strided_conv_smem<dimension::z, false, Batches, FFTX, IFFTX, FFTY, IFFTY, FFTZ, IFFTZ>;

    cudaError_t err;

    auto workspace_x = cufftdx::make_workspace<FFTX>(err, stream);
    auto workspace_y = cufftdx::make_workspace<FFTY>(err, stream);
    auto workspace_z = cufftdx::make_workspace<FFTZ>(err, stream);

    // Increase max shared memory if needed (includes extra padding)
    constexpr int x_max_bytes = io_x_front::get_shared_bytes();
    constexpr int y_max_bytes = io_y_front::get_shared_bytes();
    constexpr int z_max_bytes = io_z_front::get_shared_bytes();

    // Define kernels to set maximal shared memory in CUDA runtime
    // Refer to kernels.hpp for a detailed kernel explanation.
    auto set_kernel_shared_size = [](auto kernel, auto size) {
        CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            size));
    };

    auto kernel_z_front = fft_kernel<FFTZ, io_z_front, LoadFunctor, id_op, input_type, value_type>;
    set_kernel_shared_size(kernel_z_front, z_max_bytes);

    // No transform ops for middle dimension
    auto kernel_y_front = fft_kernel<FFTY, io_y_front, id_op, id_op, value_type>;
    set_kernel_shared_size(kernel_y_front, y_max_bytes);

    auto kernel_x = convolution_kernel<FFTX, IFFTX, FilterFunctor, io_x_front, io_x_back>;
    set_kernel_shared_size(kernel_x, x_max_bytes);

    // No transform ops for middle dimension
    auto kernel_y_back = fft_kernel<IFFTY, io_y_back, id_op, id_op, value_type>;
    set_kernel_shared_size(kernel_y_back, y_max_bytes);

    auto kernel_z_back = fft_kernel<IFFTZ, io_z_back, id_op, StoreFunctor, value_type, input_type>;
    set_kernel_shared_size(kernel_z_back, z_max_bytes);

    // Execute cuFFTDx in Z-Y-X order
    auto cufftdx_execution = [&](cudaStream_t stream) {
        // CUDA Grid configuration is as follows:
        // Grid --> (Total Subbatches / FPB, Batches, 1)
        // Block --> (Size / EPT, FPB)
        kernel_z_front<<<dim3 {example::div_up(z_batches, z_fpb), Batches, 1}, FFTZ::block_dim, z_max_bytes, stream>>>(
            z_batches,
            input,
            inter,
            workspace_z);
        kernel_y_front<<<dim3 {example::div_up(y_batches, y_fpb), Batches, 1}, FFTY::block_dim, y_max_bytes, stream>>>(
            y_batches,
            inter,
            inter,
            workspace_y);
        // Convolution is performed in the strided dimension to save on strided global memory transfers
        // which occur otherwise.
        kernel_x<<<dim3 {example::div_up(x_batches, x_fpb), Batches, 1}, FFTX::block_dim, x_max_bytes, stream>>>(
            x_batches,
            inter,
            inter,
            workspace_x);
        kernel_y_back<<<dim3 {example::div_up(y_batches, y_fpb), Batches, 1}, IFFTY::block_dim, y_max_bytes, stream>>>(
            y_batches,
            inter,
            inter,
            workspace_y);
        kernel_z_back<<<dim3 {example::div_up(z_batches, z_fpb), Batches, 1}, IFFTZ::block_dim, z_max_bytes, stream>>>(
            z_batches,
            inter,
            output,
            workspace_z);
    };

    // Correctness run
    cufftdx_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results to host
    const size_t            flat_input_size_bytes = flat_input_size * sizeof(input_type);
    std::vector<input_type> output_host(Batches * flat_input_size, std::numeric_limits<precision>::quiet_NaN());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, Batches * flat_input_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Performance measurements
    auto time = example::measure_execution_ms(
        cufftdx_execution,
        cufftdx_example_warm_up_runs,
        cufftdx_example_performance_runs,
        stream);

    // Clean up
    CUDA_CHECK_AND_EXIT(cudaFree(inter));

    // Return results
    return example::fft_results<input_type> {output_host, (time / cufftdx_example_performance_runs)};
}

template<int Arch>
int conv_3d() {
    using namespace cufftdx;

    // 3D Convolution configuration
    static constexpr unsigned int batches = 200;

    // X - the outermost (most strided) dimension
    static constexpr unsigned int fft_size_x = 81;
    static constexpr unsigned int x_ept      = 9;
    static constexpr unsigned int x_fpb      = 9;

    // Y - the middle dimension, strided
    static constexpr unsigned int fft_size_y = 81;
    static constexpr unsigned int y_ept      = 9;
    static constexpr unsigned int y_fpb      = 16;

    // Z - the contiguous dimension
    static constexpr unsigned int fft_size_z = 128;
    static constexpr unsigned int z_ept      = 16;
    static constexpr unsigned int z_fpb      = 16;


    // Definition of functors for preprocessing, filtering
    // and postprocessing. These will be fused with FFT kernels
    using load_functor   = example::rational_scaler<1, 2>;
    using filter_functor = example::rational_scaler<2, 3>;
    using store_functor  = example::rational_scaler<3, 4>;

    // Only FP32 and FP64 supported for cuFFT reference
    constexpr bool is_double_precision = true;
    using precision                    = std::conditional_t<is_double_precision, double, float>;

    // Create cuFFTDx description type summarizing all information
    // passed above regarding the outermost dimension.
    using fftx_partial = decltype(Block() + Size<fft_size_x>() + Type<fft_type::c2c>() +
                                  ElementsPerThread<x_ept>() + FFTsPerBlock<x_fpb>() +
                                  Precision<precision>() + SM<Arch>());

    // Create cuFFTDx description type summarizing all information
    // passed above regarding the middle dimension.
    using ffty_partial = decltype(Block() + Size<fft_size_y>() + Type<fft_type::c2c>() +
                                  ElementsPerThread<y_ept>() + FFTsPerBlock<y_fpb>() +
                                  Precision<precision>() + SM<Arch>());

    // Create cuFFTDx description type summarizing all information
    // passed above regarding the contiguous dimension.
    using fftz_partial = decltype(Block() + Size<fft_size_z>() +
                                  ElementsPerThread<z_ept>() + FFTsPerBlock<z_fpb>() +
                                  Precision<precision>() + SM<Arch>());

    // Helper types
    using input_type = precision;

    // Normally, we could get this information from
    const unsigned int flat_input_size = fft_size_x * fft_size_y * fft_size_z;

    // Generate random input data on host
    auto host_input = example::get_random_real_data<input_type>(batches * flat_input_size, -1, 1);

    // Allocate managed memory for device input/output
    input_type* input;
    input_type* output;
    const auto  flat_input_size_bytes = flat_input_size * sizeof(input_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input, batches * flat_input_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, batches * flat_input_size_bytes));

    // Copy input to the device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(input, host_input.data(), batches * flat_input_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Run cuFFTDx
    auto cufftdx_results = cufftdx_3d_convolution_r2c<batches, fftx_partial, ffty_partial, fftz_partial, load_functor, filter_functor, store_functor>(input, output, stream);

    // Run cuFFT
    auto cufft_results = cufft_3d_convolution<true, true, load_functor, filter_functor, store_functor>(fft_size_x, fft_size_y, fft_size_z, batches, input, output, stream, cufftdx_example_warm_up_runs, cufftdx_example_performance_runs);

    // Clean-up
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(input));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Check if cuFFTDx results are correct
    auto fft_error =
        example::fft_signal_error::calculate_for_real_values(cufftdx_results.output, cufft_results.output);

    std::cout << "FFT: (" << fft_size_x << ", " << fft_size_y << ", " << fft_size_z << ")\n";

    bool success = fft_error.l2_relative_error < 0.001;
    std::cout << "Correctness results:\n";
    std::cout << "L2 error: " << fft_error.l2_relative_error << "\n";
    std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";

    // Print performance results
    if (success) {
        std::cout << "\nPerformance results:\n";
        std::cout << std::setw(10) << "cuFFTDx: " << cufftdx_results.avg_time_in_ms << " [ms]\n";
        std::cout << std::setw(10) << "cuFFT + thrust: " << cufft_results.avg_time_in_ms << " [ms]\n";
    }

    if (success) {
        std::cout << "Success\n";
        return 0;
    } else {
        std::cout << "Failure\n";
        return 1;
    }
}

template<unsigned int Arch>
struct conv_3d_functor {
    int operator()() {
        return conv_3d<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner<conv_3d_functor>();
}