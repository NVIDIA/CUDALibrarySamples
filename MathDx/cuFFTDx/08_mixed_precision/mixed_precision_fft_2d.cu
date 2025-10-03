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
#include <cooperative_groups.h>
#include <cufftdx.hpp>
#include <cufft.h>

#include "../common/mixed_io.hpp"
#include "../common/block_io_generic_strided.hpp"
#include "../common/common.hpp"
#include "../common/random.hpp"

template<class FFTX,
         class FFTY,
         bool UseSharedMemoryStridedIO,
         typename InputOutputType,
         unsigned int RequiredStorageSize = std::max({FFTX::storage_size, FFTY::storage_size})>
__launch_bounds__(FFTX::max_threads_per_block) __global__
    void mixed_fft_2d_kernel(const InputOutputType*        input,
                             InputOutputType*              output,
                             typename FFTX::workspace_type workspace_x,
                             typename FFTY::workspace_type workspace_y) {
    using complex_type      = typename FFTX::value_type;
    constexpr bool is_mixed = !std::is_same_v<InputOutputType, complex_type>;

    // Shared memory
    extern __shared__ __align__(alignof(float4)) unsigned char shared_mem[];

    // Local array for thread
    complex_type thread_data[RequiredStorageSize];

    // FFTY

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    if (blockIdx.x < (cufftdx::size_of<FFTX>::value / FFTX::ffts_per_block)) {
        using io_type = std::conditional_t<is_mixed, example::io_mixed<FFTY>, example::io<FFTY>>;
        // Load data from global memory to registers
        io_type::load(input, thread_data, local_fft_id);

        // Execute FFTY
        FFTY().execute(thread_data, reinterpret_cast<complex_type*>(shared_mem), workspace_y);

        // Save results
        io_type::store(thread_data, output, local_fft_id);
    }

    // Synchronize the whole CUDA Grid
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();

    // FFTX
    if (blockIdx.x < (cufftdx::size_of<FFTY>::value / FFTY::ffts_per_block)) {
        using dim_desc_type = example::dimension_description;
        using io_type       = example::io_generic_strided<FFTX>;

        // Load data from global memory to registers
        // Both loads will convert the input if mixed
        // precision is used
        if constexpr (UseSharedMemoryStridedIO) {
            io_type::load_strided<dim_desc_type::X, cufftdx::size_of<FFTX>::value, cufftdx::size_of<FFTY>::value>(
                output, thread_data, reinterpret_cast<InputOutputType*>(shared_mem), local_fft_id);
        } else {
            io_type::load_strided<dim_desc_type::X, cufftdx::size_of<FFTX>::value, cufftdx::size_of<FFTY>::value>(
                output, thread_data, local_fft_id);
        }

        // Execute FFTX
        FFTX().execute(thread_data, reinterpret_cast<complex_type*>(shared_mem), workspace_x);

        // Save results
        // Both stores will convert the input if mixed
        // precision is used
        if constexpr (UseSharedMemoryStridedIO) {
            io_type::store_strided<dim_desc_type::X, cufftdx::size_of<FFTX>::value, cufftdx::size_of<FFTY>::value>(
                thread_data, reinterpret_cast<InputOutputType*>(shared_mem), output, local_fft_id);
        } else {
            io_type::store_strided<dim_desc_type::X, cufftdx::size_of<FFTX>::value, cufftdx::size_of<FFTY>::value>(
                thread_data, output, local_fft_id);
        }
    }
}

template<typename InputOutputType>
example::fft_results<InputOutputType> cufft_mixed_fft_2d(const unsigned int warm_up_runs,
                                                         const unsigned int kernel_runs,
                                                         cudaStream_t       stream,
                                                         unsigned int       fft_size_x,
                                                         unsigned int       fft_size_y,
                                                         InputOutputType*   input,
                                                         InputOutputType*   output) {
    using complex_type = cufftComplex;
    static_assert(sizeof(InputOutputType) == sizeof(complex_type), "");
    static_assert(std::alignment_of_v<InputOutputType> == std::alignment_of_v<complex_type>, "");

    complex_type* cufft_input  = reinterpret_cast<complex_type*>(input);
    complex_type* cufft_output = reinterpret_cast<complex_type*>(output);

    // Create cuFFT plan
    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan2d(&plan, fft_size_x, fft_size_y, CUFFT_C2C));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    // Execute cuFFT
    auto cufft_execution = [&](cudaStream_t /* stream */) {
        CUFFT_CHECK_AND_EXIT(cufftExecC2C(plan, cufft_input, cufft_output, CUFFT_FORWARD));
    };
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Correctness run
    cufft_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    const size_t                 flat_fft_size       = fft_size_x * fft_size_y;
    const size_t                 flat_fft_size_bytes = flat_fft_size * sizeof(complex_type);
    std::vector<InputOutputType> output_host(flat_fft_size, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, flat_fft_size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Performance measurements
    auto time = example::measure_execution_ms(
        cufft_execution,
        warm_up_runs,
        kernel_runs,
        stream);

    // Clean-up
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));

    // Return results
    return {output_host, time / kernel_runs};
}

template<class FFTX, class FFTY, bool UseSharedMemoryStridedIO, typename InputOutputType>
example::fft_results<typename FFTX::value_type> cufftdx_mixed_fft_2d(const unsigned int warm_up_runs,
                                                                     const unsigned int kernel_runs,
                                                                     cudaStream_t       stream,
                                                                     InputOutputType*   input,
                                                                     InputOutputType*   output) {
    using complex_type                       = typename FFTX::value_type;
    static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
    static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;

    // Checks that FFTX and FFTY are correctly defined
    static_assert(std::is_same_v<cufftdx::precision_of_t<FFTX>, cufftdx::precision_of_t<FFTY>>,
                  "FFTY and FFTX must have the same precision");
    static_assert(std::is_same_v<typename FFTX::value_type, typename FFTY::value_type>,
                  "FFTY and FFTX must operator on the same type");
    // Checks below are not caused by any limitation in cuFFTDx, but rather in the example IO functions.
    static_assert((fft_size_x % FFTY::ffts_per_block == 0),
                  "FFTsPerBlock for FFTX must divide Y dimension as IO doesn't check if a batch is in range");

    // Checks that FFTX and FFTY can execute in the same kernel
    static_assert((FFTX::block_dim.x == FFTY::block_dim.x) && (FFTX::block_dim.y == FFTY::block_dim.y),
                  "Required block dimensions for FFTX and FFTY must be the same");

    constexpr bool is_mixed = !std::is_same_v<InputOutputType, complex_type>;

    // Shared memory IO for strided kernel may require more memory than FFTX::shared_memory_size.
    // Note: For some fft_size_x and depending on GPU architecture fft_x_shared_memory_smem_io may exceed max shared
    // memory and cudaFuncSetAttribute will fail.
    const unsigned int fft_shared_memory_smem_io =
        std::max<unsigned>({FFTX::shared_memory_size,
                            FFTY::shared_memory_size,
                            FFTX::ffts_per_block * fft_size_x * sizeof(InputOutputType),
                            FFTY::ffts_per_block * fft_size_y * sizeof(InputOutputType)});
    const unsigned int fft_shared_memory =
        UseSharedMemoryStridedIO ? fft_shared_memory_smem_io
                                 : std::max<unsigned>({FFTX::shared_memory_size, FFTY::shared_memory_size});
    const void* kernel     = (const void*)mixed_fft_2d_kernel<FFTX, FFTY, UseSharedMemoryStridedIO, InputOutputType>;
    auto        error_code = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, fft_shared_memory);
    CUDA_CHECK_AND_EXIT(error_code);

    // Create workspaces for FFTs
    auto workspace_y = cufftdx::make_workspace<FFTY>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_x = cufftdx::make_workspace<FFTX>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);

    // Synchronize device before execution
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const dim3 block_dim = FFTX::block_dim;
    const dim3 grid_size = {
        std::max<unsigned>({(fft_size_y / FFTY::ffts_per_block), (fft_size_x / FFTX::ffts_per_block)}), 1, 1};
    typename FFTX::workspace_type workspace_x_device = workspace_x;
    typename FFTY::workspace_type workspace_y_device = workspace_y;
    void*                         args[]             = {&input, &output, &workspace_x_device, &workspace_y_device};
    // Define 2D FFT execution
    auto mixed_fft_2d_execution = [&](cudaStream_t stream) {
        CUDA_CHECK_AND_EXIT(cudaLaunchCooperativeKernel(kernel, grid_size, block_dim, args, fft_shared_memory, stream));
        CUDA_CHECK_AND_EXIT(cudaGetLastError());
    };

    // Correctness run
    mixed_fft_2d_execution(stream);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    // Copy results to host
    static constexpr size_t   flat_fft_size = fft_size_x * fft_size_y;
    std::vector<complex_type> output_host(flat_fft_size, {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()});

    if constexpr (is_mixed) {
        // If the data is of mixed precision first copy InputOutputPrecision data into host memory
        std::vector<InputOutputType> output_mixed_host(flat_fft_size);
        CUDA_CHECK_AND_EXIT(cudaMemcpy(reinterpret_cast<void*>(output_mixed_host.data()), reinterpret_cast<void*>(output), flat_fft_size * sizeof(InputOutputType), cudaMemcpyDeviceToHost));

        // And then transform it to ComputePrecision and copy to the return vector created above
        std::transform(begin(output_mixed_host), end(output_mixed_host), begin(output_host), [](const auto& v) {
            return example::convert<complex_type>(v);
        });
    } else {
        // If the data is of constant precision perform a simple copy into host memory
        CUDA_CHECK_AND_EXIT(cudaMemcpy(output_host.data(), output, flat_fft_size * sizeof(complex_type), cudaMemcpyDeviceToHost));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    }

    // Performance measurements
    auto time = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            mixed_fft_2d_execution(stream);
        },
        warm_up_runs,
        kernel_runs,
        stream);

    // Return results
    return {output_host, time / kernel_runs};
}

// Example showing how cuFFTDx can be used to perform a 2D FFT in a single kernel utilizing mixed precision storage
// and computation using a cooperative grid kernel launch.
//
// Notes:
// * This example shows how to use cuFFTDx to run multi-dimensional FFT. Final performance will vary depending on the
// FFT definitions (precision, size, type, ept, fpb) and other user customizations.
// * Best possible performance requires adapting parameters in the sample to particular set of parameters and code customizations.
// * Only FP32 was tested for this example, other types might require adjustments.
// * The shared memory IO cuFFTDx has high shared memory requirements and will not work for all possible sizes in X dimension.
// * cudaLaunchCooperativeKernel puts restrictions on how big the FFT can be. All batches must be able to execute at the same time
// on the GPU.
// * The best results are for a square FFTs (fft_size_x == fft_size_y).
template<unsigned int Arch, unsigned int fft_size = 512>
void mixed_fft_2d() {

    // Choose mixed precision options
    // Input and Output type
    using io_precision = __half;
    // Computation type
    using compute_precision = float;
    // Initialization happens to make sure this mixed configuration is supported
    using mixed = example::mixed_precision<compute_precision, io_precision>;

    // FFT Sizes
    static constexpr unsigned int fft_size_y = fft_size;
    static constexpr unsigned int fft_size_x = fft_size;
    // Kernel Settings
    static constexpr unsigned int ept_y = 8;
    static constexpr unsigned int fpb_y = 8;
    static constexpr unsigned int ept_x = 8;
    static constexpr unsigned int fpb_x = fpb_y; // fpb for X and Y dimensions must be the same

    static constexpr unsigned int warm_up_runs = 1;
    static constexpr unsigned int kernel_runs  = 10;

    using namespace cufftdx;
    using fft_base = decltype(Block() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                              Precision<compute_precision>() + SM<Arch>());
    using fft_y    = decltype(fft_base() + Size<fft_size_y>() + ElementsPerThread<ept_y>() + FFTsPerBlock<fpb_y>());
    using fft_x    = decltype(fft_base() + Size<fft_size_x>() + ElementsPerThread<ept_x>() + FFTsPerBlock<fpb_x>());
    using fft      = fft_y;

    // Data types which will be used, based on previously choisen
    // compute and input/output values
    using complex_io_type      = typename mixed::complex_io_type;
    using complex_compute_type = cufftdx::complex<compute_precision>;

    // Host data
    static constexpr size_t flat_fft_size          = fft_size_x * fft_size_y;
    static constexpr size_t flat_fft_compute_bytes = flat_fft_size * sizeof(complex_compute_type);
    static constexpr size_t flat_fft_io_bytes      = flat_fft_size * sizeof(complex_io_type);

    // Shared memory must fit input data and must be big enough to run FFT
    complex_io_type *     input_io, *output_io;
    complex_compute_type *input_compute, *output_compute;

    // Input/Output precision data buffers
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input_io, flat_fft_io_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output_io, flat_fft_io_bytes));

    // Computation precision data buffers for comparison and verification
    CUDA_CHECK_AND_EXIT(cudaMalloc(&input_compute, flat_fft_compute_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output_compute, flat_fft_compute_bytes));

    // Host data
    std::vector<complex_compute_type> single_fft_host_input =
        example::get_random_complex_data<compute_precision>(flat_fft_size, -1, 1);
    std::vector<complex_io_type> single_fft_host_mixed_input(flat_fft_size);

    std::transform(begin(single_fft_host_input), end(single_fft_host_input), begin(single_fft_host_mixed_input), [](const auto& v) {
        return example::convert<io_precision>(v);
    });

    cudaMemcpy(reinterpret_cast<void*>(input_compute), single_fft_host_input.data(), flat_fft_compute_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(input_io), single_fft_host_mixed_input.data(), flat_fft_io_bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // cuFFTDx 2D
    auto cufftdx_mixed_results = cufftdx_mixed_fft_2d<fft_x, fft_y, false>(warm_up_runs, kernel_runs, stream, input_io, output_io);
    auto cufftdx_full_results  = cufftdx_mixed_fft_2d<fft_x, fft_y, false>(warm_up_runs, kernel_runs, stream, input_compute, output_compute);

    // cuFFTDx 2D
    // * Uses shared memory to speed-up IO in the strided kernel
    auto cufftdx_mixed_smemio_results = cufftdx_mixed_fft_2d<fft_x, fft_y, true>(warm_up_runs, kernel_runs, stream, input_io, output_io);
    auto cufftdx_full_smemio_results  = cufftdx_mixed_fft_2d<fft_x, fft_y, true>(warm_up_runs, kernel_runs, stream, input_compute, output_compute);

    // cuFFT as reference
    auto cufft_results = cufft_mixed_fft_2d(warm_up_runs, kernel_runs, stream, fft_size_x, fft_size_y, input_compute, output_compute);

    // Destroy created CUDA stream
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    // Free CUDA buffers
    // Verify Correctness
    bool success          = true;
    using result_vector_t = std::vector<complex_compute_type>;
    std::cout << "============\nCorrectness results:\n";

    auto verify_correctness = [&success](const auto& result_array, const auto& result_name, const auto& reference_array, const auto reference_name) {
        auto fft_error = example::fft_signal_error::calculate_for_complex_values(
            result_array,
            reference_array);

        std::cout << "Correctness results for: " << result_name << " using " << reference_name << " as reference\n";
        std::cout << "L2 relative error: " << fft_error.l2_relative_error << "\n";
        std::cout << "Peak error (index: " << fft_error.peak_error_index << "): " << fft_error.peak_error << "\n";
        if (success) {
            success = (fft_error.l2_relative_error < 0.001);
        }
    };

    verify_correctness(cufftdx_full_results.output, "cuFFTDx Full", cufft_results.output, "cuFFT");
    verify_correctness(cufftdx_mixed_results.output, "cuFFTDx Mixed", cufft_results.output, "cuFFT");
    verify_correctness(cufftdx_mixed_results.output, "cuFFTDx Mixed", cufftdx_full_results.output, "cuFFTDx Full");
    verify_correctness(cufftdx_full_smemio_results.output, "cuFFTDx Full Strided", cufft_results.output, "cuFFT");
    verify_correctness(cufftdx_mixed_smemio_results.output, "cuFFTDx Mixed Strided", cufft_results.output, "cuFFT");

    // Report performance results
    std::cout << "============\nPerformance results:\n";

    auto report_time_and_performance = [&](std::string name, float avg_time) -> void {
        double gflops = 1.0 * kernel_runs * 5.0 * flat_fft_size *
                        (std::log(flat_fft_size) / std::log(2)) / (kernel_runs * avg_time) / 1000000.0;

        std::cout << std::endl;
        std::cout << name << std::endl;
        std::cout << "Avg Time [ms_n]: " << avg_time << std::endl;
        std::cout << "Time (all) [ms_n]: " << avg_time * kernel_runs << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    };

    const dim3 grid_size = {
        std::max<unsigned>({(fft_size_y / fft_y::ffts_per_block), (fft_size_x / fft_x::ffts_per_block)}), 1, 1};

    std::cout << "FFT size (X, Y): (" << fft_size_x << ", " << fft_size_y << ")" << std::endl;
    std::cout << "CUDA grid: (X, Y): (" << grid_size.x << "," << grid_size.y << ")" << std::endl;

    report_time_and_performance("cuFFT", cufft_results.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Full Precision", cufftdx_full_results.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Mixed Precision", cufftdx_mixed_results.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Full Strided", cufftdx_full_smemio_results.avg_time_in_ms);
    report_time_and_performance("cuFFTDx Mixed Strided", cufftdx_mixed_smemio_results.avg_time_in_ms);

    if (success) {
        std::cout << "Success" << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(input_io));
    CUDA_CHECK_AND_EXIT(cudaFree(input_compute));
    CUDA_CHECK_AND_EXIT(cudaFree(output_io));
    CUDA_CHECK_AND_EXIT(cudaFree(output_compute));
}

template<unsigned int Arch>
struct mixed_fft_2d_functor {
    void operator()() { 
        mixed_fft_2d<Arch, 512>();
    }
};

int main(int, char**) {
    return example::sm_runner<mixed_fft_2d_functor>();
}