/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include <cufft.h>
#ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
#include <cufftXt.h>
#include <nvrtc.h>
#include <nvFatbin.h>
#endif

#include "../common/block_io.hpp"
#include "../common/common.hpp"
#include "../common/random.hpp"

#ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
#    include "../common/common_nvrtc.hpp"
#endif

// Compile-time tuning knobs. 
#ifndef CUFFTDX_CONV_FFT_SIZE
#    define CUFFTDX_CONV_FFT_SIZE 512
#endif

#ifndef CUFFTDX_CONV_ELEMENTS_PER_THREAD
#    define CUFFTDX_CONV_ELEMENTS_PER_THREAD 8
#endif

#ifndef CUFFTDX_CONV_FFTS_PER_BLOCK
#    define CUFFTDX_CONV_FFTS_PER_BLOCK 2
#endif

#ifndef CUFFTDX_CONV_USE_SUGGESTED
#    define CUFFTDX_CONV_USE_SUGGESTED 1
#endif

#ifndef CUFFTDX_CONV_PRECISION
#    define CUFFTDX_CONV_PRECISION float
#endif

#ifndef CUFFTDX_CONV_TRANSPOSE_BY_COMPONENT_PADDED
#    define CUFFTDX_CONV_TRANSPOSE_BY_COMPONENT_PADDED 0
#endif

template<class FFT, class IFFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void convolution_kernel(typename FFT::value_type*     data,
                                                                                 typename FFT::workspace_type  workspace,
                                                                                 typename IFFT::workspace_type workspace_inverse) {
    CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(FFT) {

        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        // Local array for thread
        complex_type thread_data[FFT::storage_size];

        // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
        const unsigned int batchid = FFT::working_group::batchid();

        // Load data from global memory to registers
        example::io<FFT>::load(data, thread_data, batchid);

        // Execute FFT
        extern __shared__ __align__(alignof(float4)) cufftdx::byte shared_mem[];
        FFT().execute(thread_data, shared_mem, workspace);

        // Scale values
        scalar_type scale = 1.0 / cufftdx::size_of<FFT>::value;
        for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
            thread_data[i].x *= scale;
            thread_data[i].y *= scale;
        }

        // Execute inverse FFT
        IFFT().execute(thread_data, shared_mem, workspace_inverse);

        // Save results
        example::io<FFT>::store(thread_data, data, batchid);
    }
}

// Scaling kernel; transforms data between cuFFTs.
template<unsigned int fft_size, class Complex>
__global__ void scaling_kernel(Complex*            data,
                               const unsigned int input_size,
                               const unsigned int ept) {

    using scalar_type = decltype(Complex{}.x);
    static constexpr scalar_type scale = scalar_type{1.0} / fft_size;

    Complex temp;
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = 0; i < ept; i++) {
        if (index < input_size) {
            temp = data[index];
            temp.x *= scale;
            temp.y *= scale;
            data[index] = temp;
            index += blockDim.x * gridDim.x;
        }
    }
}

#ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
// cuFFT LTO (JIT) callback. The callback is compiled to LTO-IR at run time with
// NVRTC and handed to cuFFT via cufftXtSetJITCallback, which links it into the
// FFT kernel while the plan is being built. 
const char* scaling_callback_source_float = R"callback(
#include <cuda_runtime.h>
#include <cufftXt.h>

// Declared with C++ linkage on purpose: cuFFT mangles the symbol name given to
// cufftXtSetJITCallback with the standard callback signature, so an extern "C"
// definition would not be found by the generated wrapper.
__device__ cufftComplex cufftdx_example_scaling_load(void*              dataIn,
                                                     unsigned long long offset,
                                                     void*              callerInfo,
                                                     void*              sharedPtr) {
    static const float scale = 1.0f / CUFFTDX_CB_FFT_SIZE;
    cufftComplex value = ((cufftComplex*)dataIn)[offset];
    value.x *= scale;
    value.y *= scale;
    return value;
}
)callback";

const char* scaling_callback_source_double = R"callback(
#include <cuda_runtime.h>
#include <cufftXt.h>

__device__ cufftDoubleComplex cufftdx_example_scaling_load(void*              dataIn,
                                                            unsigned long long offset,
                                                            void*              callerInfo,
                                                            void*              sharedPtr) {
    static const double scale = 1.0 / CUFFTDX_CB_FFT_SIZE;
    cufftDoubleComplex value = ((cufftDoubleComplex*)dataIn)[offset];
    value.x *= scale;
    value.y *= scale;
    return value;
}
)callback";

#define NVFATBIN_SAFE_CALL(x)                                                                      \
    do {                                                                                           \
        nvFatbinResult result = x;                                                                 \
        if (result != NVFATBIN_SUCCESS) {                                                          \
            std::cout << "nvFatbin error: " << nvFatbinGetErrorString(result) << " (" << #x << ")" \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;                       \
            return {};                                                                             \
        }                                                                                          \
    } while (0)

// Compile the callback above to LTO-IR for the current device and wrap it in a
// fatbinary, which is the format cufftXtSetJITCallback expects.
template<class Complex>
inline std::vector<char> compile_scaling_callback(unsigned int fft_size, int device) {
    const char* scaling_callback_source = std::is_same_v<Complex, cufftComplex>
                                               ? scaling_callback_source_float
                                               : scaling_callback_source_double;

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, scaling_callback_source, "scaling_callback.cu", 0, nullptr, nullptr));

    // NVRTC compiles this callback for the architecture of the running `device` (queried at
    // run time), which must coincide with the architecture cuFFTDx itself was compiled for.
    // See docs/cufftdx/examples.rst, convolution_performance section, for details.
    std::string arch_opt = example::nvrtc::get_device_architecture_option(device);
    std::string def_size = "-DCUFFTDX_CB_FFT_SIZE=" + std::to_string(fft_size);
    std::vector<const char*> opts = {"--std=c++17",
                                     "--device-as-default-execution-space",
                                     "-dlto",
                                     "--relocatable-device-code=true",
                                     arch_opt.c_str(),
                                     def_size.c_str()};
    std::vector<std::string> include_dirs = example::nvrtc::get_cufftdx_include_dirs();
    for (auto& d : include_dirs) {
        opts.push_back(d.c_str());
    }

    if (nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data()) != NVRTC_SUCCESS) {
        example::nvrtc::print_program_log(prog);
        nvrtcDestroyProgram(&prog);
        return {};
    }

    size_t ltoir_size = 0;
    NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &ltoir_size));
    std::vector<char> ltoir(ltoir_size);
    NVRTC_SAFE_CALL(nvrtcGetLTOIR(prog, ltoir.data()));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    // Pack the LTO-IR into a fatbinary. nvFatbin takes the bare architecture
    // number (the XX of sm_XX / lto_XX / compute_XX).
    const std::string arch = std::to_string(example::nvrtc::get_device_architecture(device));
    nvFatbinHandle    fatbin;
    const char*       fatbin_opts[] = {"-c"};
    NVFATBIN_SAFE_CALL(nvFatbinCreate(&fatbin, fatbin_opts, 1));
    NVFATBIN_SAFE_CALL(nvFatbinAddLTOIR(fatbin, ltoir.data(), ltoir.size(), arch.c_str(),
                                        "scaling_callback", nullptr));
    size_t fatbin_size = 0;
    NVFATBIN_SAFE_CALL(nvFatbinSize(fatbin, &fatbin_size));
    std::vector<char> fatbin_data(fatbin_size);
    NVFATBIN_SAFE_CALL(nvFatbinGet(fatbin, fatbin_data.data()));
    nvFatbinDestroy(&fatbin);
    return fatbin_data;
}
#endif

template<class FFT, class IFFT>
double measure_cufftdx(const unsigned int&       kernel_runs,
                       const unsigned int&       warm_up_runs,
                       const unsigned int&       cuda_blocks,
                       typename FFT::value_type* device_buffer,
                       cudaStream_t              stream) {

    using namespace cufftdx;
    using complex_type = typename FFT::value_type;

    // create workspaces for FFT and IFFT
    cudaError_t error_code = cudaSuccess;
    auto        workspace  = make_workspace<FFT>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);
    auto workspace_inverse = make_workspace<IFFT>(error_code, stream);
    CUDA_CHECK_AND_EXIT(error_code);

    // run cuFFTDx
    double time = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            // There are (ffts_per_block * fft_size * cuda_blocks) elements
            convolution_kernel<FFT, IFFT><<<cuda_blocks, FFT::block_dim, FFT::shared_memory_size, stream>>>(
                device_buffer, workspace, workspace_inverse);
        },
        warm_up_runs, kernel_runs, stream);

    return time;
}

template<unsigned int fft_size, class Complex>
double measure_cufft(const unsigned int& kernel_runs,
                     const unsigned int& warm_up_runs,
                     const unsigned int& batch_size,
                     Complex*            device_buffer,
                     cudaStream_t        stream) {

    static constexpr unsigned int block_dim_scaling_kernel = 1024;
    static constexpr bool         is_double                = !std::is_same_v<Complex, cufftComplex>;
    static constexpr cufftType    plan_type                = is_double ? CUFFT_Z2Z : CUFFT_C2C;

    // Calculating parameters for scaling_kernel execution.
    // Get maximum number of running CUDA blocks per multiprocessor.
    int blocks_per_multiprocessor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      scaling_kernel<fft_size, Complex>,
                                                      block_dim_scaling_kernel,
                                                      0));

    // Get maximum number of CUDA blocks running on all multiprocessors.
    // This many CUDA blocks will be run for simple_kernel.
    const unsigned int cuda_blocks = blocks_per_multiprocessor * example::get_multiprocessor_count();

    const unsigned int input_length        = fft_size * batch_size;
    const unsigned int elements_per_block  = (input_length + cuda_blocks - 1) / cuda_blocks;
    const unsigned int elements_per_thread = (elements_per_block + block_dim_scaling_kernel - 1) / block_dim_scaling_kernel;

    // prepare cuFFT runs
    cufftHandle plan;
    CUFFT_CHECK_AND_EXIT(cufftPlan1d(&plan, fft_size, plan_type, batch_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan, stream));

    // run convolution
    double time_cufft = example::measure_execution_ms(
        [&](cudaStream_t stream) {
            cufftResult result;
            if constexpr (std::is_same_v<Complex, cufftComplex>) {
                result = cufftExecC2C(plan, device_buffer, device_buffer, CUFFT_FORWARD);
            } else {
                result = cufftExecZ2Z(plan, device_buffer, device_buffer, CUFFT_FORWARD);
            }
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT error: forward failed");
                return;
            }
            scaling_kernel<fft_size, Complex>
                <<<cuda_blocks, block_dim_scaling_kernel, 0, stream>>>(device_buffer, input_length, elements_per_thread);
            if constexpr (std::is_same_v<Complex, cufftComplex>) {
                result = cufftExecC2C(plan, device_buffer, device_buffer, CUFFT_INVERSE);
            } else {
                result = cufftExecZ2Z(plan, device_buffer, device_buffer, CUFFT_INVERSE);
            }
            if (result != CUFFT_SUCCESS) {
                fprintf(stderr, "CUFFT error: inverse failed");
                return;
            }
        },
        warm_up_runs, kernel_runs, stream);

    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan));
    return time_cufft;
}

#ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
template<unsigned int fft_size, class Complex>
double measure_cufft_callback(const unsigned int& kernel_runs,
                              const unsigned int& warm_up_runs,
                              const unsigned int& batch_size,
                              Complex*            device_buffer,
                              cudaStream_t        stream) {

    static constexpr unsigned int block_dim_scaling_kernel = 1024;
    static constexpr bool         is_double                = !std::is_same_v<Complex, cufftComplex>;
    static constexpr cufftType    plan_type                = is_double ? CUFFT_Z2Z : CUFFT_C2C;
    static constexpr cufftXtCallbackType callback_type = is_double ? CUFFT_CB_LD_COMPLEX_DOUBLE : CUFFT_CB_LD_COMPLEX;

    // Calculating parameters for scaling_kernel execution.
    // Get maximum number of running CUDA blocks per multiprocessor.
    int blocks_per_multiprocessor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      scaling_kernel<fft_size, Complex>,
                                                      block_dim_scaling_kernel,
                                                      0));
    // prepare cuFFT runs
    // Compile the scaling callback to LTO-IR
    int device = 0;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
    std::vector<char> ltoir = compile_scaling_callback<Complex>(fft_size, device);
    if (ltoir.empty()) {
        std::cout << "Failed to compile the cuFFT JIT callback" << std::endl;
        return 0.0;
    }

    // A JIT callback has to be registered on a created but not yet made plan, so
    // the extensible plan API is used: the callback is linked into the kernel
    // while the plan is being built.
    size_t      work_size = 0;
    cufftHandle plan_in;
    CUFFT_CHECK_AND_EXIT(cufftCreate(&plan_in));
    CUFFT_CHECK_AND_EXIT(cufftXtSetJITCallback(plan_in,
                                               "cufftdx_example_scaling_load",
                                               ltoir.data(),
                                               ltoir.size(),
                                               callback_type,
                                               nullptr));
    CUFFT_CHECK_AND_EXIT(cufftMakePlan1d(plan_in, fft_size, plan_type, batch_size, &work_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_in, stream));

    cufftHandle plan_out;
    CUFFT_CHECK_AND_EXIT(cufftCreate(&plan_out));
    CUFFT_CHECK_AND_EXIT(cufftMakePlan1d(plan_out, fft_size, plan_type, batch_size, &work_size));
    CUFFT_CHECK_AND_EXIT(cufftSetStream(plan_out, stream));

    // run convolution
    double time_cufft = example::measure_execution_ms(
        [&](cudaStream_t /* stream */) {
            cufftResult result_fwd;
            cufftResult result_inv;
            if constexpr (is_double) {
                result_fwd = cufftExecZ2Z(plan_in, device_buffer, device_buffer, CUFFT_FORWARD);
                result_inv = cufftExecZ2Z(plan_out, device_buffer, device_buffer, CUFFT_INVERSE);
            } else {
                result_fwd = cufftExecC2C(plan_in, device_buffer, device_buffer, CUFFT_FORWARD);
                result_inv = cufftExecC2C(plan_out, device_buffer, device_buffer, CUFFT_INVERSE);
            }
            CUFFT_CHECK_AND_EXIT(result_fwd);
            CUFFT_CHECK_AND_EXIT(result_inv);
        },
        warm_up_runs, kernel_runs, stream);

    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_in));
    CUFFT_CHECK_AND_EXIT(cufftDestroy(plan_out));
    return time_cufft;
}
#endif // CUFFTDX_EXAMPLES_CUFFT_CALLBACK

// This example compares performance of cuFFT and cuFFTDx when performing C2C convolution.
// Data is generated on host, copied to device buffer and processed by FFTs.
// Each cuFFTDx execution runs one kernel, each cuFFT execution - three kernels.
// The experiment runs with the following principles:
// - at least 1GB of data is allocated in GPU and transformed by both convolutions,
// - for cuFFTDx kernel run, number of CUDA blocks is divisible
//   by maximum number of CUDA blocks that can run simultaneously on the GPU.
template<unsigned int Arch>
int convolution(const bool validate_only) {
    using namespace cufftdx;

    static constexpr unsigned int minimum_input_size_bytes = (1 << 30); // At least one GB of data will be processed by FFTs.
    static constexpr unsigned int fft_size                 = CUFFTDX_CONV_FFT_SIZE;
    static constexpr unsigned int kernel_runs           = 10;
    static constexpr unsigned int warm_up_runs             = 1;
    static constexpr bool         verbose                  = true;

    static constexpr bool         use_suggested              = CUFFTDX_CONV_USE_SUGGESTED;
    static constexpr unsigned int custom_ffts_per_block      = CUFFTDX_CONV_FFTS_PER_BLOCK;
    static constexpr unsigned int custom_elements_per_thread = CUFFTDX_CONV_ELEMENTS_PER_THREAD;

    // To determine the total input length (number of fft batches to run), the maximum number of
    // simultanously running cuFFTDx CUDA blocks is calculated.

    // Declaration of cuFFTDx run
    using fft_incomplete = decltype(Block() + Size<fft_size>() + Type<fft_type::c2c>() + Precision<CUFFTDX_CONV_PRECISION>() + SM<Arch>());
    using fft_base       = decltype(fft_incomplete() + Direction<fft_direction::forward>());
    using ifft_base      = decltype(fft_incomplete() + Direction<fft_direction::inverse>());

    static constexpr unsigned int elements_per_thread = use_suggested ? fft_base::elements_per_thread : custom_elements_per_thread;
    static constexpr unsigned int ffts_per_block      = use_suggested ? fft_base::suggested_ffts_per_block : custom_ffts_per_block;

#if CUFFTDX_CONV_TRANSPOSE_BY_COMPONENT_PADDED
    using padded_transposition = experimental::Transposition<experimental::transposition::by_component_padded>;
    using fft = decltype(fft_base() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>() + padded_transposition());
    using ifft = decltype(ifft_base() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>() + padded_transposition());
#else
    using fft          = decltype(fft_base() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());
    using ifft         = decltype(ifft_base() + ElementsPerThread<elements_per_thread>() + FFTsPerBlock<ffts_per_block>());
#endif
    using complex_type = typename fft::value_type;

    int max_dynamic_shared_memory_size = 0;
    CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(
        &max_dynamic_shared_memory_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));

    if (validate_only) {
        const bool is_launch_feasible = fft::shared_memory_size <= static_cast<unsigned int>(max_dynamic_shared_memory_size);
        std::cout << "VALIDATE," << fft_size << ',' << fft::elements_per_thread << ',' << fft::ffts_per_block << ','
                  << fft::shared_memory_size << ',' << max_dynamic_shared_memory_size << ',' << is_launch_feasible << std::endl;
        return is_launch_feasible ? 0 : 2;
    }

    // Increase max shared memory if needed.
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        convolution_kernel<fft, ifft>, cudaFuncAttributeMaxDynamicSharedMemorySize, fft::shared_memory_size));

    // Get maximum number of running CUDA blocks per multiprocessor
    int blocks_per_multiprocessor = 0;
    CUDA_CHECK_AND_EXIT(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor,
                                                      convolution_kernel<fft, ifft>,
                                                      fft::block_dim.x * fft::block_dim.y * fft::block_dim.z,
                                                      fft::shared_memory_size));

    // Get maximum number of CUDA blocks running on all multiprocessors
    const unsigned int device_blocks = blocks_per_multiprocessor * example::get_multiprocessor_count();

    // Input size in bytes if device_blocks CUDA blocks were run.
    const unsigned int data_size_device_blocks_bytes = device_blocks * ffts_per_block * fft_size * sizeof(complex_type);

    // cuda_blocks = minimal number of CUDA blocks to run, such that:
    //   - cuda_blocks is divisible by device_blocks,
    //   - total input size is not less than minimum_input_size_bytes.
    // executed_blocks_multiplyer = cuda_blocks / device_blocks
    const unsigned int executed_blocks_multiplyer =
        (minimum_input_size_bytes + data_size_device_blocks_bytes - 1) / data_size_device_blocks_bytes;
    const unsigned int cuda_blocks  = device_blocks * executed_blocks_multiplyer;
    const unsigned int input_length = ffts_per_block * cuda_blocks * fft_size;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    // Host data
    std::vector<complex_type> input =
        example::get_random_complex_data<typename complex_type::value_type>(input_length, -10, 10);

    // Device data
    complex_type* device_buffer;
    auto          input_size_bytes = input.size() * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&device_buffer, input_size_bytes));

    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(device_buffer, input.data(), input_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Measure performance
    double time_cufftdx = measure_cufftdx<fft, ifft>(kernel_runs, warm_up_runs, cuda_blocks, device_buffer, stream);
    using cufft_complex_type = std::conditional_t<std::is_same_v<typename complex_type::value_type, float>, cufftComplex, cufftDoubleComplex>;
    double time_cufft   = measure_cufft<fft_size, cufft_complex_type>(
        kernel_runs, warm_up_runs, cuda_blocks * ffts_per_block, reinterpret_cast<cufft_complex_type*>(device_buffer), stream);
    #ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
    double time_cufft_cb = measure_cufft_callback<fft_size, cufft_complex_type>(
        kernel_runs, warm_up_runs, cuda_blocks * ffts_per_block, reinterpret_cast<cufft_complex_type*>(device_buffer), stream);
    #endif

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(device_buffer));

    // Report results.
    auto report_time_and_performance = [&](std::string name, double time) -> void {
        double gflops = 1.0 * kernel_runs * ffts_per_block * cuda_blocks * 5.0 * fft_size *
                        (std::log(fft_size) / std::log(2)) / time / 1000000.0;

        std::cout << std::endl;
        std::cout << name << std::endl;
        std::cout << "Avg Time [ms_n]: " << time / kernel_runs << std::endl;
        std::cout << "Time (all) [ms_n]: " << time << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    };

    if (verbose) {
        std::cout << "FFT size: " << fft_size << std::endl;
        std::cout << "FFTs run: " << ffts_per_block * cuda_blocks << std::endl;
        report_time_and_performance("cuFFTDx", time_cufftdx);
        std::cout << "FFTs elements per thread: " << fft::elements_per_thread << std::endl;
        std::cout << "FFTs per block: " << ffts_per_block << std::endl;
        std::cout << "CUDA blocks: " << cuda_blocks << std::endl;
        std::cout << "Blocks per multiprocessor: " << blocks_per_multiprocessor << std::endl;

        report_time_and_performance("cuFFT", time_cufft);
        #ifdef CUFFTDX_EXAMPLES_CUFFT_CALLBACK
        report_time_and_performance("cuFFT Callback", time_cufft_cb);
        #endif
    } else {
        double gflops_cufftdx = 1.0 * kernel_runs * ffts_per_block * cuda_blocks * 5.0 * fft_size *
                                (std::log(fft_size) / std::log(2)) / time_cufftdx / 1000000.0;
        double gflops_cufft = 1.0 * kernel_runs * ffts_per_block * cuda_blocks * 5.0 * fft_size *
                              (std::log(fft_size) / std::log(2)) / time_cufft / 1000000.0;
        std::cout << fft_size << ": " << std::endl
                  << gflops_cufftdx << ", " << time_cufftdx / kernel_runs << ", " << std::endl
                  << gflops_cufft << ", " << time_cufft / kernel_runs;
    }
    return 0;
}

bool convolution_validate_only = false;

template<unsigned int Arch>
struct convolution_functor {
    int operator()() {
        return convolution<Arch>(convolution_validate_only);
    }
};

int main(int argc, char** argv) {
    convolution_validate_only = argc == 2 && std::strcmp(argv[1], "--validate") == 0;
    return example::sm_runner<convolution_functor>();
}
