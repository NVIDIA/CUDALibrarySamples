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

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../common/common.hpp"
#include "../common/common_nvrtc.hpp"

const char* thread_fft_kernel = R"kernel(
#include <cufftdx.hpp>

using namespace cufftdx;

// FFT
using size_desc  = Size<FFT_SIZE>;
using dir_desc   = Direction<fft_direction::inverse>;
using type_desc  = Type<fft_type::c2c>;
using FFT        = decltype(size_desc() + dir_desc() + type_desc() + Thread() + Precision<double>());

extern "C" __global__ void thread_fft_kernel(typename FFT::value_type *data)
{
    // Local array for thread
    typename FFT::value_type thread_data[FFT::storage_size];

    // Load data from global memory to registers.
    // thread_data should have all fft_data data in order.
    unsigned int index = threadIdx.x * FFT::elements_per_thread;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = data[index + i];
    }

    // Execute FFT
    FFT().execute(thread_data);

    // Save results
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        data[index + i] = thread_data[i];
    }
}
)kernel";

int main(int, char**) {
    // Note that FFT description is only defined in kernel
    static constexpr unsigned int fft_size = 16;
    using value_type = double2; // or cuDoubleComplex, or cuda::std::complex<double>

    // Add FFT_SIZE definition to opts
    std::string fft_size_definition = "-DFFT_SIZE=" + std::to_string(fft_size);

    // Get current device
    int current_device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&current_device));

    // Get architecture of current device
    std::string gpu_architecture_option = example::nvrtc::get_device_architecture_option(current_device);

    // Create a program
    nvrtcProgram program;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,               // program
                                       thread_fft_kernel,      // buffer
                                       "thread_fft_kernel.cu", // name
                                       0,                      // numHeaders
                                       NULL,                   // headers
                                       NULL));                 // includeNames

    // Prepare compilation options
    std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space"
    };
    // Parse cuFFTDx include dirs
    std::vector<std::string> cufftdx_include_dirs = example::nvrtc::get_cufftdx_include_dirs();
    // Add cuFFTDx include dirs to opts
    for (auto& d : cufftdx_include_dirs) {
        opts.push_back(d.c_str());
    }
    // Add FFT_SIZE definition
    opts.push_back(fft_size_definition.c_str());
    // Add gpu-architecture flag
    opts.push_back(gpu_architecture_option.c_str());

    nvrtcResult compileResult = nvrtcCompileProgram(program,      // program
                                                    static_cast<int>(opts.size()),  // numOptions
                                                    opts.data()); // options

    // Obtain compilation log from the program
    if (compileResult != NVRTC_SUCCESS) {
        example::nvrtc::print_program_log(program);
        std::exit(1);
    }

    // Obtain cubin from the program.
    size_t cubin_size;
    NVRTC_SAFE_CALL(nvrtcGetCUBINSize(program, &cubin_size));
    auto cubin = std::make_unique<char[]>(cubin_size);
    NVRTC_SAFE_CALL(nvrtcGetCUBIN(program, cubin.get()));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

    // Load the generated PTX and get a handle to the thread_fft_kernel
    CUdevice   cuDevice;
    CUcontext  context;
    CUmodule   module;
    CUfunction kernel;
    CU_CHECK_AND_EXIT(cuInit(0));
    CU_CHECK_AND_EXIT(cuDeviceGet(&cuDevice, current_device));
    #if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
        CU_CHECK_AND_EXIT(cuCtxCreate(&context, (CUctxCreateParams*)0, 0, cuDevice));
    #else
        CU_CHECK_AND_EXIT(cuCtxCreate(&context, 0, cuDevice));
    #endif
    CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, cubin.get(), 0, 0, 0));
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "thread_fft_kernel"));

    // Generate input for execution
    value_type* fft_data;
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&fft_data, fft_size * sizeof(value_type)));

    for(size_t i = 0; i < fft_size; i++) {
        fft_data[i].x = float(i);
        fft_data[i].y = 0;
    }

    // Execute thread_fft_kernel
    void* args[] = {&fft_data};
    CU_CHECK_AND_EXIT(cuLaunchKernel(kernel,
                                       1, // number of blocks
                                       1,
                                       1,
                                       1, // number of threads
                                       1,
                                       1,
                                       0,    // no shared memory
                                       NULL, // NULL stream
                                       args,
                                       0));
    CU_CHECK_AND_EXIT(cuCtxSynchronize());

    // Retrieve and print output
    for (size_t i = 0; i < fft_size; ++i) {
        std::cout << i << ": (" << fft_data[i].x << ", " << fft_data[i].y << ")" << std::endl;
    }

    // Validate results before destroying context
    double expected_value = (fft_size * (fft_size - 1)) / 2;
    if (std::abs(fft_data[0].x - expected_value) > 0.01) {
        std::cout << "Failed" << std::endl;
        CUDA_CHECK_AND_EXIT(cudaFree(fft_data));
        CU_CHECK_AND_EXIT(cuModuleUnload(module));
        return 1;
    }
    std::cout << "Success" << std::endl;

    // Release resources
    CUDA_CHECK_AND_EXIT(cudaFree(fft_data));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));
    return 0;
}
