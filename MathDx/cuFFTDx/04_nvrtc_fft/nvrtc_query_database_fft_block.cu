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

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#define CUFFTDX_ENABLE_RUNTIME_DATABASE
#include <cufftdx/utils.hpp>

#include "../common/common.hpp"
#include "../common/common_nvrtc.hpp"

const char* test_kernel = R"kernel(
#include <cufftdx.hpp>

using namespace cufftdx;

// FFT Operators
using size_desc = Size<FFT_SIZE>;
using dir_desc  = Direction<fft_direction::inverse>;
using type_desc = Type<fft_type::c2c>;
using FFT = decltype(Block() + size_desc() + dir_desc() + type_desc() + Precision<double>() + SM<FFT_SM>() + ElementsPerThread<FFT_EPT>());

inline __device__ unsigned int batch_offset(const unsigned int local_fft_id,
                                            const unsigned int ffts_per_block = blockDim.y) {
    unsigned int global_fft_id = ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * ffts_per_block + local_fft_id);
    return cufftdx::size_of<FFT>::value * global_fft_id;
}

extern "C" __global__ void test_kernel(typename FFT::value_type* fft_data)
{
  typename FFT::value_type thread_data[FFT::storage_size];

  const unsigned int offset = batch_offset(threadIdx.y, FFT::ffts_per_block);
  constexpr unsigned int stride = FFT::stride;
  unsigned int index = offset + threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
        thread_data[i] = fft_data[index];
        index += stride;
    }
  }

  extern __shared__ FFT::value_type shared_mem[];
  FFT().execute(thread_data, shared_mem);

  index = offset + threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
        fft_data[index] = thread_data[i];
        index += stride;
    }
  }
}
)kernel";

int main(int, char**) {

    // Get current device
    int current_device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&current_device));

    // Note that FFT description is only defined in kernel
    // Precision: double
    static constexpr unsigned int fft_size = 64;
    static constexpr unsigned int fft_ept  = 8;

    // Complex type according to precision
    using value_type = cuDoubleComplex; // or double2, or cuda::std::complex<double>

    //query the database to get optimal block configuration and shared memory size
    auto result = cufftdx::experimental::utils::query_database(fft_size,
                                                 cufftdx::fft_direction::inverse,
                                                 cufftdx::fft_type::c2c,
                                                 example::nvrtc::get_device_architecture(current_device) * 10,
                                                 cufftdx::detail::execution_type::block,
                                                 cufftdx::precision::f64,
                                                 fft_ept);

    // Assign block configuration and shared memory size
    dim3     fft_block_dim {};
    unsigned shared_memory_size;
    if (result.has_value()) {
        fft_block_dim      = dim3(result.value().block_dim_x, result.value().block_dim_y, result.value().block_dim_z);
        shared_memory_size = result.value().shared_memory_size;
    } else {
        std::cout << "No database result found" << std::endl;
        exit(1);
    }


    nvrtcProgram program;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,         // program
                                       test_kernel,      // buffer
                                       "test_kernel.cu", // name
                                       0,                // numHeaders
                                       NULL,             // headers
                                       NULL));           // includeNames

    // Prepare compilation options
    std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space"};

    // Parse cuFFTDx include dirs
    std::vector<std::string> cufftdx_include_dirs = example::nvrtc::get_cufftdx_include_dirs();
    // Add cuFFTDx include dirs to opts
    for (auto& d : cufftdx_include_dirs) {
        opts.push_back(d.c_str());
    }

    // Example how to pass part of FFT description to NVRTC kernel
    // Add FFT_SIZE definition to opts
    std::string fft_size_definition = "-DFFT_SIZE=" + std::to_string(fft_size);
    opts.push_back(fft_size_definition.c_str());

    // Add FFT_EPT definition to opts
    std::string fft_ept_definition = "-DFFT_EPT=" + std::to_string(fft_ept);
    opts.push_back(fft_ept_definition.c_str());

    // Add GPU_ARCHITECTURE definition to opts
    std::string gpu_architecture_definition = "-DFFT_SM=" + std::to_string(example::nvrtc::get_device_architecture(current_device) * 10);
    opts.push_back(gpu_architecture_definition.c_str());

    // Add gpu-architecture to opts
    std::string gpu_architecture_option = example::nvrtc::get_device_architecture_option(current_device);
    opts.push_back(gpu_architecture_option.c_str());

    nvrtcResult compileResult = nvrtcCompileProgram(program,                       // program
                                                    static_cast<int>(opts.size()), // numOptions
                                                    opts.data());                  // options

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

    // Load the generated Cubin and get a handle to the test_kernel
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
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "test_kernel"));

    // Generate input for execution
    value_type* fft_data;
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&fft_data, fft_size*sizeof(value_type)));

    for(size_t i = 0; i < fft_size; i++) {
        fft_data[i].x = float(i);
        fft_data[i].y = 0;
    }

    // Print fft_data
    for (size_t i = 0; i < fft_size; ++i) {
        std::cout << i << ": (" << fft_data[i].x << ", " << fft_data[i].y << ")" << std::endl;
    }

    // Increase dynamic shared memory limit
    CU_CHECK_AND_EXIT(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_memory_size));

    // Execute test_kernel
    void* args[] = {&fft_data};
    CU_CHECK_AND_EXIT(cuLaunchKernel(kernel,
                                     1, // number of blocks
                                     1,
                                     1,
                                     fft_block_dim.x, // number of threads
                                     fft_block_dim.y,
                                     1,
                                     shared_memory_size,
                                     NULL, // NULL stream
                                     args,
                                     0));
    CU_CHECK_AND_EXIT(cuCtxSynchronize());

    // Retrieve and print output
    for (size_t i = 0; i < fft_size; ++i) {
        std::cout << i << ": (" << fft_data[i].x << ", " << fft_data[i].y << ")" << std::endl;
    }

    // Validate results before destroying context
    double sum = (fft_size * (fft_size - 1)) / 2.0;
    if (std::abs(fft_data[0].x - sum) > 0.01) {
        std::cout << "[nvrtc_fft_block] Failed" << std::endl;
        CUDA_CHECK_AND_EXIT(cudaFree(fft_data));
        CU_CHECK_AND_EXIT(cuModuleUnload(module));
        CU_CHECK_AND_EXIT(cuCtxDestroy(context));
        return 1;
    }
    std::cout << "[nvrtc_fft_block] Passed" << std::endl;

    // Release resources
    CUDA_CHECK_AND_EXIT(cudaFree(fft_data));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));
    CU_CHECK_AND_EXIT(cuCtxDestroy(context));
    return 0;
}
