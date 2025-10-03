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
#include <vector>
#include <memory>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "nvrtc_helper.hpp" 
#include "../common.hpp"

const char* generate_kernel = R"kernel(
#include <curanddx.hpp>

// CURANDDX Operators 

using RNG = decltype(curanddx::Generator<curanddx::pcg>() + curanddx::SM<RNG_SM>() + curanddx::Thread());

extern "C" __global__ void generate_kernel(unsigned int*        d_out,
                                const unsigned long long        seed,
                                const typename RNG::offset_type offset,
                                const size_t                    size) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size)
        return;

    curanddx::uniform_bits<unsigned int> dist;

    // compare with NVPL RAND PCG strict ordering
    RNG rng(seed, 0, offset + tid);

    d_out[tid] = dist.generate(rng);

}
)kernel";


// This example demonstrates how to use NVRTC to runtime compile a kernel using CURANDDX functions and check correctness

int main(int, char**) {

    // Get current device
    int current_device;
    CU_CHECK_AND_EXIT(cudaGetDevice(&current_device));

    // Create NVRTC program
    nvrtcProgram program;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,           // program
                                       generate_kernel,      // buffer
                                       "generate_kernel.cu", // name
                                       0,                  // numHeaders
                                       NULL,               // headers
                                       NULL));             // includeNames

    // Prepare compilation options
    std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space",
        "--include-path=" CUDA_INCLUDE_DIR, // Add path to CUDA include directory
        "--include-path=" CUDA_CCCL_INCLUDE_DIR // Add path to CCCL include directory for CTK 13.0
    };

    // Parse cuRANDDx include dirs
    std::vector<std::string> curanddx_include_dirs = example::nvrtc::get_curanddx_include_dirs();
    // Add cuBLASDx include dirs to opts
    for (auto& d : curanddx_include_dirs) {
        opts.push_back(d.c_str());
    }

    // Add GPU_ARCHITECTURE definition to opts
    std::string gpu_architecture_definition =
        "-DRNG_SM=" + std::to_string(example::nvrtc::get_device_architecture(current_device) * 10);
    opts.push_back(gpu_architecture_definition.c_str());

    // Add gpu-architecture to opts
    std::string gpu_architecture_option = example::nvrtc::get_device_architecture_option(current_device);
    opts.push_back(gpu_architecture_option.c_str());

    nvrtcResult compileResult = nvrtcCompileProgram(program,                       // program
                                                    static_cast<int>(opts.size()), // numOptions
                                                    opts.data());                  // options

    // Obtain compilation log from the program
    if (compileResult != NVRTC_SUCCESS) {
        for (auto o : opts) {
            std::cout << o << std::endl;
        }
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

    // Load the generated Cubin and get a handle to the generate_kernel
    CUdevice   cuDevice;
    CUcontext  context;
    CUmodule   module;
    CUfunction kernel;
    CU_CHECK_AND_EXIT(cuInit(0));
    CU_CHECK_AND_EXIT(cuDeviceGet(&cuDevice, current_device));
#if CUDA_VERSION >= 13000
    CU_CHECK_AND_EXIT(cuCtxCreate(&context, (CUctxCreateParams*)0, 0, cuDevice));
#else
    CU_CHECK_AND_EXIT(cuCtxCreate(&context, 0, cuDevice));
#endif
    CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, cubin.get(), 0, 0, 0));
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "generate_kernel"));

    // Allocate output memory
    using DataType = unsigned int;

    CUdeviceptr   d_out;
    size_t size = 5000;
    CU_CHECK_AND_EXIT(cuMemAlloc(&d_out, size * sizeof(DataType)));

    unsigned long long seed   = 1234ULL;
    unsigned long long offset = 1ULL;

    // Invokes kernel
    const unsigned int block_dim = 256;
    const unsigned int grid_size = (size + block_dim - 1) / block_dim;

    // Execute generate_kernel
    void* args[] = {&d_out, &seed, &offset, &size};
    CU_CHECK_AND_EXIT(cuLaunchKernel(kernel,
                                     grid_size, // number of blocks
                                     1,
                                     1,
                                     block_dim, // number of threads
                                     1,
                                     1,
                                     0,    // shared memory size
                                     NULL, // NULL stream
                                     args,
                                     0));
    CU_CHECK_AND_EXIT(cuCtxSynchronize());

    // Copy to host
    std::vector<DataType> h_out(size);
    CU_CHECK_AND_EXIT(cuMemcpyDtoH(h_out.data(), d_out, size * sizeof(DataType)));
    CU_CHECK_AND_EXIT(cuMemFree(d_out));

    // Compare hash
    unsigned int xor_curand = 0x0;
    for (auto i = 0U; i < size; i++) {
        xor_curand ^= h_out[i];
    }
    if (xor_curand == 0xaa706742) {
        std::cout << "Compared to the hash value: Same sequence is generated with NVPL RAND and cuRANDDx generator "
                     "using STRICT ordering.\n";
        std::cout << "SUCCESS \n";
        return 0;
    } else {
        std::cout
            << "FAILED: different sequence is generated with NVPL RAND and cuRANDDx generator using STRICT ordering.\n";
        return -1;
    }
}