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

#include <vector>
#include <iostream>

#include <nvrtc.h>
#include <nvJitLink.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cufftdx.hpp>

#include "../common/common.hpp"
#include "../common/common_nvrtc.hpp"
#include "../common/common_nvjitlink.hpp"

const char* test_kernel = R"kernel(
using namespace cufftdx;

// FFT Operators
using size_desc = Size<FFT_SIZE>;
using dir_desc  = Direction<fft_direction::inverse>;
using type_c2c  = Type<fft_type::c2c>;
using FFT = decltype(size_desc() + dir_desc() + type_c2c() + Precision<double>() + experimental::CodeType<experimental::code_type::ltoir>() + Thread());

extern "C" __global__ void test_kernel(typename FFT::value_type *values)
{
  typename FFT::value_type regs[FFT::storage_size];

  static constexpr auto size = cufftdx::size_of<FFT>::value;
  for (size_t i = 0; i < size; i++) {
      regs[i] = values[i];
  }

  FFT().execute(regs);

  for (size_t i = 0; i < size; i++) {
      values[i] = regs[i];
  }
}
)kernel";


int main(int, char**) {
    // Note that FFT description is only defined in kernel
    static constexpr unsigned int fft_size = 16;
    using value_type = double2; // or cuDoubleComplex, or cuda::std::complex<double>

    // Get current device
    int current_device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&current_device));

    // Get LTOIR and database from cuFFT
    auto [lto_db, ltoirs, block_dim, shared_memory_size] =
        cufftdx::utils::get_database_and_ltoir(fft_size,
                                               cufftdx::fft_direction::inverse,
                                               cufftdx::fft_type::c2c,
                                               example::nvrtc::get_device_architecture(current_device) * 10,
                                               cufftdx::utils::execution_type::thread,
                                               cufftdx::precision::f64);

    if (ltoirs.size() == 0 || lto_db.empty()) {
        std::cout << "LTOIR or database from cuFFT is empty. Exit.\n";
        return 1;
    }

    // Build CUDA source code. Plug in cuFFT-dumped database file after cufftdx.hpp inclusion.
    std::string dx_source_code;
    dx_source_code.append("#include <cufftdx.hpp>\n");
    dx_source_code.append(lto_db);
    dx_source_code.append(test_kernel);

    // Create a program
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,                  // prog
                                       dx_source_code.c_str(), // buffer
                                       "test_kernel.cu",       // name
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

    // Add FFT_SIZE definition to opts
    std::string fft_size_definition = "-DFFT_SIZE=" + std::to_string(fft_size);
    opts.push_back(fft_size_definition.c_str());

    // Add gpu-architecture to opts
    std::string nvrtc_gpu_architecture_option = example::nvrtc::get_device_architecture_option(current_device);
    opts.push_back(nvrtc_gpu_architecture_option.c_str());

    // Add flags for LTO compilation
    opts.push_back("-dlto");
    opts.push_back("--relocatable-device-code=true");

    const auto compile_start = std::chrono::high_resolution_clock::now();
    nvrtcResult compileResult = nvrtcCompileProgram(prog,         // prog
                                                    static_cast<int>(opts.size()),  // numOptions
                                                    opts.data()); // options
    std::cout << "NVRTC compile time (ms): " << (double) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - compile_start).count() / 1e3 << "\n";

    // Obtain compilation log from the program
    if (compileResult != NVRTC_SUCCESS) {
        example::nvrtc::print_program_log(prog);
        std::exit(1);
    }

    // Obtain LTOIR from the program.
    size_t ltoir_size;
    NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &ltoir_size));
    auto ltoir = std::vector<char>(ltoir_size);
    NVRTC_SAFE_CALL(nvrtcGetLTOIR(prog, ltoir.data()));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    // Load the generated LTOIR and the cuFFT-dumped LTOIRs and link them together
    nvJitLinkHandle handle;
    std::vector<const char*> link_opts;
    link_opts.push_back("-dlto");

    std::string nvjitlink_gpu_architecture_option = example::nvjitlink::get_device_architecture_option(current_device);
    link_opts.push_back(nvjitlink_gpu_architecture_option.c_str());

    NVJITLINK_SAFE_CALL(handle, nvJitLinkCreate(&handle, static_cast<int>(link_opts.size()), link_opts.data()));
    NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_ANY, ltoir.data(), ltoir_size, "nvrtc_ltoir"));
    for (unsigned i = 0; i < ltoirs.size(); i++) {
        NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_ANY, ltoirs[i].data(), ltoirs[i].size(), "cufft_generated_ltoir"));
    }
    const auto link_start = std::chrono::high_resolution_clock::now();
    NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));
    std::cout << "NVJITLINK link time (ms): " << (double) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - link_start).count() / 1e3 << "\n";

    // Obtain linked Cubin
    size_t cubin_size;
    NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
    auto cubin = std::vector<char>(cubin_size);
    NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubin(handle, cubin.data()));
    NVJITLINK_SAFE_CALL(handle, nvJitLinkDestroy(&handle));

    // Load the Cubin and get a handle to the test_kernel
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
    CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, cubin.data(), 0, 0, 0));
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "test_kernel"));

    // Generate input for execution
    std::vector<value_type> host_input(fft_size);
    float                   i = 0.0f;
    for (auto& v : host_input) {
        v.x = i++;
        v.y = 0;
    }
    size_t fft_buffer_size = fft_size * sizeof(value_type);

    CUdeviceptr device_values;
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_values, fft_buffer_size));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(device_values, host_input.data(), fft_buffer_size));

    // Execute test_kernel
    void* args[] = {&device_values};
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

    // Retrieve and print output.
    std::vector<value_type> host_output(fft_size);
    CU_CHECK_AND_EXIT(cuMemcpyDtoH(host_output.data(), device_values, fft_buffer_size));
    // for (size_t i = 0; i < fft_size; ++i) {
    //     std::cout << i << ": (" << host_output[i].x << ", " << host_output[i].y << ")" << std::endl;
    // }

    // Release resources.
    CU_CHECK_AND_EXIT(cuMemFree(device_values));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));
    CU_CHECK_AND_EXIT(cuCtxDestroy(context));

    if ((host_output[0].x - 120.0 > 0.01)) {
        std::cout << "[test_nvrtc_lto_thread] Failed" << std::endl;
        return 1;
    }
    std::cout << "[test_nvrtc_lto_thread] Passed" << std::endl;
    return 0;
}
