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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <nvrtc.h>
#include <cuda.h>
#include <nvJitLink.h>
#include <cuda_runtime_api.h>

#include "common_nvrtc.hpp"
#include "nvrtc_cublas_reference.hpp"

const char* test_kernel = R"kernel(
#include <cublasdx.hpp>

using namespace cublasdx;

// BLAS Operators
using size_desc = Size<BLAS_M, BLAS_N>;
using type_desc = Type<type::real>;
using arr_desc  = Arrangement<cublasdx::col_major, cublasdx::col_major>;
using BLAS      = decltype(Block() + Function<function::TRSM>() + size_desc() + type_desc() + arr_desc() +
                      Precision<float>() + Side<side::left>() + FillMode<fill_mode::lower>() +
                      Diag<diag::non_unit>() + SM<BLAS_SM>());

__constant__ dim3         blas_block_dim     = BLAS::block_dim;
__constant__ unsigned int blas_shared_memory = cublasdx::get_shared_storage_size_ab<BLAS>();

extern "C" __global__ void test_kernel(const typename BLAS::a_value_type* a, typename BLAS::b_value_type* b)
{
    CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(BLAS);
    extern __shared__ __align__(16) cublasdx::byte smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());

    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<BLAS>(smem);
    auto a_shared_tensor  = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor  = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    BLAS().execute(a_shared_tensor, b_shared_tensor);
    __syncthreads();

    cublasdx::copy<BLAS, alignment::b>(b_shared_tensor, b_global_tensor);
}
)kernel";

int main(int, char**) {
    // Note that BLAS description is only defined in kernel
    // Precision: float
    // Type: Real
    // Side: Left
    // FillMode: Lower
    // Diag: Non-unit
    unsigned int blas_m = 32;
    unsigned int blas_n = 32;

    nvrtcProgram program;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,         // program
                                       test_kernel,      // buffer
                                       "test_kernel.cu", // name
                                       0,                // numHeaders
                                       NULL,             // headers
                                       NULL));           // includeNames

    int current_device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&current_device));

    std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space",
    };

    std::vector<std::string> cublasdx_include_dirs = example::nvrtc::get_cublasdx_include_dirs();
    for (auto& d : cublasdx_include_dirs) {
        opts.push_back(d.c_str());
    }

    std::string blas_m_definition = "-DBLAS_M=" + std::to_string(blas_m);
    opts.push_back(blas_m_definition.c_str());
    std::string blas_n_definition = "-DBLAS_N=" + std::to_string(blas_n);
    opts.push_back(blas_n_definition.c_str());

    std::string gpu_architecture_definition =
        "-DBLAS_SM=" + std::to_string(example::nvrtc::get_device_architecture(current_device) * 10);
    opts.push_back(gpu_architecture_definition.c_str());

    std::string gpu_architecture_option = example::nvrtc::get_device_architecture_option(current_device);
    opts.push_back(gpu_architecture_option.c_str());

    opts.push_back("-dlto");
    opts.push_back("--relocatable-device-code=true");

    nvrtcResult compileResult = nvrtcCompileProgram(program,
                                                    static_cast<int>(opts.size()),
                                                    opts.data());
    if (compileResult != NVRTC_SUCCESS) {
        for (auto o : opts) {
            std::cout << o << std::endl;
        }
        example::nvrtc::print_program_log(program);
        std::exit(1);
    }

    size_t lto_size;
    NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(program, &lto_size));
    auto ltoir = std::make_unique<char[]>(lto_size);
    NVRTC_SAFE_CALL(nvrtcGetLTOIR(program, ltoir.get()));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

    CUdevice   cuDevice;
    CUcontext  context;
    CUmodule   module;
    CUfunction kernel;
    CU_CHECK_AND_EXIT(cuInit(0));
    CU_CHECK_AND_EXIT(cuDeviceGet(&cuDevice, current_device));
    CU_CHECK_AND_EXIT(cuCtxCreate(&context, (CUctxCreateParams*)0, 0, cuDevice));

    nvJitLinkHandle linker = nullptr;
    std::string     nvjitlink_gpu_architecture_option = example::nvjitlink::get_device_architecture_option(current_device);
    const char*     link_opts[]                       = {"-lto", nvjitlink_gpu_architecture_option.c_str()};
    NVJITLINK_SAFE_CALL(linker, nvJitLinkCreate(&linker, 2, link_opts));

    const char* fatbin_env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUBLASDX_FATBIN");
    if (fatbin_env_ptr != nullptr) {
        NVJITLINK_SAFE_CALL(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, fatbin_env_ptr));
    } else {
#ifdef CUBLASDX_FATBIN
        NVJITLINK_SAFE_CALL(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, CUBLASDX_FATBIN));
#else
        std::cout << "Please set CUBLASDX_EXAMPLE_CUBLASDX_FATBIN env or define CUBLASDX_FATBIN\n";
        return 1;
#endif
    }
    NVJITLINK_SAFE_CALL(linker, nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, ltoir.get(), lto_size, "test_kernel_ltoir"));
    NVJITLINK_SAFE_CALL(linker, nvJitLinkComplete(linker));

    size_t cubin_size;
    NVJITLINK_SAFE_CALL(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    auto cubin = std::make_unique<char[]>(cubin_size);
    NVJITLINK_SAFE_CALL(linker, nvJitLinkGetLinkedCubin(linker, cubin.get()));
    NVJITLINK_SAFE_CALL(linker, nvJitLinkDestroy(&linker));

    CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, cubin.get(), 0, 0, 0));
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "test_kernel"));

    const size_t blas_a_size = blas_m * blas_m;
    const size_t blas_b_size = blas_m * blas_n;

    using a_value_type = float;
    using b_value_type = float;

    auto host_a = example::nvrtc::generate_random_data(blas_a_size, -4.0, 4.0);
    auto host_b = example::nvrtc::generate_random_data(blas_b_size, -1.0, 1.0);

    example::nvrtc::make_lower_col_major_diagonal_dominant(host_a, blas_m);
    auto host_b_ref = example::nvrtc::cublas_reference::left_lower_trsm(host_a, host_b, blas_m, blas_n);

    const size_t blas_a_size_bytes = blas_a_size * sizeof(a_value_type);
    const size_t blas_b_size_bytes = blas_b_size * sizeof(b_value_type);

    CUdeviceptr device_a;
    CUdeviceptr device_b;
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_a, blas_a_size_bytes));
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_b, blas_b_size_bytes));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(device_a, host_a.data(), blas_a_size_bytes));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(device_b, host_b.data(), blas_b_size_bytes));

    dim3         blas_block_dim = example::nvrtc::get_global_from_module<dim3>(module, "blas_block_dim");
    unsigned int blas_shared_memory =
        example::nvrtc::get_global_from_module<unsigned int>(module, "blas_shared_memory");

    CU_CHECK_AND_EXIT(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, blas_shared_memory));

    void* args[] = {&device_a, &device_b};
    CU_CHECK_AND_EXIT(cuLaunchKernel(kernel,
                                     1,
                                     1,
                                     1,
                                     blas_block_dim.x,
                                     blas_block_dim.y,
                                     blas_block_dim.z,
                                     blas_shared_memory,
                                     NULL,
                                     args,
                                     0));
    CU_CHECK_AND_EXIT(cuCtxSynchronize());

    CU_CHECK_AND_EXIT(cuMemcpyDtoH(host_b.data(), device_b, blas_b_size_bytes));

    CU_CHECK_AND_EXIT(cuMemFree(device_a));
    CU_CHECK_AND_EXIT(cuMemFree(device_b));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));
    CU_CHECK_AND_EXIT(cuCtxDestroy(context));

    if (!example::nvrtc::check_float_result(host_b, host_b_ref)) {
        std::cout << "Failure: results do not match cuBLAS reference" << std::endl;
        return 1;
    }

    std::cout << "Success" << std::endl;
    return 0;
}
