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
#include <iomanip> // for std::setw and std::setprecision

#include <nvrtc.h>
#include <nvJitLink.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define CUFFTDX_ENABLE_RUNTIME_DATABASE
#include <cufftdx.hpp>

#include "../common/common.hpp"
#include "../common/common_nvrtc.hpp"
#include "../common/common_nvjitlink.hpp"

const char* fft_kernel_source = R"kernel(
using namespace cufftdx;

using size_desc = Size<FFT_SIZE>;
using dir_desc  = Direction<fft_direction::forward>;
using type_desc = Type<fft_type::c2c>;
using fpb_desc = FFTsPerBlock<FFT_FPB>;
using FFT = decltype(Block() + size_desc() + dir_desc() + type_desc() + Precision<float>() + SM<FFT_SM>() + ElementsPerThread<FFT_EPT>() + fpb_desc() + experimental::CodeType<experimental::code_type::ltoir>());

inline __device__ unsigned int batch_offset(const unsigned int local_fft_id,
                                            const unsigned int ffts_per_block = blockDim.y) {
    unsigned int global_fft_id = ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * ffts_per_block + local_fft_id);
    return cufftdx::size_of<FFT>::value * global_fft_id;
}

extern "C" __global__ void fft_kernel(typename FFT::value_type* fft_data)
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

struct autotune_result {
    unsigned int ept;
    unsigned int ffts_per_block;
    unsigned int block_dim_x;
    unsigned int block_dim_y;
    unsigned int block_dim_z;
    unsigned int shared_memory_size;
    float        kernel_ms;
};

// Compile kernel with NVRTC, link with nvJitLink, return cubin
std::vector<char> compile_and_link(const std::string&                    source,
                                   unsigned int                          fft_size,
                                   unsigned int                          fft_ept,
                                   unsigned int                          fft_fpb,
                                   unsigned int                          fft_sm,
                                   int                                   current_device,
                                   const std::vector<std::vector<char>>& ltoirs) {
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, source.c_str(), "fft_kernel.cu", 0, NULL, NULL));

    std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space"};

    std::vector<std::string> cufftdx_include_dirs = example::nvrtc::get_cufftdx_include_dirs();
    for (auto& d : cufftdx_include_dirs) {
        opts.push_back(d.c_str());
    }

    std::string def_size = "-DFFT_SIZE=" + std::to_string(fft_size);
    std::string def_ept  = "-DFFT_EPT=" + std::to_string(fft_ept);
    std::string def_fpb  = "-DFFT_FPB=" + std::to_string(fft_fpb);
    std::string def_sm   = "-DFFT_SM=" + std::to_string(fft_sm);
    opts.push_back(def_size.c_str());
    opts.push_back(def_ept.c_str());
    opts.push_back(def_fpb.c_str());
    opts.push_back(def_sm.c_str());

    std::string arch_opt = example::nvrtc::get_device_architecture_option(current_device);
    opts.push_back(arch_opt.c_str());
    opts.push_back("-dlto");
    opts.push_back("--relocatable-device-code=true");

    nvrtcResult compileResult = nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
    if (compileResult != NVRTC_SUCCESS) {
        example::nvrtc::print_program_log(prog);
        nvrtcDestroyProgram(&prog);
        return {};
    }

    size_t ltoir_size;
    NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &ltoir_size));
    std::vector<char> ltoir(ltoir_size);
    NVRTC_SAFE_CALL(nvrtcGetLTOIR(prog, ltoir.data()));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    nvJitLinkHandle          handle;
    std::vector<const char*> link_opts;
    link_opts.push_back("-dlto");
    std::string jitlink_arch = example::nvjitlink::get_device_architecture_option(current_device);
    link_opts.push_back(jitlink_arch.c_str());

    NVJITLINK_SAFE_CALL(handle, nvJitLinkCreate(&handle, static_cast<int>(link_opts.size()), link_opts.data()));
    NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_ANY, ltoir.data(), ltoir_size, "nvrtc_ltoir"));
    for (unsigned i = 0; i < ltoirs.size(); i++) {
        NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_ANY, ltoirs[i].data(), ltoirs[i].size(), "cufft_ltoir"));
    }

    NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));

    size_t cubin_size;
    NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubinSize(handle, &cubin_size));
    std::vector<char> cubin(cubin_size);
    NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubin(handle, cubin.data()));
    NVJITLINK_SAFE_CALL(handle, nvJitLinkDestroy(&handle));

    return cubin;
}

// Load cubin, launch kernel with timing, return average kernel time in ms
float benchmark_kernel(const std::vector<char>& cubin,
                       int                      current_device,
                       unsigned int             fft_size,
                       unsigned int             ffts_per_block,
                       dim3                     block_dim,
                       unsigned int             shared_memory_size,
                       unsigned int             warmup_runs,
                       unsigned int             timed_runs) {
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
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "fft_kernel"));

    CU_CHECK_AND_EXIT(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_memory_size));

    const size_t total_size = fft_size * ffts_per_block;
    float2* fft_data;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&fft_data, total_size * sizeof(float2)));

    std::vector<float2> host_data(total_size);
    for (unsigned b = 0; b < ffts_per_block; b++) {
        for (unsigned i = 0; i < fft_size; i++) {
            host_data[b * fft_size + i] = {static_cast<float>(i), 0.0f};
        }
    }
    CUDA_CHECK_AND_EXIT(cudaMemcpy(fft_data, host_data.data(), total_size * sizeof(float2), cudaMemcpyHostToDevice));

    void* args[] = {&fft_data};

    for (unsigned i = 0; i < warmup_runs; i++) {
        CU_CHECK_AND_EXIT(cuLaunchKernel(kernel, 1, 1, 1, block_dim.x, block_dim.y, block_dim.z, shared_memory_size, NULL, args, 0));
    }
    CU_CHECK_AND_EXIT(cuCtxSynchronize());

    CUevent start, stop;
    CU_CHECK_AND_EXIT(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CU_CHECK_AND_EXIT(cuEventCreate(&stop, CU_EVENT_DEFAULT));

    CU_CHECK_AND_EXIT(cuEventRecord(start, NULL));
    for (unsigned i = 0; i < timed_runs; i++) {
        CU_CHECK_AND_EXIT(cuLaunchKernel(kernel, 1, 1, 1, block_dim.x, block_dim.y, block_dim.z, shared_memory_size, NULL, args, 0));
    }
    CU_CHECK_AND_EXIT(cuEventRecord(stop, NULL));
    CU_CHECK_AND_EXIT(cuEventSynchronize(stop));

    float total_ms = 0.0f;
    CU_CHECK_AND_EXIT(cuEventElapsedTime(&total_ms, start, stop));

    CU_CHECK_AND_EXIT(cuEventDestroy(start));
    CU_CHECK_AND_EXIT(cuEventDestroy(stop));
    CUDA_CHECK_AND_EXIT(cudaFree(fft_data));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));
    CU_CHECK_AND_EXIT(cuCtxDestroy(context));

    return total_ms / static_cast<float>(timed_runs);
}

int main(int, char**) {
    static constexpr unsigned int fft_size    = 128;
    static constexpr unsigned int warmup_runs = 5;
    static constexpr unsigned int timed_runs  = 5;

    int current_device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&current_device));

    if (!cufftdx::utils::check_cufft_device_api_version()) {
        return 1;
    }

    unsigned int sm = example::nvrtc::get_device_architecture(current_device) * 10;

    std::cout << "=== cuFFTDx LTO Autotuning ===" << std::endl;
    std::cout << "FFT: size=" << fft_size << ", C2C, forward, f32, block" << std::endl;
    std::cout << "Device SM: " << sm << std::endl;
    std::cout << std::endl;

    std::cout << "Querying cuFFT device API for available implementations..." << std::endl;
    auto impls = cufftdx::experimental::utils::get_all_implementations(
        fft_size,
        cufftdx::fft_direction::forward,
        cufftdx::fft_type::c2c,
        sm,
        cufftdx::utils::execution_type::block,
        cufftdx::precision::f32,
        0,
        0,
        {0, 0, 0},
        cufftdx::complex_layout::natural,
        cufftdx::real_mode::normal,
        cufftdx::experimental::query_code_type::ltoir_online);

    if (impls.empty()) {
        std::cerr << "No implementations found for this configuration." << std::endl;
        return 1;
    }

    std::cout << "Found " << impls.size() << " implementation(s):" << std::endl;
    for (unsigned i = 0; i < impls.size(); ++i) {
        const auto& impl = impls[i];
        std::cout << "  [" << i << "]"
                  << " ept=" << impl.elements_per_thread
                  << " block_dim=(" << impl.block_dim_x << "," << impl.block_dim_y << "," << impl.block_dim_z << ")"
                  << " shared_mem=" << impl.shared_memory_size
                  << " storage_size=" << impl.storage_size
                  << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Benchmarking each implementation (" << warmup_runs << " warmup, " << timed_runs << " timed runs):" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    std::vector<autotune_result> results;

    for (unsigned i = 0; i < impls.size(); ++i) {
        const auto&  impl = impls[i];
        unsigned int ept  = impl.elements_per_thread;
        unsigned int fpb  = impl.ffts_per_block;
        dim3         bdim(impl.block_dim_x, impl.block_dim_y, impl.block_dim_z);

        // unused block_dim and shared memory as it comes from query_all_cufft_implementations
        auto [lto_db, ltoirs, _0, _1] =
            cufftdx::utils::get_database_and_ltoir(
                fft_size,
                cufftdx::fft_direction::forward,
                cufftdx::fft_type::c2c,
                sm,
                cufftdx::utils::execution_type::block,
                cufftdx::precision::f32,
                cufftdx::complex_layout::natural,
                cufftdx::real_mode::normal,
                ept);

        if (ltoirs.empty() || lto_db.empty()) {
            std::cout << "  [" << i << "] ept=" << ept << "  SKIPPED (no LTOIR/database)" << std::endl;
            continue;
        }

        std::string source;
        source.append("#include <cufftdx.hpp>\n");
        source.append(lto_db);
        source.append(fft_kernel_source);

        auto cubin = compile_and_link(source, fft_size, ept, fpb, sm, current_device, ltoirs);
        if (cubin.empty()) {
            std::cout << "  [" << i << "] ept=" << ept << "  FAILED (compile/link error)" << std::endl;
            continue;
        }

        float kernel_ms = benchmark_kernel(cubin, current_device, fft_size, fpb, bdim, impl.shared_memory_size, warmup_runs, timed_runs);

        results.push_back({ept, fpb, bdim.x, bdim.y, bdim.z, impl.shared_memory_size, kernel_ms});

        std::cout << "  [" << i << "]"
                  << " ept=" << std::setw(3) << ept
                  << "  kernel=" << std::fixed << std::setprecision(4) << std::setw(10) << kernel_ms << "ms"
                  << std::endl;
    }

    std::cout << std::string(55, '-') << std::endl;

    if (results.empty()) {
        std::cerr << "No implementations could be benchmarked." << std::endl;
        return 1;
    }

    auto best = std::min_element(results.begin(), results.end(), [](const autotune_result& a, const autotune_result& b) { return a.kernel_ms < b.kernel_ms; });

    std::cout << std::endl;
    std::cout << "Best: ept=" << best->ept
              << ", fpb=" << best->ffts_per_block
              << ", block_dim=(" << best->block_dim_x << "," << best->block_dim_y << "," << best->block_dim_z << ")"
              << ", shared_mem=" << best->shared_memory_size
              << ", kernel=" << std::fixed << std::setprecision(4) << best->kernel_ms << "ms"
              << std::endl;

    return 0;
}
