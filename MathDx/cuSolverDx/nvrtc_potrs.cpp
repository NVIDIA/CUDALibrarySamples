// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>
#include <chrono>

#include <nvrtc.h>
#include <cuda.h>
#include <nvJitLink.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include "nvrtc_helper.hpp" // defines CUSOLVERDX_EXAMPLE_NVRTC
#include "common/random.hpp"

// This example demonstrates how to use cuSolverDx functions with NVRTC to runtime compile a kernel into LTO IR. 
// Then nvJitLInk is used to link the generated LTO IR with cuSolverDx's LTO library, conduct optimization on the linked LTO IR, 
// and generate cubin for the defined GPU architecture. 

const char* test_kernel = R"kernel(
#include <cusolverdx.hpp>

using namespace cusolverdx;

// CUSOLVERDX Operators with inputs

using Base =
        decltype(Size<M_SIZE, M_SIZE>() + Precision<double>() + Type<type::complex>() + Block() +
                 LeadingDimension<SOLVER_LDA>() + SM<SOLVER_SM>() + BatchesPerBlock<1>());
using POTRF = decltype(Base() + Function<potrf>());
using POTRS = decltype(Base() + Function<potrs>());

__constant__ dim3         solver_block_dim     = POTRF::block_dim;
__constant__ unsigned int solver_shared_memory_size = POTRS::shared_memory_size;

extern "C" __global__ void potrs_kernel(typename POTRS::a_data_type* A, typename POTRS::a_data_type* B, typename POTRF::status_type* info)
{
    static_assert(POTRS::batches_per_block == 1, "the kernel is written for bpb==1");

    extern __shared__ unsigned char smem[];

    using DataType = typename POTRS::a_data_type;

    constexpr auto m = POTRS::m_size;
    constexpr auto lda_smem = POTRF::lda;
    constexpr auto lda_gmem = m;
    constexpr auto one_batch_size_a_smem = lda_smem * m;
    constexpr auto one_batch_size_a_gmem = lda_gmem * m;
    constexpr auto one_batch_size_b_gmem = lda_gmem;

    const auto batch_idx = blockIdx.x;
    const auto tid = threadIdx.x;
    const auto nthreads = POTRF::block_dim.x;

    DataType* As = reinterpret_cast<DataType*>(smem);
    DataType* Bs = As + one_batch_size_a_smem;
    auto Ag = A + size_t(one_batch_size_a_gmem) * batch_idx;
    auto Bg = B + size_t(one_batch_size_b_gmem) * batch_idx;

	// Load from gmem to smem
    for (int k = 0; k < (m * m); k += nthreads) {
        if (k + tid < m * m) {
            int r            = (k + tid) % m;
            int c            = (k + tid) / m;
            As[r + c * lda_smem] = Ag[r + c * lda_gmem];
        }
    }
    for (int k = tid; k < m; k += nthreads) {
        Bs[k] = Bg[k];
    }

    __syncthreads();

    POTRF().execute(As, info);
    POTRS().execute(As, Bs);
    __syncthreads();

    // Store from smem to gmem
    for (int k = 0; k < (m * m); k += nthreads) {
        if (k + tid < m * m) {
            int r          = (k + tid) % m;
            int c          = (k + tid) / m;
            Ag[r + c * lda_gmem] = As[r + c * lda_smem];
        }
    }
    for (int k = tid; k < m; k += nthreads) {
        Bg[k] = Bs[k];
    }

}
)kernel";


int main(int, char**) {

    // Get current device
    int current_device;
    CU_CHECK_AND_EXIT(cudaGetDevice(&current_device));

    // CUSOLVERDX sizes
    static constexpr unsigned int m_size   = 32;
    static constexpr unsigned int lda_smem = 33;
    const unsigned int            batches  = 2;

    using data_type = cuDoubleComplex;

    // Create an instance of nvrtcProgram with the code string.
    nvrtcProgram program;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,          // program
                                       test_kernel,       // buffer
                                       "potrs_kernel.cu", // name
                                       0,                 // numHeaders
                                       NULL,              // headers
                                       NULL));            // includeNames

    // Prepare compilation options
    std::vector<const char*> opts = {
        "--std=c++17",
        "--device-as-default-execution-space",
        // specify that LTO IR should be generated for LTO operation
        "-dlto",
        "--relocatable-device-code=true",
        "--include-path=" CUDA_INCLUDE_DIR // Add path to CUDA include directory
    };

    // Parse cuSOLVERDx include dirs
    std::vector<std::string> solver_include_dirs = common::nvrtc::get_solver_include_dirs();
    // Add cuSOLVERDx include dirs to opts
    for (auto& d : solver_include_dirs) {
        opts.push_back(d.c_str());
    }

    // Example how to pass part of cusolverdx description to NVRTC kernel
    // Add M_SIZE and LDA definition to opts
    std::string m_size_definition = "-DM_SIZE=" + std::to_string(m_size);
    opts.push_back(m_size_definition.c_str());

    std::string lda_definition = "-DSOLVER_LDA=" + std::to_string(lda_smem);
    opts.push_back(lda_definition.c_str());

    // Add GPU_ARCHITECTURE definition to opts
    std::string gpu_architecture_definition = "-DSOLVER_SM=" + std::to_string(common::nvrtc::get_device_architecture(current_device) * 10);
    opts.push_back(gpu_architecture_definition.c_str());

    // Add gpu-architecture to opts
    std::string gpu_architecture_option = common::nvrtc::get_device_architecture_option(current_device);
    opts.push_back(gpu_architecture_option.c_str());

    auto        start         = std::chrono::high_resolution_clock::now();
    nvrtcResult compileResult = nvrtcCompileProgram(program,                       // program
                                                    static_cast<int>(opts.size()), // numOptions
                                                    opts.data());                  // options
    auto        stop          = std::chrono::high_resolution_clock::now();
    std::cout << " nvrtcCompileProgram takes " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms\n";

    // Obtain compilation log from the program
    if (compileResult != NVRTC_SUCCESS) {
        for (auto o : opts) {
            std::cout << o << std::endl;
        }
        common::nvrtc::print_program_log(program);
        std::exit(1);
    }

    // Obtain generated LTO IR from the program
    size_t lto_size;
    NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(program, &lto_size));
    auto ltoir = std::make_unique<char[]>(lto_size);
    NVRTC_SAFE_CALL(nvrtcGetLTOIR(program, ltoir.get()));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

    CUdevice   cuDevice;
    CUcontext  context;
    CUmodule   module;
    CUfunction kernel;
    CU_CHECK_AND_EXIT(cuInit(0));
    CU_CHECK_AND_EXIT(cuDeviceGet(&cuDevice, current_device));
    CU_CHECK_AND_EXIT(cuCtxCreate(&context, 0, cuDevice));

    // Load the generated LTO IR and the static cusolverdx LTO library
    nvJitLinkHandle linker;
    // Dynamically determine the arch to link for
    int major = 0;
    int minor = 0;
    CU_CHECK_AND_EXIT(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CU_CHECK_AND_EXIT(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    int  arch = major * 10 + minor;
    char smbuf[16];
    sprintf(smbuf, "-arch=sm_%d", arch);
    const char* lopts[] = {"-lto", smbuf};
    NVJITLINK_SAFE_CALL(linker, nvJitLinkCreate(&linker, 2, lopts));

    // CUSOLVERDX_LIBRARY should point to cuSolverDx static library file; only for Linux x86-64
    // CUSOLVERDX_FATBIN should point to cuSolverDx fatbin file
    const char* library_env_ptr = std::getenv("CUSOLVERDX_EXAMPLE_CUSOLVERDX_LIBRARY");
    const char* fatbin_env_ptr  = std::getenv("CUSOLVERDX_EXAMPLE_CUSOLVERDX_FATBIN");

    start = std::chrono::high_resolution_clock::now();
    if (fatbin_env_ptr != nullptr) {
        NVJITLINK_SAFE_CALL(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, std::string(fatbin_env_ptr).c_str()));
    } else if(library_env_ptr != nullptr) {
        NVJITLINK_SAFE_CALL(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_LIBRARY, std::string(library_env_ptr).c_str()));
    } else {
        #ifdef CUSOLVERDX_FATBIN
        NVJITLINK_SAFE_CALL(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_FATBIN, std::string(CUSOLVERDX_FATBIN).c_str()));
        #elif defined(CUSOLVERDX_LIBRARY)
        NVJITLINK_SAFE_CALL(linker, nvJitLinkAddFile(linker, NVJITLINK_INPUT_LIBRARY, std::string(CUSOLVERDX_LIBRARY).c_str()));
        #else
        std::cout << "Please set CUSOLVERDX_EXAMPLE_CUSOLVERDX_LIBRARY env or define CUSOLVERDX_LIBRARY\n";
        return 1;
        #endif
    }
    auto stop1 = std::chrono::high_resolution_clock::now();
    NVJITLINK_SAFE_CALL(linker, nvJitLinkAddData(linker, NVJITLINK_INPUT_LTOIR, ltoir.get(), lto_size, "lto_online"));
    auto stop2 = std::chrono::high_resolution_clock::now();

    // The call to nvJitLinkComplete causes linker to link together the two
    // LTO IR modules (libcusolverdx.a and online), do optimization on the linked LTO IR, and generate cubin from it.
    NVJITLINK_SAFE_CALL(linker, nvJitLinkComplete(linker));
    stop = std::chrono::high_resolution_clock::now();

    std::cout << " nvJitLinkAddFile libcusolverdx takes " << std::chrono::duration_cast<std::chrono::milliseconds>(stop1 - start).count() << " ms\n"
              << " nvJitLinkAddData online data takes " << std::chrono::duration_cast<std::chrono::milliseconds>(stop2 - stop1).count() << " ms\n"
              << " nvJitLinkComplete takes " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - stop2).count() << " ms\n";

    size_t cubin_size;
    NVJITLINK_SAFE_CALL(linker, nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
    auto cubin = std::make_unique<char[]>(cubin_size);
    NVJITLINK_SAFE_CALL(linker, nvJitLinkGetLinkedCubin(linker, cubin.get()));
    NVJITLINK_SAFE_CALL(linker, nvJitLinkDestroy(&linker));

    CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, cubin.get(), 0, 0, 0));
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "potrs_kernel"));

    std::vector<data_type> A(m_size * m_size * batches);
    common::fillup_random_diagonal_dominant_matrix_col_major<data_type>(m_size, m_size, A.data(), m_size, false, 2, 4, batches);

    std::vector<data_type> B = common::generate_random_data<data_type>(-1, 1, m_size * batches);

    CUdeviceptr d_A;
    CU_CHECK_AND_EXIT(cuMemAlloc(&d_A, A.size() * sizeof(data_type)));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(d_A, A.data(), A.size() * sizeof(data_type)));

    CUdeviceptr d_B;
    CU_CHECK_AND_EXIT(cuMemAlloc(&d_B, B.size() * sizeof(data_type)));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(d_B, B.data(), B.size() * sizeof(data_type)));

    CUdeviceptr d_info;
    CU_CHECK_AND_EXIT(cuMemAlloc(&d_info, sizeof(int) * batches));
    std::vector<int> info(batches, 9); // set to a non-zero number

    // Get BLAS::block_dim and BLAS::shared_memory required for kernel launch
    dim3         block_dim          = common::nvrtc::get_global_from_module<dim3>(module, "solver_block_dim");
    unsigned int shared_memory_size = common::nvrtc::get_global_from_module<unsigned int>(module, "solver_shared_memory_size");
    std::cout << "shared memory size " << shared_memory_size << std::endl;

    // Increase dynamic shared memory limit
    CU_CHECK_AND_EXIT(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_memory_size));

    // Execute test_kernel
    void* args[] = {&d_A, &d_B, &d_info};
    CU_CHECK_AND_EXIT(cuLaunchKernel(kernel,
                                     batches, // number of blocks
                                     1,
                                     1,
                                     block_dim.x, // number of threads
                                     block_dim.y,
                                     block_dim.z,
                                     shared_memory_size,
                                     NULL, // NULL stream
                                     args,
                                     0));
    CU_CHECK_AND_EXIT(cuCtxSynchronize());

    // Retrieve info
    CU_CHECK_AND_EXIT(cuMemcpyDtoH(info.data(), d_info, sizeof(int) * batches))
    if (std::accumulate(info.begin(), info.end(), 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches";
        return -1;
    }
    // Copy d_B to B_output
    CU_CHECK_AND_EXIT(cuMemcpyDtoH(B.data(), d_B, sizeof(data_type) * B.size()));
    for (auto i = 0; i < B.size(); i += 10) {
        std::cout << "B[" << i << "] = (" << B[i].x << ", " << B[i].y << ")\n";
    }

    // Release resources.
    CU_CHECK_AND_EXIT(cuMemFree(d_A));
    CU_CHECK_AND_EXIT(cuMemFree(d_B));
    CU_CHECK_AND_EXIT(cuMemFree(d_info));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));
    CU_CHECK_AND_EXIT(cuCtxDestroy(context));

    std::cout << "Success\n";
    return 0;
}
