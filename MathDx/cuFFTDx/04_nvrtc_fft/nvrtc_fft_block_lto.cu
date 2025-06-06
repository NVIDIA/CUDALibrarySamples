#include <vector>
#include <iostream>
#include <chrono>

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
using FFT = decltype(Block() + size_desc() + dir_desc() + type_c2c() + Precision<double>() + SM<FFT_SM>() + ElementsPerThread<FFT_EPT>() + experimental::CodeType<experimental::code_type::ltoir>());

inline __device__ unsigned int batch_offset(const unsigned int local_fft_id,
                                            const unsigned int ffts_per_block = blockDim.y) {
    unsigned int global_fft_id = ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * ffts_per_block + local_fft_id);
    return cufftdx::size_of<FFT>::value * global_fft_id;
}

extern "C" __global__ void test_kernel(typename FFT::value_type* input)
{
  typename FFT::value_type thread_data[FFT::storage_size];

  const unsigned int offset = batch_offset(threadIdx.y, FFT::ffts_per_block);
  constexpr unsigned int stride = FFT::stride;
  unsigned int index = offset + threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
        thread_data[i] = input[index];
        index += stride;
    }
  }

  extern __shared__ FFT::value_type shared_mem[];
  FFT().execute(thread_data, shared_mem);

  index = offset + threadIdx.x;
  for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
    if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
        input[index] = thread_data[i];
        index += stride;
    }
  }
}
)kernel";


int main(int, char**) {
    // Note that FFT description is only defined in kernel
    static constexpr unsigned int fft_size = 64;
    static constexpr unsigned int fft_ept  = 8;

    using value_type = double2; // or cuDoubleComplex, or cuda::std::complex<double>

    // Get current device
    int current_device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&current_device));

    auto [lto_db, ltoirs, block_dim, shared_memory_size] =
        cufftdx::utils::get_database_and_ltoir(fft_size,
                                               CUFFT_DESC_INVERSE,
                                               CUFFT_DESC_C2C,
                                               example::nvrtc::get_device_architecture(current_device) * 10,
                                               CUFFT_DESC_BLOCK,
                                               CUFFT_DESC_DOUBLE,
                                               CUFFT_DESC_NORMAL,
                                               fft_ept);

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
        "--device-as-default-execution-space",
        "--include-path=" CUDA_INCLUDE_DIR // Add path to CUDA include directory
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

    // Add FFT_EPT definition to opts
    std::string fft_ept_definition = "-DFFT_EPT=" + std::to_string(fft_ept);
    opts.push_back(fft_ept_definition.c_str());

    // Add GPU_ARCHITECTURE definition to opts
    std::string gpu_architecture_definition = "-DFFT_SM=" + std::to_string(example::nvrtc::get_device_architecture(current_device) * 10);
    opts.push_back(gpu_architecture_definition.c_str());

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
    #if CUDA_VERSION >= 12090
    CUctxCreateParams params;
    CU_CHECK_AND_EXIT(cuCtxCreate_v4(&context, &params, 0, cuDevice));
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
                                     block_dim.x, // number of threads
                                     block_dim.y,
                                     block_dim.z,
                                     shared_memory_size, // shared memory size
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

    double sum = (fft_size * (fft_size - 1)) / 2.0;
    if ((host_output[0].x - sum > 0.01)) {
        std::cout << "[test_nvrtc_lto_block] Failed" << std::endl;
        return 1;
    }
    std::cout << "[test_nvrtc_lto_block] Passed" << std::endl;
    return 0;
}
