#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include "common.hpp"
#include "common_nvrtc.hpp"

const char* test_kernel = R"kernel(
#include <cufftdx.hpp>

using namespace cufftdx;

// FFT Operators
using size_desc = Size<FFT_SIZE>;
using dir_desc  = Direction<fft_direction::inverse>;
using type_desc = Type<fft_type::c2c>;
using FFT = decltype(Block() + size_desc() + dir_desc() + type_desc() + Precision<double>() + SM<FFT_SM>() + ElementsPerThread<FFT_EPT>());

__constant__ dim3         fft_block_dim     = FFT::block_dim;
__constant__ unsigned int fft_shared_memory = FFT::shared_memory_size;

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
    // Precision: double
    static constexpr unsigned int fft_size = 64;
    static constexpr unsigned int fft_ept  = 8;

    // Complex type according to precision
    using value_type = cuDoubleComplex; // or double2, or cuda::std::complex<double>

    nvrtcProgram program;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,            // program
                                       test_kernel,      // buffer
                                       "test_kernel.cu", // name
                                       0,                // numHeaders
                                       NULL,             // headers
                                       NULL));           // includeNames

    // Get current device
    int current_device;
    CUDA_CHECK_AND_EXIT(cudaGetDevice(&current_device));

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

    nvrtcResult compileResult = nvrtcCompileProgram(program,         // program
                                                    static_cast<int>(opts.size()),  // numOptions
                                                    opts.data()); // options

    // Obtain compilation log from the program
    if (compileResult != NVRTC_SUCCESS) {
        example::nvrtc::print_program_log(program);
        std::exit(1);
    }

    // Obtain PTX from the program.
    size_t ptx_size;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &ptx_size));
    auto ptx = std::make_unique<char[]>(ptx_size);
    NVRTC_SAFE_CALL(nvrtcGetPTX(program, ptx.get()));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

    // Load the generated PTX and get a handle to the test_kernel
    CUdevice   cuDevice;
    CUcontext  context;
    CUmodule   module;
    CUfunction kernel;
    CU_CHECK_AND_EXIT(cuInit(0));
    CU_CHECK_AND_EXIT(cuDeviceGet(&cuDevice, current_device));
    CU_CHECK_AND_EXIT(cuCtxCreate(&context, 0, cuDevice));
    CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, ptx.get(), 0, 0, 0));
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "test_kernel"));

    // Generate input for execution
    std::vector<value_type> host_input(fft_size);
    float                   i = 0.0f;
    for (auto& v : host_input) {
        v.x = i++;
        v.y = 0;
    }
    size_t fft_buffer_size = fft_size * sizeof(value_type);
    // Print input
    for (size_t i = 0; i < fft_size; ++i) {
        std::cout << i << ": (" << host_input[i].x << ", " << host_input[i].y << ")" << std::endl;
    }

    CUdeviceptr device_values;
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_values, fft_buffer_size));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(device_values, host_input.data(), fft_buffer_size));

    // Get FFT::block_dim and FFT::shared_memory required for kernel launch
    dim3 fft_block_dim = example::nvrtc::get_global_from_module<dim3>(module, "fft_block_dim");
    unsigned int fft_shared_memory = example::nvrtc::get_global_from_module<unsigned int>(module, "fft_shared_memory");

    // Increase dynamic shared memory limit
    CU_CHECK_AND_EXIT(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, fft_shared_memory));

    // Execute test_kernel
    void* args[] = {&device_values};
    CU_CHECK_AND_EXIT(cuLaunchKernel(kernel,
                                       1, // number of blocks
                                       1,
                                       1,
                                       fft_block_dim.x, // number of threads
                                       fft_block_dim.y,
                                       1,
                                       fft_shared_memory,
                                       NULL, // NULL stream
                                       args,
                                       0));
    CU_CHECK_AND_EXIT(cuCtxSynchronize());

    // Retrieve and print output
    std::vector<value_type> host_output(fft_size);
    CU_CHECK_AND_EXIT(cuMemcpyDtoH(host_output.data(), device_values, fft_buffer_size));
    for (size_t i = 0; i < fft_size; ++i) {
        std::cout << i << ": (" << host_output[i].x << ", " << host_output[i].y << ")" << std::endl;
    }

    // Release resources.
    CU_CHECK_AND_EXIT(cuMemFree(device_values));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));
    CU_CHECK_AND_EXIT(cuCtxDestroy(context));

    double sum = (fft_size * (fft_size - 1)) / 2.0;
    if ((host_output[0].x - sum > 0.01)) {
        std::cout << "[nvrtc_fft_block] Failed" << std::endl;
        return 1;
    }
    std::cout << "[nvrtc_fft_block] Passed" << std::endl;
    return 0;
}
