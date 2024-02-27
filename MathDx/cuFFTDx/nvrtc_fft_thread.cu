#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "common.hpp"
#include "common_nvrtc.hpp"

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
    // thread_data should have all input data in order.
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
        "--device-as-default-execution-space",
        "--include-path=" CUDA_INCLUDE_DIR // Add path to CUDA include directory
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

    // Obtain PTX from the program.
    size_t ptx_size;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(program, &ptx_size));
    auto ptx = std::make_unique<char[]>(ptx_size);
    NVRTC_SAFE_CALL(nvrtcGetPTX(program, ptx.get()));

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

    // Load the generated PTX and get a handle to the thread_fft_kernel
    CUcontext  context;
    CUmodule   module;
    CUfunction kernel;
    CUDA_CHECK_AND_EXIT(cudaFree(0));               // Initialize CUDA context
    CU_CHECK_AND_EXIT(cuCtxGetCurrent(&context)); // Get current context
    CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, ptx.get(), 0, 0, 0));
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "thread_fft_kernel"));

    // Generate input for execution
    std::vector<value_type> host_input(fft_size);
    float                   i = 0.0f;
    for (auto& v : host_input) {
        v.x = i++;
        v.y = 0;
    }

    size_t fft_buffer_size = fft_size * sizeof(value_type);
    void*  device_values;
    CUDA_CHECK_AND_EXIT(cudaMalloc(&device_values, fft_buffer_size));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(device_values, host_input.data(), fft_buffer_size, cudaMemcpyHostToDevice));

    // Execute thread_fft_kernel
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
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), device_values, fft_buffer_size, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < fft_size; ++i) {
        std::cout << i << ": (" << host_output[i].x << ", " << host_output[i].y << ")" << std::endl;
    }

    // Release resources.
    CUDA_CHECK_AND_EXIT(cudaFree(device_values));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));

    double expected_value = (fft_size * (fft_size + 1)) / 2;
    if ((host_output[0].x - expected_value) > 0.01) {
        std::cout << "Failed" << std::endl;
        return 1;
    }
    std::cout << "Success" << std::endl;
    return 0;
}
