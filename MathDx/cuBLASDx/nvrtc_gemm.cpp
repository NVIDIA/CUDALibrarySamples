#include <vector>
#include <iostream>
#include <string>
#include <algorithm>
#include <memory>
#include <random>

#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuComplex.h>

#include "common_nvrtc.hpp"

const char* test_kernel = R"kernel(
#include <cublasdx.hpp>

using namespace cublasdx;

// BLAS Operators
using size_desc = Size<BLAS_M, BLAS_N, BLAS_K>;
using type_desc = Type<type::real>;
using tm_desc = TransposeMode<cublasdx::N, cublasdx::N>;
using BLAS = decltype(Block() + Function<function::MM>() + size_desc() + type_desc() + tm_desc() + Precision<double>() + SM<BLAS_SM>());

__constant__ dim3         blas_block_dim     = BLAS::block_dim;
__constant__ unsigned int blas_shared_memory = BLAS::shared_memory_size;

template<class T>
__device__ void copy(T* dst, const T* src, const unsigned int size /* number of elements */) {
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
        // Note: This copies values in padding too
        for (unsigned int idx = 0; idx < size; ++idx) {
            dst[idx] = src[idx];
        }
    }
}

extern "C" __global__ void test_kernel(typename BLAS::value_type* a, typename BLAS::value_type* b, typename BLAS::value_type* c)
{
    using value_type = BLAS::value_type;
    extern __shared__ __align__(16) char smem[];

    auto a_size = BLAS::a_size;
    auto b_size = BLAS::b_size;
    auto c_size = BLAS::c_size;

    value_type* smem_a = reinterpret_cast<value_type*>(&smem[0]);
    value_type* smem_b = reinterpret_cast<value_type*>(smem + a_size * sizeof(value_type));
    value_type* smem_c = reinterpret_cast<value_type*>(smem + a_size * sizeof(value_type) + b_size * sizeof(value_type));

    // Load
    copy(smem_a, a, a_size);
    copy(smem_b, b, b_size);
    copy(smem_c, c, c_size);
    __syncthreads();

    BLAS blas;
    blas.execute(1.0, smem_a, smem_b, 1.0, smem_c);

    // Store
    __syncthreads();
    copy(c, smem_c, c_size);
}
)kernel";

int main(int, char**) {
    // Note that BLAS description is only defined in kernel
    // Precision: double
    // TransposeMode: NN
    // Type: Real
    // alpha: 1
    // beta: 1
    unsigned int blas_m = 32;
    unsigned int blas_n = 32;
    unsigned int blas_k = 32;

    // Complex type according to precision
    // using value_type = cuDoubleComplex; // or double2, or cuda::std::complex<double>
    using value_type = double;

    nvrtcProgram program;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program,         // program
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

    // Parse cuBLASDx include dirs
    std::vector<std::string> cublasdx_include_dirs = example::nvrtc::get_cublasdx_include_dirs();
    // Add cuBLASDx include dirs to opts
    for (auto& d : cublasdx_include_dirs) {
        opts.push_back(d.c_str());
    }

    // Example how to pass part of BLAS description to NVRTC kernel
    // Add BLAS_M, BLAS_N, BLAS_K definition to opts
    std::string blas_m_definition = "-DBLAS_M=" + std::to_string(blas_m);
    opts.push_back(blas_m_definition.c_str());
    std::string blas_n_definition = "-DBLAS_N=" + std::to_string(blas_n);
    opts.push_back(blas_n_definition.c_str());
    std::string blas_k_definition = "-DBLAS_K=" + std::to_string(blas_k);
    opts.push_back(blas_k_definition.c_str());

    // Add GPU_ARCHITECTURE definition to opts
    std::string gpu_architecture_definition =
        "-DBLAS_SM=" + std::to_string(example::nvrtc::get_device_architecture(current_device) * 10);
    opts.push_back(gpu_architecture_definition.c_str());

    // Add gpu-architecture to opts
    std::string gpu_architecture_option = example::nvrtc::get_device_architecture_option(current_device);
    opts.push_back(gpu_architecture_option.c_str());

    nvrtcResult compileResult = nvrtcCompileProgram(program,                       // program
                                                    static_cast<int>(opts.size()), // numOptions
                                                    opts.data());                  // options

    // Obtain compilation log from the program
    if (compileResult != NVRTC_SUCCESS) {
        for(auto o : opts) {
            std::cout << o << std::endl;
        }
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

    const size_t blas_a_size = blas_m * blas_k;
    const size_t blas_b_size = blas_k * blas_n;
    const size_t blas_c_size = blas_m * blas_n;

    // Generate input for execution
    std::vector<value_type> host_a(blas_a_size);
    std::vector<value_type> host_b(blas_b_size);
    std::vector<value_type> host_c(blas_c_size);
    {
        std::random_device                     rd;
        std::mt19937                           gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        auto                                   random_value_func = [&dist, &gen]() { return dist(gen); };
        std::generate(host_a.begin(), host_a.end(), random_value_func);
        std::generate(host_b.begin(), host_b.end(), random_value_func);
    }


    const size_t blas_a_size_bytes = blas_a_size * sizeof(value_type);
    const size_t blas_b_size_bytes = blas_b_size * sizeof(value_type);
    const size_t blas_c_size_bytes = blas_c_size * sizeof(value_type);
    CUdeviceptr  device_a;
    CUdeviceptr  device_b;
    CUdeviceptr  device_c;
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_a, blas_a_size_bytes));
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_b, blas_b_size_bytes));
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_c, blas_c_size_bytes));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(device_a, host_a.data(), blas_a_size_bytes));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(device_b, host_b.data(), blas_b_size_bytes));

    // Get FFT::block_dim and FFT::shared_memory required for kernel launch
    dim3         blas_block_dim = example::nvrtc::get_global_from_module<dim3>(module, "blas_block_dim");
    unsigned int blas_shared_memory =
        example::nvrtc::get_global_from_module<unsigned int>(module, "blas_shared_memory");

    // Increase dynamic shared memory limit
    CU_CHECK_AND_EXIT(cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, blas_shared_memory));

    // Execute test_kernel
    void* args[] = {&device_a, &device_b, &device_c};
    CU_CHECK_AND_EXIT(cuLaunchKernel(kernel,
                                     1, // number of blocks
                                     1,
                                     1,
                                     blas_block_dim.x, // number of threads
                                     blas_block_dim.y,
                                     blas_block_dim.z,
                                     blas_shared_memory,
                                     NULL, // NULL stream
                                     args,
                                     0));
    CU_CHECK_AND_EXIT(cuCtxSynchronize());

    // Retrieve C
    CU_CHECK_AND_EXIT(cuMemcpyDtoH(host_c.data(), device_c, blas_c_size_bytes))

    // Release resources.
    CU_CHECK_AND_EXIT(cuMemFree(device_a));
    CU_CHECK_AND_EXIT(cuMemFree(device_b));
    CU_CHECK_AND_EXIT(cuMemFree(device_c));
    CU_CHECK_AND_EXIT(cuModuleUnload(module));
    CU_CHECK_AND_EXIT(cuCtxDestroy(context));

    std::cout << "Success" << std::endl;
    return 0;
}
