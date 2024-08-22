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
using arr_desc  = Arrangement<cublasdx::row_major, cublasdx::col_major>;
using BLAS = decltype(Block() + Function<function::MM>() + size_desc() + type_desc() + arr_desc() + Precision<__half, __half, float>() + SM<BLAS_SM>());

__constant__ dim3         blas_block_dim     = BLAS::block_dim;
__constant__ unsigned int blas_shared_memory = BLAS::shared_memory_size;

extern "C" __global__ void test_kernel(typename BLAS::a_value_type* a, typename BLAS::b_value_type* b, typename BLAS::c_value_type* c)
{
    extern __shared__ __align__(16) char smem[];

    // Load
    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

// libcu++ doesn't support structured bindings for cuda::std::tuple before 2.1.0 version
#if _LIBCUDACXX_CUDA_API_VERSION >= 2001000
    auto [smem_a, smem_b, smem_c] = BLAS::slice_shared_memory(smem);
#else
    auto sliced_smem = BLAS::slice_shared_memory(smem);
    auto smem_a = cuda::std::get<0>(sliced_smem);
    auto smem_b = cuda::std::get<1>(sliced_smem);
    auto smem_c = cuda::std::get<2>(sliced_smem);
#endif
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    typename BLAS::c_value_type alpha = 1.0, beta = 1.0;
    BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);

    // Store
    __syncthreads();
    cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, c_global_tensor);
}
)kernel";

int main(int, char**) {
    // Note that BLAS description is only defined in kernel
    // Precision: __half, __half, float
    // TransposeMode: NN
    // Type: Real
    // alpha: 1
    // beta: 1
    unsigned int blas_m = 32;
    unsigned int blas_n = 32;
    unsigned int blas_k = 32;

    // Complex type according to precision
    // using value_type = cuDoubleComplex; // or double2, or cuda::std::complex<double>
    using a_value_type = __half;
    using b_value_type = __half;
    using c_value_type = float;

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
    CU_CHECK_AND_EXIT(cuCtxCreate(&context, 0, cuDevice));
    CU_CHECK_AND_EXIT(cuModuleLoadDataEx(&module, cubin.get(), 0, 0, 0));
    CU_CHECK_AND_EXIT(cuModuleGetFunction(&kernel, module, "test_kernel"));

    const size_t blas_a_size = blas_m * blas_k;
    const size_t blas_b_size = blas_k * blas_n;
    const size_t blas_c_size = blas_m * blas_n;

    // Generate input for execution
    std::vector<a_value_type> host_a(blas_a_size);
    std::vector<b_value_type> host_b(blas_b_size);
    std::vector<c_value_type> host_c(blas_c_size);
    {
        std::random_device                     rd;
        std::mt19937                           gen(rd());
        std::uniform_real_distribution<double> dist(1.0, 1.0);
        auto                                   random_value_func = [&dist, &gen]() { return dist(gen); };
        std::generate(host_a.begin(), host_a.end(), random_value_func);
        std::generate(host_b.begin(), host_b.end(), random_value_func);
    }

    const size_t blas_a_size_bytes = blas_a_size * sizeof(a_value_type);
    const size_t blas_b_size_bytes = blas_b_size * sizeof(b_value_type);
    const size_t blas_c_size_bytes = blas_c_size * sizeof(c_value_type);
    CUdeviceptr  device_a;
    CUdeviceptr  device_b;
    CUdeviceptr  device_c;
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_a, blas_a_size_bytes));
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_b, blas_b_size_bytes));
    CU_CHECK_AND_EXIT(cuMemAlloc(&device_c, blas_c_size_bytes));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(device_a, host_a.data(), blas_a_size_bytes));
    CU_CHECK_AND_EXIT(cuMemcpyHtoD(device_b, host_b.data(), blas_b_size_bytes));

    // Get BLAS::block_dim and BLAS::shared_memory required for kernel launch
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
