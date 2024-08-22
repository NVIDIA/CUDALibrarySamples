#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "reference.hpp"

template<unsigned int Scenario, class BLAS, class ValueType = example::uniform_value_type_t<BLAS>>
__global__ void gemm_kernel(const ValueType* a,
                            const ValueType* b,
                            const ValueType* c,
                            const ValueType  alpha,
                            const ValueType  beta,
                            ValueType*       output) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());
    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());

    auto [smem_a, smem_b, smem_c] = BLAS::slice_shared_memory(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());

    using alignment = cublasdx::alignment_of<BLAS>;
    // Kernel launched with block dimensions equal to BLAS::block_dim
    if constexpr (Scenario == 0) {
        cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
        cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
        cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
        cublasdx::copy_wait();
        BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
        __syncthreads();
        cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);
    }
    // Kernel launched with block dimensions of the same rank as BLAS::block_dim and
    // at least the same number of threads
    else if constexpr (Scenario == 1) {
        cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
        cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
        cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
        cublasdx::copy_wait();
        BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
        __syncthreads();
        cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);
    }
    // Kernel launched with 2D block dimensions blockDim=dim3(X, Y), BLAS::block_dim is 1D,
    // blockDim must have at least as many threads in X dimension (1st dim) as BLAS::block_dim.
    else if constexpr (Scenario == 2) {
        if (threadIdx.y == 0) {
            cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
            cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
            cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
        }
        cublasdx::copy_wait();
        if (threadIdx.y == 0) {
            BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
        }
        __syncthreads();
        if (threadIdx.y == 0) {
            cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);
        }
    }
    // Kernel launched with 3D block dimensions blockDim=dim3(X, Y, Z), BLAS::block_dim is 2D,
    // blockDim must have at least as many threads in the first 2 dimensions (X, Y) as BLAS::block_dim.
    else if constexpr (Scenario == 3) {
        if (threadIdx.z == 0) {
            cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
            cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
            cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
        }
        cublasdx::copy_wait();
        if (threadIdx.z == 0) {
            BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
        }
        __syncthreads();
        if (threadIdx.z == 0) {
            cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);
        }
    }
}

constexpr dim3 get_blas_block_dim(unsigned int scenario) {
    if(scenario == 0) {
        return dim3(128);
    } else if(scenario == 1) {
        return dim3(128);
    } else if(scenario == 2) {
        return dim3(128);
    } else if(scenario == 3) {
        return dim3(32, 4);
    }
    return dim3(128);
}

constexpr dim3 get_kernel_block_dim(unsigned int scenario) {
    if(scenario == 0) {
        return dim3(128);
    } else if(scenario == 1) {
        return dim3(255);
    } else if(scenario == 2) {
        return dim3(196, 2);
    } else if(scenario == 3) {
        return dim3(32, 4, 2);
    }
    return dim3(128);
}

// This is an example of fp16 general matrix-matrix multiplication (GEMM) performed
// in a single CUDA block:
//
//              C = alpha * A * B + beta * C
//
// * A, B, and C are matrices containing real single precision floating-point values.
// * alpha and beta are real single precision floating-point values.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory. The results are verified against cuBLAS.
//
// In this example the number of threads participating in the GEMM operation is imposed by providing
// BlockDim operator in definition of the GEMM.
//
// There are 4 scenarios (0, 1, 2, 3) listed below. Each scenario represents a different combination of block
// dimensions set via BlockDim in BLAS description and block dimensions the kernel is launched with. The examples
// demonstrates how to handle launching kernel with different layout and number of thread than the block dimensions
// configured in BLAS description.
//
// Possible Scenarios:
//
// 0 - Kernel launched with block dimensions equal to BLAS::block_dim.
// 1 - Kernel launched with block dimensions of the same rank as BLAS::block_dim and
//     at least the same number of threads.
// 2 - Kernel launched with 2D block dimensions blockDim=dim3(X, Y), BLAS::block_dim is 1D,
//     blockDim must have at least as many threads in X dimension (1st dim) as BLAS::block_dim.
// 3 - Kernel launched with 3D block dimensions blockDim=dim3(X, Y, Z), BLAS::block_dim is 2D,
//     blockDim must have at least as many threads in the first 2 dimensions (X, Y) as BLAS::block_dim.
//
// In every scenario only one GEMM is performed. To see manual batching in a single CUDA block see
// example batched_gemm_fp64.
template<unsigned int Scenario, unsigned int Arch>
int simple_gemm() {
    // Parameters m, n, k define the dimensions of matrices A, B, and C
    constexpr unsigned int m = 64;
    constexpr unsigned int n = 64;
    constexpr unsigned int k = 64;

    // Selected CUDA block size (1D)
    constexpr auto blas_block_dim = get_blas_block_dim(Scenario);
    constexpr auto kernel_block_dim = get_kernel_block_dim(Scenario);

    // If matrix A is column-major (or not transposed in BLAS nomenclature) its logical dimensions are: [m, k] (m rows, k columns)
    // If matrix B is column-major its logical dimensions are: [k, n]
    // If matrix A is row-major (or transposed in BLAS nomenclature) its logical dimensions are: [k, m]
    // If matrix B is row-major its logical dimensions are: [n, k]
    // The dimensions of matrix C are: [m, n]
    constexpr auto arrangement_a = cublasdx::col_major;
    constexpr auto arrangement_b = cublasdx::row_major;

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. Block operator informs that GEMM should be performed on CUDA block level.
    // 4. BlockDim operator sets layout and number of threads.
    // 5. Targeted CUDA compute capability is selected with SM operator.
    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<__half>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<arrangement_a, arrangement_b>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<blas_block_dim.x, blas_block_dim.y, blas_block_dim.z>() +
                          cublasdx::SM<Arch>());

    using value_type = example::uniform_value_type_t<BLAS>;

    // Allocate managed memory for a, b, c, and output
    value_type* inputs;
    value_type* output;
    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;
    auto inputs_size       = global_a_size + global_b_size + global_c_size;

    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, global_c_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + global_a_size;
    value_type* c     = b + global_b_size;
    value_type  alpha = value_type(1.0);
    value_type  beta  = value_type(2.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, global_a_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, global_b_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<Scenario, BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, BLAS::shared_memory_size));

    // Execute kernel
    gemm_kernel<Scenario, BLAS><<<1, kernel_block_dim, BLAS::shared_memory_size>>>(a, b, c, alpha, beta, output);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    // Copy results back to host
    std::vector<value_type> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Calculate reference
    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a, host_b, beta, host_c);

    // Check against reference
    if (example::check(host_output, reference_host_output)) {
        std::cout << Scenario << ": Success" << std::endl;
        return 0;
    }
    std::cout << Scenario << ": Failure" << std::endl;
    return 1;
}

template<unsigned int Arch>
struct simple_gemm_functor {
    int operator()() {
        int status = 0;
        status = simple_gemm<0, Arch>();
        if(status) return status;
        status = simple_gemm<1, Arch>();
        if(status) return status;
        status = simple_gemm<2, Arch>();
        if(status) return status;
        status = simple_gemm<3, Arch>();
        return status;
    }
};

int main(int, char**) {
    return example::sm_runner<simple_gemm_functor>();
}
