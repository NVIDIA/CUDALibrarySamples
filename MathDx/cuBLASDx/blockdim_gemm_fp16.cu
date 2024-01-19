#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

template<unsigned int Scenario, class BLAS, class ValueType = typename BLAS::value_type>
__global__ void gemm_kernel(const ValueType* a,
                            const ValueType* b,
                            const ValueType* c,
                            const ValueType  alpha,
                            const ValueType  beta,
                            ValueType*       output) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    value_type* smem_a = reinterpret_cast<value_type*>(smem);
    value_type* smem_b = reinterpret_cast<value_type*>(smem) + BLAS::a_size;
    value_type* smem_c = reinterpret_cast<value_type*>(smem) + BLAS::a_size + BLAS::b_size;

    example::io<BLAS>::load(smem_a, a, BLAS::a_size);
    example::io<BLAS>::load(smem_b, b, BLAS::b_size);
    example::io<BLAS>::load(smem_c, c, BLAS::c_size);
    __syncthreads();

    // Kernel launched with block dimensions equal to BLAS::block_dim
    if constexpr (Scenario == 0) {
        BLAS().execute(alpha, smem_a, smem_b, beta, smem_c);
    }
    // Kernel launched with block dimensions of the same rank as BLAS::block_dim and
    // at least the same number of threads
    else if (Scenario == 1) {
        BLAS().execute(alpha, smem_a, smem_b, beta, smem_c);
    }
    // Kernel launched with 2D block dimensions blockDim=dim3(X, Y), BLAS::block_dim is 1D,
    // blockDim must have at least as many threads in X dimension (1st dim) as BLAS::block_dim.
    else if (Scenario == 2) {
        if (threadIdx.y == 0) {
            BLAS().execute(alpha, smem_a, smem_b, beta, smem_c);
        }
    }
    // Kernel launched with 3D block dimensions blockDim=dim3(X, Y, Z), BLAS::block_dim is 2D,
    // blockDim must have at least as many threads in the first 2 dimensions (X, Y) as BLAS::block_dim.
    else if (Scenario == 3) {
        if (threadIdx.z == 0) {
            BLAS().execute(alpha, smem_a, smem_b, beta, smem_c);
        }
    }

    __syncthreads();
    example::io<BLAS>::store(output, smem_c, BLAS::c_size);
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

    // If matrix A is not transposed its logical dimensions are: [m, k] (m rows, k columns)
    // If matrix B is not transposed its logical dimensions are: [k, n]
    // If matrix A is transposed its logical dimensions are: [k, m]
    // If matrix B is transposed its logical dimensions are: [n, k]
    // The dimensions of matrix C are: [m, n]
    constexpr auto a_transpose_mode = cublasdx::transpose_mode::non_transposed;
    constexpr auto b_transpose_mode = cublasdx::transpose_mode::transposed;

    // Selected CUDA block size (1D)
    constexpr auto blas_block_dim = get_blas_block_dim(Scenario);
    constexpr auto kernel_block_dim = get_kernel_block_dim(Scenario);

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. The transpose modes of A and B matrices are set.
    // 4. Block operator informs that GEMM should be performed on CUDA block level.
    // 5. BlockDim operator sets layout and number of threads.
    // 6. Targeted CUDA compute capability is selected with SM operator.
    using BLAS       = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<__half>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::TransposeMode<a_transpose_mode, b_transpose_mode>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<blas_block_dim.x, blas_block_dim.y, blas_block_dim.z>() +
                          cublasdx::SM<Arch>());
    #ifdef CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using value_type = example::value_type_t<BLAS>;
    #else
    using value_type = typename BLAS::value_type;
    #endif

    // Allocate managed memory for a, b, c, and output
    value_type* inputs;
    value_type* output;
    // BLAS::a_size/b_size/c_size include padding (take into account the leading dimension if set)
    auto inputs_size       = BLAS::a_size + BLAS::b_size + BLAS::c_size;
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, BLAS::c_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + (BLAS::a_size);
    value_type* c     = b + (BLAS::b_size);
    value_type  alpha = value_type(1.0);
    value_type  beta  = value_type(2.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, BLAS::a_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, BLAS::b_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, BLAS::c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), BLAS::a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), BLAS::b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), BLAS::c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<Scenario, BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, BLAS::shared_memory_size));

    // Execute kernel
    gemm_kernel<Scenario, BLAS><<<1, kernel_block_dim, BLAS::shared_memory_size>>>(a, b, c, alpha, beta, output);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    // Copy results back to host
    std::vector<value_type> host_output(BLAS::c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output, BLAS::c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
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
