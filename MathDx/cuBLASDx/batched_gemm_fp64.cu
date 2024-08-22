#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

// Number of batches
constexpr unsigned int batches = 2;

template<class BLAS, class ValueType = example::uniform_value_type_t<BLAS>>
__global__ void gemm_kernel(const ValueType* a,
                            const ValueType* b,
                            const ValueType* c,
                            const ValueType  alpha,
                            const ValueType  beta,
                            ValueType*       output) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    // threads (X, 0) calculates the first batch
    // threads (X, 1) calculates the 2nd batch etc.
    // We need to move pointers to corresponding batches
    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    a += threadIdx.y * global_a_size;
    b += threadIdx.y * global_b_size;
    c += threadIdx.y * global_c_size;
    output += threadIdx.y * global_c_size;

    value_type* smem_a = reinterpret_cast<value_type*>(smem) + threadIdx.y * BLAS::a_size;
    value_type* smem_b = reinterpret_cast<value_type*>(smem) + (batches * BLAS::a_size) + threadIdx.y * BLAS::b_size;
    value_type* smem_c = reinterpret_cast<value_type*>(smem) + (batches * BLAS::a_size) + (batches * BLAS::b_size) + threadIdx.y * BLAS::c_size;

    // Load all batches
    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    auto a_shared_tensor = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();
    BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);

    __syncthreads();
    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());
    cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);
}

// This is an example of two fp32 general matrix-matrix multiplications (GEMM) performed in a single CUDA block:
//
//              C(X) = alpha * A(X) * B(X) + beta * C(X)
//
// * X - batch id
// * A, B, and C are matrices containing real single precision floating-point values.
// * alpha and beta are real single precision floating-point values.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory. The results are verified against cuBLAS.
//
// In this example BlockDim operator added to the GEMM definition defines the number and the layout of threads
// participating in the calculations.
//
// In order to achieve batching on a block level 1D BlockDim operator (BlockDim<64>) is added to the description,
// and launching the kernel with 2D block - dim3(64,2). Threads with the same 2nd dimension (threadIdx.y) participate
// in the same calculations.
//
// Note: Examples demonstrates how to set block dimensions to enable manual batching. The performance of included
// kernel was not checked and it is not optimized.
template<unsigned int Arch>
int simple_gemm() {
    // Parameters m, n, k define the dimensions of matrices A, B, and C
    constexpr unsigned int m = 16;
    constexpr unsigned int n = 16;
    constexpr unsigned int k = 16;

    // Selected CUDA block size (1D)
    constexpr unsigned int block_size = 64;

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. The data arrangements of A, B matrices are set (C is defaulted to column major).
    //    - Optional
    // 4. The data alignment of A, B and C matrices is set to the max accepted value.
    //    - Optional
    // 5. Block operator informs that GEMM should be performed on CUDA block level.
    // 6. BlockDim operator sets layout and number of threads.
    //    - Optional
    // 7. Targeted CUDA compute capability is selected with SM operator.
    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<double>() +
                          cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
                          cublasdx::Alignment<16, 16, 16>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() +
                          cublasdx::SM<Arch>());

    using value_type = example::uniform_value_type_t<BLAS>;

    // Allocate managed memory for a, b, c, and output
    value_type* inputs;
    value_type* output;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;
    auto inputs_size       = batches * (global_a_size + global_b_size + global_c_size);

    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, batches * global_c_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + (batches * global_a_size);
    value_type* c     = b + (batches * global_b_size);
    value_type  alpha = value_type(1.0);
    value_type  beta  = value_type(2.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, batches * global_a_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, batches * global_b_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, batches * global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), batches * global_a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), batches * global_b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), batches * global_c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const dim3 block_dim = dim3(block_size, batches);
    const auto shared_memory_size = batches * BLAS::shared_memory_size;

    // Increase max dynamic shared memory for the kernel if needed
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    // Execute kernel
    gemm_kernel<BLAS><<<1, block_dim, shared_memory_size>>>(a, b, c, alpha, beta, output);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results of the 1st batch back to host
    std::vector<value_type> host_output1(global_c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output1.data(), output, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    // Copy results of the 2nd batch back to host
    std::vector<value_type> host_output2(global_c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output2.data(), output + global_c_size, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Calculate reference for the 1st batch
    decltype(host_a) host_a1(host_a.begin(), host_a.begin() + global_a_size);
    decltype(host_b) host_b1(host_b.begin(), host_b.begin() + global_b_size);
    decltype(host_c) host_c1(host_c.begin(), host_c.begin() + global_c_size);
    auto reference_host_output1 = example::reference_gemm<BLAS>(alpha, host_a1, host_b1, beta, host_c1);
    // Calculate reference for the 2nd batch
    decltype(host_a) host_a2(host_a.begin() + global_a_size, host_a.begin() + 2 * global_a_size);
    decltype(host_b) host_b2(host_b.begin() + global_b_size, host_b.begin() + 2 * global_b_size);
    decltype(host_c) host_c2(host_c.begin() + global_c_size, host_c.begin() + 2 * global_c_size);
    auto reference_host_output2 = example::reference_gemm<BLAS>(alpha, host_a2, host_b2, beta, host_c2);

    // Check against reference
    if (example::check(host_output1, reference_host_output1) && example::check(host_output2, reference_host_output2)) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

template<unsigned int Arch>
struct simple_gemm_functor {
    int operator()() { return simple_gemm<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_gemm_functor>();
}
