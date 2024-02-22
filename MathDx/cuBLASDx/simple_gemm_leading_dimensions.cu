#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

template<class BLAS, class ValueType = typename BLAS::value_type>
__launch_bounds__(BLAS::max_threads_per_block) //
    __global__                                 //
    void gemm_kernel(const ValueType  alpha,
                     const ValueType* a,
                     const ValueType* b,
                     const ValueType  beta,
                     const ValueType* c,
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

    BLAS().execute(alpha, smem_a, smem_b, beta, smem_c);

    __syncthreads();
    example::io<BLAS>::store(output, smem_c, BLAS::c_size);
}

template<class BLASWithoutLD, class ValueType = typename BLASWithoutLD::value_type>
__launch_bounds__(BLASWithoutLD::max_threads_per_block) //
    __global__                                 //
    void gemm_kernel_dynamic_ld(const ValueType  alpha,
                     const ValueType* a,
                     const unsigned int lda,
                     const ValueType* b,
                     const unsigned int ldb,
                     const ValueType  beta,
                     const ValueType* c,
                     const unsigned int ldc,
                     ValueType*       output) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    const auto [a_rows, a_cols] = BLASWithoutLD::a_dim;
    const auto [b_rows, b_cols] = BLASWithoutLD::b_dim;
    const auto [c_rows, c_cols] = BLASWithoutLD::c_dim;

    const auto a_size = (lda * a_cols);
    const auto b_size = (ldb * b_cols);
    // const auto c_size = (ldc * c_cols);

    value_type* smem_a = reinterpret_cast<value_type*>(smem);
    value_type* smem_b = reinterpret_cast<value_type*>(smem) + a_size;
    value_type* smem_c = reinterpret_cast<value_type*>(smem) + a_size + b_size;

    example::io<BLASWithoutLD>::load(smem_a, a, a_rows, a_cols, lda);
    example::io<BLASWithoutLD>::load(smem_b, b, b_rows, b_cols, ldb);
    example::io<BLASWithoutLD>::load(smem_c, c, c_rows, c_cols, ldc);
    __syncthreads();

    BLASWithoutLD().execute(alpha, smem_a, lda, smem_b, ldb, beta, smem_c, ldc);

    __syncthreads();
    example::io<BLASWithoutLD>::store(output, smem_c, c_rows, c_cols, ldc);
}

// This is an example of fp64 general matrix-matrix multiplication (GEMM) performed
// in a single CUDA block:
//
//              C = alpha * A * B + beta * C
//
// * A, B, and C are matrices containing real double precision floating-point values.
// * alpha and beta are real double precision floating-point values.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory. The results are verified against cuBLAS.
//
// In this example the number of threads participating in the GEMM operation is imposed by providing
// BlockDim operator in definition of the GEMM. If BlockDim operator is not used, cuBLASDx automatically
// selects number of threads. Block dimensions are provided via BLAS::block_dim trait.
//
// Additionally, this example sets leading dimensions for the A, B, C  matrices participating in the GEMM.
// For BLAS type they are set statically via LeadingDimension operator. For comparision, there's  also
// BLASWithoutLD type without that operator, and it is run with dynamic leading dimensions that are passed
// as arguments to the execute() method.
template<unsigned int Arch>
int simple_gemm_with_leading_dimensions() {
    // Parameters m, n, k define the dimensions of matrices A, B, and C
    constexpr unsigned int m = 30;
    constexpr unsigned int n = 31;
    constexpr unsigned int k = 33;

    // If matrix A is not transposed its logical dimensions are: [m, k] (m rows, k columns)
    // If matrix B is not transposed its logical dimensions are: [k, n]
    // If matrix A is transposed its logical dimensions are: [k, m]
    // If matrix B is transposed its logical dimensions are: [n, k]
    // The dimensions of matrix C are: [m, n]
    constexpr auto a_transpose_mode = cublasdx::transpose_mode::non_transposed;
    constexpr auto b_transpose_mode = cublasdx::transpose_mode::transposed;

    // Leading dimensions defined for matrices A, B, C

    // If A is not transposed, A is a matrix of real dimensions [lda, k] with lda >= m
    // If A is transposed, A is a matrix of real dimensions [lda, m] with lda >= k
    constexpr unsigned int lda = 32;
    // If B is not transposed, B is a matrix of real dimensions [ldb, n] with ldb >= k
    // If B is transposed, B is a matrix of real dimensions [ldb, k] with ldb >= n
    constexpr unsigned int ldb = 33;
    // C is a matrix of real dimensions [ldc, n] with ldx >= m
    constexpr unsigned int ldc = 31;

    // Selected CUDA block size (2D)
    constexpr dim3 block_dim(16, 16, 1);

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. The transpose modes of A and B matrices are set.
    // 4. Block operator informs that GEMM should be performed on CUDA block level.
    // 5. BlockDim operator sets CUDA block dimensions (in this cases 2D) that the kernel will be executed with.
    // 6. Targeted CUDA compute capability is selected with SM operator.
    using BLASWithoutLD =
        decltype(cublasdx::Size<m, n, k>() +
                 cublasdx::Precision<double>() +
                 cublasdx::Type<cublasdx::type::real>() +
                 cublasdx::Function<cublasdx::function::MM>() +
                 cublasdx::TransposeMode<a_transpose_mode, b_transpose_mode>() +
                 cublasdx::Block() +
                 cublasdx::BlockDim<block_dim.x, block_dim.y, block_dim.z>() +
                 cublasdx::SM<Arch>());
    // 6. Leading dimensions for matrices A, B, C
    using BLAS = decltype(BLASWithoutLD() + cublasdx::LeadingDimension<lda, ldb, ldc>());
    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
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
    CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, BLAS::c_size * sizeof(value_type)));

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
        cudaFuncSetAttribute(gemm_kernel<BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, BLAS::shared_memory_size));

    // Execute kernel with GEMM with defined leading dimensions (known at compile time)
    gemm_kernel<BLAS><<<1, BLAS::block_dim, BLAS::shared_memory_size>>>(alpha, a, b, beta, c, output);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<value_type> host_output(BLAS::c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), output, BLAS::c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Perform same GEMM but without defined leading dimensions, instead use dynamic leading dimensions.
    // Dynamic leading dimensions are the same are for BLAS which will enable us to reuse the same device buffers.

    // Increase max dynamic shared memory for the kernel if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(gemm_kernel_dynamic_ld<BLASWithoutLD>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             BLASWithoutLD::get_shared_memory_size(lda, ldb, ldc)));

    // Execute kernel
    gemm_kernel_dynamic_ld<BLASWithoutLD><<<1, BLASWithoutLD::block_dim, BLASWithoutLD::get_shared_memory_size(lda, ldb, ldc)>>>(
        alpha, a, lda, b, ldb, beta, c, ldc, output);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<value_type> host_output_dynamic_ld(BLAS::c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output_dynamic_ld.data(), output, BLAS::c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Calculate reference
    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a, host_b, beta, host_c);

    // Check both results against reference
    auto correct_static_ld  = example::check(host_output, reference_host_output);
    auto correct_dynamic_ld = example::check(host_output_dynamic_ld, reference_host_output);
    if (correct_static_ld && correct_dynamic_ld) {
        std::cout << "Success" << std::endl;
        return 0;
    }

    std::cout << "Success" << std::endl;
    return 0;
}

template<unsigned int Arch>
struct simple_gemm_with_leading_dimensions_functor {
    int operator()() { return simple_gemm_with_leading_dimensions<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_gemm_with_leading_dimensions_functor>();
}
