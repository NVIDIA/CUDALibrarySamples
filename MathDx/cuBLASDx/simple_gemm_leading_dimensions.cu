#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "reference.hpp"

template<class BLAS, class ValueType = typename example::uniform_value_type_t<BLAS>>
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

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    auto [smem_a, smem_b, smem_c] = BLAS::slice_shared_memory(smem);
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

template<class BLASWithoutLD, class ValueType = typename example::uniform_value_type_t<BLASWithoutLD>>
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

    auto a_global_tensor = cublasdx::make_tensor(a, BLASWithoutLD::get_layout_gmem_a(lda));
    auto b_global_tensor = cublasdx::make_tensor(b, BLASWithoutLD::get_layout_gmem_b(ldb));
    auto c_global_tensor = cublasdx::make_tensor(c, BLASWithoutLD::get_layout_gmem_c(ldc));

    auto [smem_a, smem_b, smem_c] = BLASWithoutLD::slice_shared_memory(smem, lda, ldb, ldc);
    auto ta = cublasdx::make_tensor(smem_a, BLASWithoutLD::get_layout_smem_a(lda));
    auto tb = cublasdx::make_tensor(smem_b, BLASWithoutLD::get_layout_smem_b(ldb));
    auto tc = cublasdx::make_tensor(smem_c, BLASWithoutLD::get_layout_smem_c(ldc));

    using alignment = cublasdx::alignment_of<BLASWithoutLD>;
    cublasdx::copy<BLASWithoutLD, alignment::a>(a_global_tensor, ta);
    cublasdx::copy<BLASWithoutLD, alignment::b>(b_global_tensor, tb);
    cublasdx::copy<BLASWithoutLD, alignment::c>(c_global_tensor, tc);
    cublasdx::copy_wait();

    BLASWithoutLD().execute(alpha, smem_a, lda, smem_b, ldb, beta, smem_c, ldc);

    __syncthreads();
    auto tgout = cublasdx::make_tensor(output, BLASWithoutLD::get_layout_gmem_c(ldc));
    cublasdx::copy<BLASWithoutLD, alignment::c>(tc, tgout);
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

    // If matrix A is column-major (or not transposed in BLAS nomenclature) its logical dimensions are: [m, k] (m rows, k columns)
    // If matrix B is column-major its logical dimensions are: [k, n]
    // If matrix A is row-major (or transposed in BLAS nomenclature) its logical dimensions are: [k, m]
    // If matrix B is row-major its logical dimensions are: [n, k]
    // The dimensions of matrix C are: [m, n]
    constexpr auto a_arrangement = cublasdx::col_major;
    constexpr auto b_arrangement = cublasdx::row_major;
    constexpr auto c_arrangement = cublasdx::row_major;

    // Leading dimensions defined for matrices A, B, C

    // If A is column-major, A is a matrix of real dimensions [lda, k] with lda >= m
    // If A is row-major, A is a matrix of real dimensions [lda, m] with lda >= k
    constexpr unsigned int lda = 32;
    // If B is column-major, B is a matrix of real dimensions [ldb, n] with ldb >= k
    // If B is row-major, B is a matrix of real dimensions [ldb, k] with ldb >= n
    constexpr unsigned int ldb = 33;
    // C is a matrix of real dimensions [ldc, n] with ldx >= m
    constexpr unsigned int ldc = 31;

    // Selected CUDA block size (2D)
    constexpr dim3 block_dim(16, 16, 1);

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. The data arrangements of A, B matrices are set (C is defaulted to column major).
    //    - Optional
    // 4. The data alignment of A, B and C matrices is set to the max accepted value.
    //    - Optional
    // 5. Block operator informs that GEMM should be performed on CUDA block level.
    // 6. BlockDim operator sets CUDA block dimensions (in this cases 2D) that the kernel will be executed with.
    // 7. Targeted CUDA compute capability is selected with SM operator.
    using BLASWithoutLD =
        decltype(cublasdx::Size<m, n, k>() +
                 cublasdx::Precision<double>() +
                 cublasdx::Type<cublasdx::type::real>() +
                 cublasdx::Function<cublasdx::function::MM>() +
                 cublasdx::Arrangement<a_arrangement, b_arrangement, c_arrangement>() +
                 cublasdx::Block() +
                 cublasdx::BlockDim<block_dim.x, block_dim.y, block_dim.z>() +
                 cublasdx::SM<Arch>());
    // 6. Leading dimensions for matrices A, B, C
    using BLAS = decltype(BLASWithoutLD() + cublasdx::LeadingDimension<lda, ldb, ldc>());
    using value_type = typename example::uniform_value_type_t<BLAS>;

    // Allocate managed memory for a, b, c, and output
    value_type* inputs;
    value_type* output;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    auto inputs_size       = global_a_size + global_b_size + global_c_size;
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, global_c_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + (global_a_size);
    value_type* c     = b + (global_b_size);
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
        cudaFuncSetAttribute(gemm_kernel<BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, BLAS::shared_memory_size));

    // Execute kernel with GEMM with defined leading dimensions (known at compile time)
    gemm_kernel<BLAS><<<1, BLAS::block_dim, BLAS::shared_memory_size>>>(alpha, a, b, beta, c, output);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<value_type> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), output, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
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
    std::vector<value_type> host_output_dynamic_ld(global_c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output_dynamic_ld.data(), output, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
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
