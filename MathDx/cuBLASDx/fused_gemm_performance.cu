#include <iostream>
#include <vector>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

template<class BLAS1, class BLAS2, class ValueType = typename BLAS1::value_type>
__launch_bounds__(BLAS1::max_threads_per_block) __global__
void fused_gemm_kernel(const ValueType alpha1,
                       const ValueType* a,
                       const ValueType* b,
                       const ValueType  beta1,
                       const ValueType* c,
                       const ValueType  alpha2,
                       const ValueType* d,
                       const ValueType  beta2,
                       const ValueType* f,
                       ValueType*       output) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];
    constexpr unsigned int block_size = BLAS1::block_dim.x * BLAS1::block_dim.y * BLAS1::block_dim.z;

    static_assert(std::is_same_v<value_type, example::value_type_t<BLAS2>>, "BLAS1 and BLAS2 must have the same type and precision");
    static_assert((BLAS1::c_dim == BLAS2::a_dim), "The dimensions of C matrix are different in BLAS1 and BLAS2");

    // Matrix C is the first in shared memory, because it's reused in the 2nd GEMM. Moreover,
    // matrices A and B might have different sizes than F and D.
    value_type* smem_c = reinterpret_cast<value_type*>(smem);
    value_type* smem_a = reinterpret_cast<value_type*>(smem) + BLAS1::c_size;
    value_type* smem_b = reinterpret_cast<value_type*>(smem) + BLAS1::c_size + BLAS1::a_size;

    example::io<BLAS1>::a_fast_load<block_size>(smem_a, a);
    example::io<BLAS1>::b_fast_load<block_size>(smem_b, b);
    example::io<BLAS1>::c_fast_load<block_size>(smem_c, c);
    __syncthreads();

    BLAS1().execute(alpha1, smem_a, smem_b, beta1, smem_c);
    __syncthreads();

    static_assert((BLAS1::c_size == BLAS2::a_size), "The sizes of C matrix are different in BLAS1 and BLAS2");
    value_type* smem_d = smem_c + BLAS2::a_size;
    value_type* smem_f = smem_c + BLAS2::a_size + BLAS2::b_size;

    example::io<BLAS2>::b_fast_load<block_size>(smem_d, d);
    example::io<BLAS2>::c_fast_load<block_size>(smem_f, f);
    __syncthreads();

    BLAS2().execute(alpha2, smem_c, smem_d, beta2, smem_f);

    __syncthreads();
    example::io<BLAS2>::c_fast_store<block_size>(output, smem_f);
}

template<class BLAS1, class BLAS2, class ValueType = typename BLAS1::value_type>
double measure_cublasdx(unsigned int kernel_warm_up_repeats,
                        unsigned int kernel_repeats,
                        const ValueType alpha1,
                        const ValueType* a,
                        const ValueType* b,
                        const ValueType  beta1,
                        const ValueType* c,
                        const ValueType  alpha2,
                        const ValueType* d,
                        const ValueType  beta2,
                        const ValueType* f,
                        ValueType*       output,
                        cudaStream_t     stream) {

    // Increase max dynamic shared memory for the kernel if needed.
    const auto shared_memory = std::max<size_t>(BLAS1::shared_memory_size, BLAS2::shared_memory_size);
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(fused_gemm_kernel<BLAS1, BLAS2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory));

    // Execute kernel.
    double time = example::measure::execution(
        [&](cudaStream_t stream) {
            fused_gemm_kernel<BLAS1, BLAS2>
                <<<1, BLAS1::block_dim, shared_memory, stream>>>(alpha1, a, b, beta1, c, alpha2, d, beta2, f, output);
        },
        kernel_warm_up_repeats,
        kernel_repeats,
        stream);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    return time;
}

template<class BLAS1, class BLAS2, class ValueType=typename BLAS1::value_type>
double measure_cublas(unsigned int kernel_warm_up_repeats,
                      unsigned int kernel_repeats,
                      ValueType        alpha1,
                      const ValueType* a,
                      const ValueType* b,
                      ValueType        beta1,
                      ValueType*       c,
                      const ValueType  alpha2,
                      const ValueType* d,
                      const ValueType  beta2,
                      ValueType* f,
                      cudaStream_t stream) {

    static_assert(std::is_same_v<ValueType, example::value_type_t<BLAS2>>, "BLAS1 and BLAS2 must have the same type and precision");
    static_assert((BLAS1::c_dim == BLAS2::a_dim), "The dimensions of C matrix are different in BLAS1 and BLAS2");

    const unsigned int m1 = cublasdx::size_of<BLAS1>::m;
    const unsigned int n1 = cublasdx::size_of<BLAS1>::n;
    const unsigned int k1 = cublasdx::size_of<BLAS1>::k;

    const unsigned int lda1 = cublasdx::leading_dimension_of<BLAS1>::a;
    const unsigned int ldb1 = cublasdx::leading_dimension_of<BLAS1>::b;
    const unsigned int ldc1 = cublasdx::leading_dimension_of<BLAS1>::c;

    const unsigned int m2 = cublasdx::size_of<BLAS2>::m;
    const unsigned int n2 = cublasdx::size_of<BLAS2>::n;
    const unsigned int k2 = cublasdx::size_of<BLAS2>::k;

    const unsigned int lda2 = cublasdx::leading_dimension_of<BLAS2>::a;
    const unsigned int ldb2 = cublasdx::leading_dimension_of<BLAS2>::b;
    const unsigned int ldc2 = cublasdx::leading_dimension_of<BLAS2>::c;

    static_assert(example::is_complex<ValueType>() && std::is_same_v<float, typename example::get_precision<ValueType>::type>,
        "Type or precision is currently not supported for cuBLAS measurement.");

    //
    // cuBLAS
    //
    cublasHandle_t handle;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));

    const auto a_transpose = example::detail::get_cublas_transpose_mode(cublasdx::transpose_mode_of<BLAS1>::a_transpose_mode);
    const auto b_transpose = example::detail::get_cublas_transpose_mode(cublasdx::transpose_mode_of<BLAS1>::b_transpose_mode);

    const auto c_transpose = example::detail::get_cublas_transpose_mode(cublasdx::transpose_mode_of<BLAS2>::a_transpose_mode);
    const auto d_transpose = example::detail::get_cublas_transpose_mode(cublasdx::transpose_mode_of<BLAS2>::b_transpose_mode);

    cublasSetStream(handle, stream);

    double time_cublas = example::measure::execution(
        [&](cudaStream_t) {
            // C = alpha1 * A * B + beta1 * C
            CUBLAS_CHECK_AND_EXIT(cublasCgemm(handle,
                a_transpose,
                b_transpose,
                m1,
                n1,
                k1,
                reinterpret_cast<const cuComplex*>(&alpha1),
                reinterpret_cast<const cuComplex*>(a),
                lda1,
                reinterpret_cast<const cuComplex*>(b),
                ldb1,
                reinterpret_cast<const cuComplex*>(&beta1),
                reinterpret_cast<cuComplex*>(c),
                ldc1));
            // F = alpha2 * C * D + beta2 * F
            CUBLAS_CHECK_AND_EXIT(cublasCgemm(handle,
                c_transpose,
                d_transpose,
                m2,
                n2,
                k2,
                reinterpret_cast<const cuComplex*>(&alpha2),
                reinterpret_cast<const cuComplex*>(c),
                lda2,
                reinterpret_cast<const cuComplex*>(d),
                ldb2,
                reinterpret_cast<const cuComplex*>(&beta2),
                reinterpret_cast<cuComplex*>(f),
                ldc2));
        },
        kernel_warm_up_repeats, kernel_repeats, stream);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));

    return time_cublas;
}

// This example compares the performance of cuBLAS and cuBLASDx for the example described in
// "gemm_fusion.cu". Using cuBLASDx we fuse the chained matrix multiplication into one kernel,
// whereas we compose the final result using cuBLAS using two matrix multiplications.
//
//             1) C = alpha1 * (A * B) + beta1 * C
//             2) F = alpha2 * (C * D) + beta2 * F
//
// * A, B, C, D and F are matrices containing complex single precision floating-point values.
// * (alpha1, beta1) and (alpha2, beta2) and complex single precision floating-point values.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Then we compare the performance of the fused kernel in cuBLASDx with the
// performance of the two matrix multiplications needed to compute the result using cuBLAS.
//
// For cuBLASDx the number of threads participating in the GEMM operation is imposed by providing
// BlockDim operator in definition of the GEMM. If BlockDim operator is not used, cuBLASDx automatically
// selects number of threads. Block dimensions are provided via BLAS::block_dim trait.
//
// General Notes:
// * The matrix sizes should be small enough so that a single block is the optimal choice, since
//   cuBLASDx is limited to one using one block whereas cuBLAS can use a larger number of blocks.
//
// cuBLASDx Notes:
// * Both GEMM operations use the same number of threads, however, it's not a requirement.
// * It's important that the dimensions of the first and the 2nd GEMM are set in such a way that
//   the C matrix has the same dimensions in both operations.
template<unsigned int Arch>
int fused_gemm_performance() {
    // Parameters m1, n1, k1 define the dimensions of matrices A, B, and C.
    constexpr unsigned int m1          = 32;
    constexpr unsigned int n1          = 32;
    constexpr unsigned int k1          = 32;

    // Parameters m2, n2, k2 define the dimensions of matrices C, D and F.
    // Note: (m1, n1) and (m2, k2) must be equal as describe the same matrix (matrix C).
    constexpr unsigned int m2          = m1;
    constexpr unsigned int n2          = 32;
    constexpr unsigned int k2          = n1;

    // The logical dimensions of matrix A are: [m1, k1] (m rows, k columns).
    // The logical dimensions of matrix B are: [k1, n1].
    // The logical dimensions of matrix C are: [m1, n1].
    constexpr auto a_transpose_mode = cublasdx::transpose_mode::non_transposed;
    constexpr auto b_transpose_mode = cublasdx::transpose_mode::non_transposed;

    // The logical dimensions of matrix C are: [m2, k2] == [m1, n1].
    // The logical dimensions of matrix D are: [k2, n2].
    // The logical dimensions of matrix F are: [m2, n2].
    constexpr auto c_transpose_mode = cublasdx::transpose_mode::non_transposed;
    constexpr auto d_transpose_mode = cublasdx::transpose_mode::non_transposed;

    // Use the same block size for both GEMM operations, so BLAS1::block_dim == BLAS2::block_dim which
    // simplifies the example.
    constexpr unsigned int block_size = 512;

    // Choose the precision and data type. In this example, we limit ourselves single precision complex
    // data to keep the cuBLAS measurement code simple.
    using precision = float;
    constexpr auto type = cublasdx::type::complex;

    using BLAS1       = decltype(cublasdx::Size<m1, n1, k1>() +
                          cublasdx::Precision<precision>() +
                          cublasdx::Type<type>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::TransposeMode<a_transpose_mode, b_transpose_mode>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() +
                          cublasdx::SM<Arch>());
    using BLAS2       = decltype(cublasdx::Size<m2, n2, k2>() +
                          cublasdx::Precision<precision>() +
                          cublasdx::Type<type>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::TransposeMode<c_transpose_mode, d_transpose_mode>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() +
                          cublasdx::SM<Arch>());
    #if CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using value_type = example::value_type_t<BLAS1>;
    #else
    using value_type = typename BLAS1::value_type;
    #endif

    // Set the beta values (beta1 and beta2) for the two GEMMs to 0. since cuBLAS accumulates into the result and
    // we perform multiple repeats.
    // alpha and beta for the first GEMM.
    value_type alpha1 = example::make_value<value_type>(1., 2.);
    value_type beta1  = example::make_value<value_type>(0.);

    // alpha and beta for the second GEMM.
    value_type alpha2 = example::make_value<value_type>(3., 4.);
    value_type beta2  = example::make_value<value_type>(0.);

    // Allocate device memory for a, b, c, d, f and output.
    value_type* inputs;
    value_type* output;
    auto inputs_size       = BLAS1::a_size + BLAS1::b_size + BLAS1::c_size + BLAS2::b_size + BLAS2::c_size;
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, BLAS2::c_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + (BLAS1::a_size);
    value_type* c     = b + (BLAS1::b_size); // C matrix for BLAS1, A matrix for BLAS2
    value_type* d     = c + (BLAS1::c_size); // D is B matrix for BLAS2
    value_type* f     = d + (BLAS2::b_size); // F is C matrix for BLAS2

    // Fill the A, B, C matrices with random values.
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, BLAS1::a_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, BLAS1::b_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, BLAS1::c_size);
    auto host_d = example::get_random_data<value_type>(1.0, 2.0, BLAS2::b_size);
    auto host_f = example::get_random_data<value_type>(1.0, 10.0, BLAS2::c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), BLAS1::a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), BLAS1::b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), BLAS1::c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(d, host_d.data(), BLAS2::b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(f, host_f.data(), BLAS2::c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    const unsigned int kernel_repeats = 100;
    const unsigned int kernel_warm_up_repeats = 1;
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))

    // Measure cuBLASDx performance.
    double time_cublasdx =
        measure_cublasdx<BLAS1, BLAS2>(kernel_warm_up_repeats, kernel_repeats, alpha1, a, b, beta1, c, alpha2, d, beta2, f, output, stream);

    // Measure cuBLAS performance.
    double time_cublas =
        measure_cublas<BLAS1, BLAS2>(kernel_warm_up_repeats, kernel_repeats, alpha1, a, b, beta1, c, alpha2, d, beta2, f, stream);

    // Write performance data.
    using cublasdx::size_of;
    std::cout << "m1, n1, k1: " << size_of<BLAS1>::m << ", " << size_of<BLAS1>::n << ", " << size_of<BLAS1>::k
              << std::endl;
    std::cout << "m2, n2, k2: " << size_of<BLAS2>::m << ", " << size_of<BLAS2>::n << ", " << size_of<BLAS2>::k
              << std::endl;
    std::cout << "Type: " << example::type_string<value_type>() << std::endl;
    std::cout << "Precision: " << example::precision_string<value_type>() << std::endl;

    std::cout << "\ncuBLASDx (fused kernel)\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Avg time [ms]  = " << time_cublasdx / kernel_repeats << "\n";

    std::cout << "\ncuBLAS\n";
    std::cout << "Avg time [ms]  = " << time_cublas / kernel_repeats << "\n";

    // Free resources.
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    return 0;
}

template<unsigned int Arch>
struct fused_gemm_performance_functor {
    int operator()() {
        return fused_gemm_performance<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner<fused_gemm_performance_functor>();
}
