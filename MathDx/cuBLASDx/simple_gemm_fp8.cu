#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "reference.hpp"

template<class BLAS>
__launch_bounds__(BLAS::max_threads_per_block) __global__ void gemm_kernel(const typename BLAS::a_value_type* a,
                                                                           const typename BLAS::b_value_type* b,
                                                                           typename BLAS::c_value_type*       c,
                                                                           const typename BLAS::c_value_type  alpha,
                                                                           const typename BLAS::c_value_type  beta) {
    extern __shared__ __align__(16) char smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    auto [smem_a, smem_b, smem_c] = BLAS::slice_shared_memory(smem);
    auto a_shared_tensor          = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor          = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());
    auto c_shared_tensor          = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    BLAS().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);

    __syncthreads();
    cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, c_global_tensor);
}

// This is an example of general matrix-matrix multiplication (GEMM) with 8-bit floating point format
// performanced in a single CUDA block:
//
//              C = alpha * A * B + beta * C
//
// * A is the input matrix containing complex FP8 e4m3 data format, B is another input matrix containing complex FP8 e5m2 data format,
// * C is the output matrix containing complex FP32 data format, and alpha and beta are complex FP32 values.
// If (A, B, C) are any of the following data formats:
//    (e4m3, e4m3, FP32)
//    (e4m3, e5m2, FP32)
//    (e5m2, e4m3, FP32)
//    (e5m2, e5m2, FP32)
// and GEMM is excuted on SM89 or higher GPUs with CUDA NVCC 12.4+, then matrix multiply-and-accumulation (MMA) instructions are used.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory. The results are verified against cuBLAS.
//
template<unsigned int Arch>
int simple_gemm() {
#if CUBLASDX_EXAMPLE_SUPPORTS_FP8
    // Parameters m, n, k define the dimensions of matrices A, B, and C
    constexpr unsigned int m = 64;
    constexpr unsigned int n = 64;
    constexpr unsigned int k = 64;

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. The data arrangements of A, B matrices are set (C is defaulted to column major).
    //    - Optional
    // 4. The data alignment of A, B and C matrices is set to the max accepted value using alias MaxAlignment.
    //    - Optional
    // 4. Block operator informs that GEMM should be performed on CUDA block level.
    // 5. Targeted CUDA compute capability is selected with SM operator.
    using BLAS = decltype(cublasdx::Size<m, n, k>() + cublasdx::Precision<__nv_fp8_e4m3, __nv_fp8_e5m2, float>() +
                          cublasdx::Type<cublasdx::type::complex>() + cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
                          cublasdx::Alignment<2, 2, 8>() + cublasdx::Block() + cublasdx::SM<Arch>());

    // Allocate managed memory for a, b, c
#ifdef CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using TA = typename example::a_value_type_t<BLAS>;
    using TB = typename example::b_value_type_t<BLAS>;
    using TC = typename example::c_value_type_t<BLAS>;
#else
    using TA = typename BLAS::a_value_type;
    using TB = typename BLAS::b_value_type;
    using TC = typename BLAS::c_value_type;
#endif
    TA* a;
    TB* b;
    TC* c;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&a, global_a_size * sizeof(TA)));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&b, global_b_size * sizeof(TB)));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&c, global_c_size * sizeof(TC)));

    auto alpha = TC(1.0, 1.0);
    auto beta  = TC(2.0, 1.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<TA>(0.1, 1.0, global_a_size);
    auto host_b = example::get_random_data<TB>(0.1, 1.0, global_b_size);
    auto host_c = example::get_random_data<TC>(0.1, 1.0, global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(TA), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_a_size * sizeof(TB), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_a_size * sizeof(TC), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, BLAS::shared_memory_size));

    // Execute kernel
    gemm_kernel<BLAS><<<1, BLAS::block_dim, BLAS::shared_memory_size>>>(a, b, c, alpha, beta);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<TC> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), c, global_c_size * sizeof(TC), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(a));
    CUDA_CHECK_AND_EXIT(cudaFree(b));
    CUDA_CHECK_AND_EXIT(cudaFree(c));

    // Calculate reference
    auto reference_host_output = example::reference_gemm<BLAS, TA, TB, TC>(alpha, host_a, host_b, beta, host_c);

    // Check against reference
    if (example::check(host_output, reference_host_output)) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
#else
    std::cout << "Compiler version does not support FP8 datatype, skip the test" << std::endl;
    return 0;
#endif
}

template<unsigned int Arch>
struct simple_gemm_functor {
    int operator()() { return simple_gemm<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_gemm_functor>();
}
