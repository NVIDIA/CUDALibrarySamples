#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "common.hpp"
#include "block_io.hpp"
#include "reference.hpp"

template<class BLAS, class TA, class TB, class TC>
__launch_bounds__(BLAS::max_threads_per_block) __global__
void gemm_kernel(const TA* a, const TB* b, const TC* c, const TC alpha, const TC beta, TC* output) {

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

// This is an example of mixed-precision general matrix-matrix multiplication (GEMM) performed
// in a single CUDA block:
//
//              C = alpha * A * B + beta * C
//
// A, B, and C are matrices. Mixed percisions are supported.
// Note that alpha and beta are expected to have the same precision and type (real or complex value)
// as the matrix C elements.
//
// Input data A, B, and C is generated on host using random number generators, and later copied to
// the device global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory. The results are verified against cuBLAS.

template<unsigned int Arch>
int simple_gemm_mixed_precision() {

    using PA = cublasdx::tfloat32_t;
    using PB = cublasdx::tfloat32_t;
    using PC = float;

    constexpr unsigned int m = 16;
    constexpr unsigned int n = 24;
    constexpr unsigned int k = 32;

    constexpr unsigned int block_size = 256;

    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<PA, PB, PC>() +
                          cublasdx::Type<cublasdx::type::complex>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
                          cublasdx::Alignment<16, 16, 16>()+
                          cublasdx::Block() +
                          cublasdx::BlockDim<block_size>() + cublasdx::SM<Arch>());

#ifdef CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using TA = typename example::a_value_type_t<BLAS>;
    using TB = typename example::b_value_type_t<BLAS>;
    using TC = typename example::c_value_type_t<BLAS>;
#else
    using TA = typename BLAS::a_value_type;
    using TB = typename BLAS::b_value_type;
    using TC = typename BLAS::c_value_type;
#endif

    std::cout << "Precisions: A is " << example::precision_string<TA>() << ", B is " << example::precision_string<TB>()
              << " and C is " << example::precision_string<TC>() << " \nType: A/B/C is "
              << example::type_string<TA>() << "\n";

    static_assert(std::is_same<typename cublasdx::precision_of<BLAS>::a_type, PA>::value, "TA and PA do not match");
    static_assert(std::is_same<typename cublasdx::precision_of<BLAS>::b_type, PB>::value, "TB and PB do not match");
    static_assert(std::is_same<typename cublasdx::precision_of<BLAS>::c_type, PC>::value, "TC and PC do not match");

    // Allocate managed memory for a, b, c, and output
    TA* a      = nullptr;
    TB* b      = nullptr;
    TC* c      = nullptr;
    TC* output = nullptr;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&a, global_a_size * sizeof(TA)));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&b, global_b_size * sizeof(TB)));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&c, global_c_size * sizeof(TC)));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, global_a_size * sizeof(TC)));

    TC alpha = TC(1.0, 0.0);
    TC beta  = TC(2.0, 0.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<TA>(0.1, 1.0, global_a_size);
    auto host_b = example::get_random_data<TB>(0.1, 1.0, global_b_size);
    auto host_c = example::get_random_data<TC>(0.1, 1.0, global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(TA), cudaMemcpyDefault));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b_size * sizeof(TB), cudaMemcpyDefault));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c_size * sizeof(TC), cudaMemcpyDefault));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Instantiate kernel
    auto kernel = gemm_kernel<BLAS, TA, TB, TC>;

    // Execute kernel
    kernel<<<1, BLAS::block_dim, BLAS::shared_memory_size, 0>>>(a, b, c, alpha, beta, output);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<TC> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), output, global_c_size * sizeof(TC), cudaMemcpyDefault));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(a));
    CUDA_CHECK_AND_EXIT(cudaFree(b));
    CUDA_CHECK_AND_EXIT(cudaFree(c));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // Calculate reference
    std::vector<TC> host_a_ref = example::convert<TA, TC>(host_a);
    std::vector<TC> host_b_ref = example::convert<TB, TC>(host_b);
    std::vector<TC> host_c_ref = host_c;
    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a_ref, host_b_ref, beta, host_c_ref);

    // Check against reference
    if (example::check(host_output, reference_host_output)) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

template<unsigned int Arch>
struct simple_gemm_functor {
    int operator()() { return simple_gemm_mixed_precision<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_gemm_functor>();
}
