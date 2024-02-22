#ifndef CUBLASDX_EXAMPLE_SINGLE_GEMM_PERFORMANCE_HPP_
#define CUBLASDX_EXAMPLE_SINGLE_GEMM_PERFORMANCE_HPP_

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cublasdx.hpp>

#include "block_io.hpp"
#include "common.hpp"
#include "reference.hpp"

template<class GEMM, class ValueType = typename GEMM::value_type>
__launch_bounds__(GEMM::max_threads_per_block) __global__
void gemm_kernel(const ValueType* a,
                 const ValueType* b,
                 const ValueType* c,
                 const ValueType  alpha,
                 const ValueType  beta,
                 ValueType*       output,
                 unsigned int repeats) {
    using value_type = ValueType;
    extern __shared__ __align__(16) char smem[];

    value_type* smem_a = reinterpret_cast<value_type*>(smem);
    value_type* smem_b = reinterpret_cast<value_type*>(smem) + GEMM::a_size;
    value_type* smem_c = reinterpret_cast<value_type*>(smem) + GEMM::a_size + GEMM::b_size;

    example::io<GEMM>::a_load(smem_a, a);
    example::io<GEMM>::b_load(smem_b, b);
    example::io<GEMM>::c_load(smem_c, c);
    __syncthreads();

    for (unsigned int i = 0; i < repeats; i++) {
        GEMM().execute(alpha, smem_a, smem_b, beta, smem_c);
    }

    __syncthreads();
    example::io<GEMM>::c_store(output, smem_c);
}

template<class GEMM, unsigned int Arch, unsigned int BlockSize, bool UseSuggestedLD>
int benchmark_single_gemm(const cudaStream_t& stream, bool verbose = false) {
    using namespace cublasdx;

    static constexpr unsigned int inside_repeats = 4000;
    static constexpr unsigned int kernel_repeats = 1;
    static constexpr unsigned int kernel_warm_up_repeats = 1;

    using suggested_ld = suggested_leading_dimension_of_t<GEMM, Arch>;
    constexpr bool set_block_size{BlockSize > 0};
    using gemm_base_type = std::conditional_t<set_block_size, decltype(GEMM() + BlockDim<BlockSize>()), GEMM>;
    using gemm_type      = std::conditional_t<UseSuggestedLD, decltype(gemm_base_type() + suggested_ld()), gemm_base_type>;
    using value_type = typename gemm_type::value_type;

    // Allocate device memory for A, B, C.
    value_type* inputs;
    value_type* output;

    // BLAS::a_size/b_size/c_size include padding.
    auto inputs_size       = gemm_type::a_size + gemm_type::b_size + gemm_type::c_size;
    auto inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, gemm_type::c_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + gemm_type::a_size;
    value_type* c     = b + gemm_type::b_size;

    value_type  alpha = example::make_value<value_type>(1.f);
    value_type  beta  = example::make_value<value_type>(0.f);

    // Fill the A, B, C matrices with random values.
    auto host_a = example::get_random_data<value_type>(0.1, 1.0, gemm_type::a_size);
    auto host_b = example::get_random_data<value_type>(0.1, 1.0, gemm_type::b_size);
    auto host_c = example::get_random_data<value_type>(0.1, 1.0, gemm_type::c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), gemm_type::a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), gemm_type::b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), gemm_type::c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed.
    CUDA_CHECK_AND_EXIT(
        cudaFuncSetAttribute(gemm_kernel<gemm_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_type::shared_memory_size));

    // Measure performance of N trials.
    double ms_n1 = example::measure::execution(
        [&](cudaStream_t stream) {
            gemm_kernel<gemm_type><<<1, gemm_type::block_dim, gemm_type::shared_memory_size, stream>>>(
                a, b, c, alpha, beta, output, inside_repeats);
        },
        kernel_warm_up_repeats, kernel_repeats, stream);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Measure performance of 2N trials.
    double ms_n2 = example::measure::execution(
        [&](cudaStream_t stream) {
            gemm_kernel<gemm_type><<<1, gemm_type::block_dim, gemm_type::shared_memory_size, stream>>>(
                a, b, c, alpha, beta, output, 2 * inside_repeats);
        },
        kernel_warm_up_repeats, kernel_repeats, stream);

    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    double time_n     = ms_n2 - ms_n1;
    double avg_time_n = time_n / (inside_repeats * kernel_repeats);

    double gflops = example::gemm_flops<value_type>(size_of<gemm_type>::m, size_of<gemm_type>::n, size_of<gemm_type>::k)
                          / avg_time_n / 1000000.;

    // Copy results back to host.
    std::vector<value_type> host_output(gemm_type::c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output, gemm_type::c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory.
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    if (verbose) {
        const auto [lda, ldb, ldc] = cublasdx::leading_dimension_of_v<gemm_type>;
        std::cout << "m, n, k: " << size_of<gemm_type>::m << ", " << size_of<gemm_type>::n << ", " << size_of<gemm_type>::k
                  << std::endl;
        std::cout << "Type: " << example::type_string<value_type>() << std::endl;
        std::cout << "Precision: " << example::precision_string<value_type>() << std::endl;
        std::cout << "Block size: " << gemm_type::block_dim.x << std::endl;
        std::cout << "Leading dimensions: " << lda << ", " << ldb << ", " << ldc << std::endl;
        std::cout << "Shared memory: " << gemm_type::shared_memory_size << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Avg time [ms]: " << avg_time_n << std::endl;
        std::cout << "Time (all) [ms]: " << time_n << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    } else {
        std::cout << "(" << size_of<gemm_type>::m << ", " << size_of<gemm_type>::n << ", " << size_of<gemm_type>::k << ") "
                  << example::precision_string<value_type>() << " precision " <<  example::type_string<value_type>()
                  << ": " << std::fixed << std::setprecision(4) << gflops << " GFLOPS, " << avg_time_n << " ms." << std::endl;
    }
    // Calculate reference solution.
    const auto [lda, ldb, ldc] = cublasdx::leading_dimension_of_v<gemm_type>;
    auto reference_host_output = example::reference_gemm<gemm_type>(alpha, host_a, lda, host_b, ldb, beta, host_c, ldc);

    // Check against reference (requires beta = 0. to avoid accumulating into C due to the repeated runs on the GPU).
    if (example::check(host_output, reference_host_output)) {
        std::cout << "The results are verified to be correct." << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

#endif // CUBLASDX_EXAMPLE_SINGLE_GEMM_PERFORMANCE_HPP_
