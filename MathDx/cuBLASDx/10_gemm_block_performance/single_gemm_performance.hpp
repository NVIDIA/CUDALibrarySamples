#ifndef CUBLASDX_EXAMPLE_SINGLE_GEMM_PERFORMANCE_HPP_
#define CUBLASDX_EXAMPLE_SINGLE_GEMM_PERFORMANCE_HPP_

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

template<class GEMM, class AInputType, class BInputType, class CInputType>
__launch_bounds__(GEMM::max_threads_per_block) __global__ void gemm_kernel(const AInputType*                  a,
                                                                           const BInputType*                  b,
                                                                           const CInputType*                  c,
                                                                           const typename GEMM::c_value_type  alpha,
                                                                           const typename GEMM::c_value_type  beta,
                                                                           CInputType*                        output,
                                                                           unsigned int                       repeats) {
    extern __shared__ __align__(16) char smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<GEMM, AInputType, BInputType>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::suggest_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::suggest_layout_smem_b());

    using blas_alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, blas_alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, blas_alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    auto partitioner = GEMM::suggest_partitioner();

    // Initialize accumulator
    auto c_frag = partitioner.make_accumulator_fragment();
    cublasdx::clear(c_frag);

    // Download d data
    auto d_frag_io = cublasdx::make_fragment_like<CInputType>(c_frag);
    auto d_frag_compute = cublasdx::make_fragment_like(c_frag);
    cublasdx::copy_fragment<blas_alignment::c>(c_global_tensor, d_frag_io, partitioner);
    cublasdx::transform(d_frag_io, d_frag_compute, example::converter<typename GEMM::c_value_type>{});

    #pragma unroll 1
    for (unsigned int i = 0; i < repeats; i++) {
        GEMM().execute(a_shared_tensor, b_shared_tensor, c_frag,
                       example::converter<typename GEMM::a_value_type>{},
                       example::converter<typename GEMM::b_value_type>{});
        cublasdx::axpby(alpha, c_frag, beta, d_frag_compute);
    }

    auto out_global_tensor = cublasdx::make_tensor(output, GEMM::get_layout_gmem_c());
    cublasdx::transform(d_frag_compute, d_frag_io, example::converter<CInputType>{});
    cublasdx::copy_fragment<blas_alignment::c>(d_frag_io, out_global_tensor, partitioner);
}

template<class GEMM, class AInputType, class BInputType, class CInputType,
         unsigned int Arch, unsigned int BlockSize, bool UseSuggestedLD>
int benchmark_mixed_precision_gemm(const cudaStream_t& stream, bool verbose = false) {
    using namespace cublasdx;

    static constexpr unsigned int inside_repeats         = 1000;
    static constexpr unsigned int kernel_repeats         = 5;
    static constexpr unsigned int kernel_warm_up_repeats = 5;

    using suggested_ld = suggested_leading_dimension_of_t<GEMM, Arch>;
    constexpr bool set_block_size {BlockSize > 0};
    using gemm_base_type = std::conditional_t<set_block_size, decltype(GEMM() + BlockDim<BlockSize>()), GEMM>;
    using gemm_type = std::conditional_t<UseSuggestedLD, decltype(gemm_base_type() + suggested_ld()), gemm_base_type>;

    using TA = AInputType;
    using TB = BInputType;
    using TC = CInputType;

    // Allocate device memory for A, B, C.
    TA* a;
    TB* b;
    TC* c;
    TC* output;

    constexpr auto global_a_size = example::global_memory_size_of<gemm_type>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<gemm_type>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<gemm_type>::c_size;

    CUDA_CHECK_AND_EXIT(cudaMalloc(&a, global_a_size * sizeof(TA)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&b, global_b_size * sizeof(TB)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&c, global_c_size * sizeof(TC)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&output, global_c_size * sizeof(TC)));

    auto alpha = example::make_value<typename GEMM::c_value_type>(1.f);
    // Beta has to be zero for the check to pass, as gemm is done repeatedly
    auto beta  = example::make_value<typename GEMM::c_value_type>(2.f);

    // Fill the A, B, C matrices with random values.
    auto host_a = example::get_random_data<TA>(-0.1, 0.1, global_a_size);
    auto host_b = example::get_random_data<TB>(-0.1, 0.1, global_b_size);
    auto host_c = example::get_random_data<TC>(-0.1, 0.1, global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(TA), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b_size * sizeof(TB), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c_size * sizeof(TC), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed.
    auto shared_memory_size = cublasdx::get_shared_storage_size_ab<gemm_type, TA, TB>();
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        gemm_kernel<gemm_type, TA, TB, TC>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    // Correctness run
    gemm_kernel<gemm_type, TA, TB, TC><<<1, gemm_type::block_dim, shared_memory_size, stream>>>(
        a, b, c, alpha, beta, output, 1);

    std::vector<TC> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), output, global_c_size * sizeof(TC), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Measure performance of N trials.
    double ms_n1 = example::measure::execution(
        [&](cudaStream_t stream) {
            gemm_kernel<gemm_type, TA, TB, TC><<<1, gemm_type::block_dim, shared_memory_size, stream>>>(
                a, b, c, alpha, beta, output, inside_repeats);
        },
        kernel_warm_up_repeats,
        kernel_repeats,
        stream);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Measure performance of 2N trials.
    double ms_n2 = example::measure::execution(
        [&](cudaStream_t stream) {
            gemm_kernel<gemm_type, TA, TB, TC><<<1, gemm_type::block_dim, shared_memory_size, stream>>>(
                a, b, c, alpha, beta, output, 2 * inside_repeats);
        },
        kernel_warm_up_repeats,
        kernel_repeats,
        stream);

    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    double time_n     = ms_n2 - ms_n1;
    double avg_time_n = time_n / (inside_repeats * kernel_repeats);

    double gflops =
        example::gemm_flops<TA, TB, TC>(size_of<gemm_type>::m, size_of<gemm_type>::n, size_of<gemm_type>::k) /
        avg_time_n / 1000000.;

    // Free device memory.
    CUDA_CHECK_AND_EXIT(cudaFree(a));
    CUDA_CHECK_AND_EXIT(cudaFree(b));
    CUDA_CHECK_AND_EXIT(cudaFree(c));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    if (verbose) {
        const auto [lda, ldb, ldc] = cublasdx::leading_dimension_of_v<gemm_type>;
        std::cout << "m, n, k: " << size_of<gemm_type>::m << ", " << size_of<gemm_type>::n << ", "
                  << size_of<gemm_type>::k << std::endl;
        std::cout << "Type: " << example::type_string<TA>() << std::endl;
        std::cout << "A Precision: " << example::precision_string<TA>() << std::endl;
        std::cout << "B Precision: " << example::precision_string<TB>() << std::endl;
        std::cout << "C Precision: " << example::precision_string<TC>() << std::endl;
        std::cout << "Block size: " << gemm_type::block_dim.x << ", " << gemm_type::block_dim.y <<  ", " << gemm_type::block_dim.z << std::endl;
        std::cout << "Leading dimensions: " << lda << ", " << ldb << ", " << ldc << std::endl;
        std::cout << "Shared memory: " << shared_memory_size << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Avg time [ms]: " << avg_time_n << std::endl;
        std::cout << "Time (all) [ms]: " << time_n << std::endl;
        std::cout << "Performance [GFLOPS]: " << gflops << std::endl;
    } else {
        std::cout << "(" << size_of<gemm_type>::m << ", " << size_of<gemm_type>::n << ", " << size_of<gemm_type>::k
                  << ") " << example::precision_string<TA>() << " A precision " << example::type_string<TB>()
                  << " B precision " << example::precision_string<TC>() << " c precision "
                  << ": " << std::fixed << std::setprecision(4) << gflops << " GFLOPS, " << avg_time_n << " ms."
                  << std::endl;
    }
    // Calculate reference solution.
    std::vector<TC> host_a_ref = example::convert<TA, TC>(host_a);
    std::vector<TC> host_b_ref = example::convert<TB, TC>(host_b);
    std::vector<TC> host_c_ref = host_c;
    auto            reference_host_output =
        example::reference_gemm<gemm_type>(alpha, host_a_ref, host_b_ref, beta, host_c_ref, stream);

    // Check against reference (requires beta = 0. to avoid accumulating into C due to the repeated runs on the GPU).
    if (example::check_error_custom<TA, TB, TC>(host_output, reference_host_output)) {
        std::cout << "The results are verified to be correct." << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

#endif // CUBLASDX_EXAMPLE_SINGLE_GEMM_PERFORMANCE_HPP_
