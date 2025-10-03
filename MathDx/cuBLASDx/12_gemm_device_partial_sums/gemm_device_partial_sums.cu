/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <array>
#include <iostream>
#include <system_error>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

template<class BLAS,
         int AccumulationIterations,
         class GEMMShape,
         class GEMMArr,
         class GEMMLD,
         class Alpha,
         class AValueType,
         class BValueType,
         class Beta,
         class CValueType>
__launch_bounds__(BLAS::max_threads_per_block, 1)
__global__ void gemm_kernel(GEMMShape  const  gemm_shape,
                            GEMMArr    const  gemm_arr,
                            GEMMLD     const  gemm_ld,
                            Alpha      const  alpha,
                            AValueType const* a,
                            BValueType const* b,
                            Beta       const  beta,
                            CValueType      * c) {

    extern __shared__ __align__(16) char smem[];

    using alignment = cublasdx::alignment_of<BLAS>;

    // ================================
    // 1. PREPARE GLOBAL MEMORY TENSORS

    // Handle row-major symmetrically to col-major
    const auto block_coord = example::get_block_coord(gemm_arr);

    // Create tensors for global A / B / C corresponding to set MNK, arrangement and LDs
    auto [global_a, global_b, global_c] = example::make_device_global_tensors(gemm_shape, gemm_arr, gemm_ld, a, b, c);

    // Get a row of tiles from A, containing K / tile_k stages
    const auto tile_slice_a_gmem = example::get_block_tile_slice_a<BLAS>(global_a, block_coord);
    // Get a column of tiles from B containing K / tile_k stages
    const auto tile_slice_b_gmem = example::get_block_tile_slice_b<BLAS>(global_b, block_coord);
    // Get a single tile from C
    auto tile_c_gmem = example::get_block_tile_c<BLAS>(global_c, block_coord);

    // ================================
    // 2. PREPARE SHARED MEMORY TENSORS

    // Slice shared memory into tensors for proper alignment in 2-stage pipelining
    auto [s_a, s_b, s_a_n, s_b_n] =
        cublasdx::shared_memory::slice<AValueType, BValueType, AValueType, BValueType>(
            smem,
            cublasdx::alignment_of_v_a<BLAS>, BLAS::suggest_layout_smem_a(),
            cublasdx::alignment_of_v_b<BLAS>, BLAS::suggest_layout_smem_b(),
            cublasdx::alignment_of_v_a<BLAS>, BLAS::suggest_layout_smem_a(),
            cublasdx::alignment_of_v_b<BLAS>, BLAS::suggest_layout_smem_b()
        );

    // ==================================
    // 3. PREPARE 2-STAGE MEMORY PIPELINE

    // Since both slices have an iteration dimension equal to number of necessary
    // GEMM stages, we can just use it for iteration
    const auto k_stages = cute::get<2>(cute::shape(tile_slice_a_gmem.layout()));

    // Schedule first stage into memory pipeline queue
    // cute::Int<X> is a static integer similar to std::integral_constant
    constexpr auto static_first_stage_index = cute::Int<0>{};
    cublasdx::copy<BLAS, alignment::a>(example::get_tile_from_slice(tile_slice_a_gmem, static_first_stage_index), s_a);
    cublasdx::copy<BLAS, alignment::b>(example::get_tile_from_slice(tile_slice_b_gmem, static_first_stage_index), s_b);

    // ==============================================
    // 4. EXECUTE GEMM WITH ACCUMULATION IN REGISTERS

    auto partitioner = BLAS().suggest_partitioner();
    auto c_frag = partitioner.make_accumulator_fragment();
    cublasdx::clear(c_frag);

    auto c_frag_accumulator = cublasdx::make_fragment_like<CValueType>(c_frag);
    cublasdx::clear(c_frag_accumulator);

    #pragma unroll 1
    for(int stage = 0; stage < k_stages; stage += AccumulationIterations) {
        // Inner loop for accumulating partial sums (fast accumulator is cleared every AccumulationIterations)
        #pragma unroll 1
        for(int iter = 0; iter < AccumulationIterations; ++iter) {
            const auto next_stage = stage + iter + 1;

            // Wait for previous stage
            cublasdx::copy_wait();

            // Load next stage of A and B
            if(next_stage < k_stages) {
                cublasdx::copy<BLAS, alignment::a>(example::get_tile_from_slice(tile_slice_a_gmem, next_stage), s_a_n);
                cublasdx::copy<BLAS, alignment::b>(example::get_tile_from_slice(tile_slice_b_gmem, next_stage), s_b_n);
            }

            // Accumulate results from this stage
            BLAS().execute(s_a, s_b, c_frag);

            // Swap for next iteration
            example::swap(s_a_n, s_a);
            example::swap(s_b_n, s_b);
        }

        // Accumulate partial sums in higher precision
        cublasdx::transform(c_frag, c_frag_accumulator, c_frag_accumulator, [](auto a, auto b) { return a + b; } );

        // Clear fast accumulator for next iteration
        cublasdx::clear(c_frag);
    }

    // ===========
    // 5. EPILOGUE
    auto d_frag = cublasdx::make_fragment_like<CValueType>(c_frag);
    cublasdx::copy_fragment<alignment::c>(tile_c_gmem, d_frag, partitioner);
    cublasdx::axpby(static_cast<CValueType>(alpha), c_frag_accumulator, static_cast<CValueType>(beta), d_frag);
    cublasdx::copy_fragment<alignment::c>(d_frag, tile_c_gmem, partitioner);
}

template<class BLAS,
         int AccumulationIterations,
         class GEMMShape,
         class GEMMArr,
         class GEMMLD,
         class Alpha,
         class AValueType,
         class BValueType,
         class Beta,
         class CValueType>
auto measure_cublasdx(GEMMShape         gemm_shape,
                      GEMMArr           gemm_arr,
                      GEMMLD            gemm_ld,
                      const Alpha       alpha,
                      const AValueType* a,
                      const BValueType* b,
                      const Beta        beta,
                      CValueType*       c,
                      cudaStream_t      stream) {
    // Grid size configuration
    const auto m = cute::get<0>(gemm_shape);
    const auto n = cute::get<1>(gemm_shape);

    constexpr auto tile_m = cublasdx::size_of<BLAS>::m;
    constexpr auto tile_n = cublasdx::size_of<BLAS>::n;
    constexpr auto tile_k = cublasdx::size_of<BLAS>::k;

    std::vector<CValueType> results(m * n);

    std::cout << "tile_m, tile_n, tile_k: " << tile_m << ", " << tile_n << ", " << tile_k << std::endl;
    std::cout << "Alignment: " << cublasdx::alignment_of<BLAS>::a << ", " << cublasdx::alignment_of<BLAS>::b << ", " << cublasdx::alignment_of<BLAS>::c << std::endl;
    std::cout << "tile_lda, tile_ldb, tile_ldc: " << BLAS::lda << ", " << BLAS::ldb << ", " << BLAS::ldc << std::endl;
    std::cout << "block_size: " << BLAS::max_threads_per_block << std::endl;

    constexpr bool reverse_block_coord = (cute::get<0>(gemm_arr) == cublasdx::row_major) and
                                         (cute::get<1>(gemm_arr) == cublasdx::row_major);

    dim3 grid_dim = cute::conditional_return<reverse_block_coord>(dim3{n / tile_n, m / tile_m, 1}, dim3{(m / tile_m), (n / tile_n), 1});

    // Increase max dynamic shared memory for the kernel if needed.
    auto shared_memory_size =
        cublasdx::make_shared_storage_calculator()
        .add(cublasdx::alignment_of_v_a<BLAS>, sizeof(AValueType), BLAS::suggest_layout_smem_a())
        .add(cublasdx::alignment_of_v_b<BLAS>, sizeof(BValueType), BLAS::suggest_layout_smem_b())
        .add(cublasdx::alignment_of_v_a<BLAS>, sizeof(AValueType), BLAS::suggest_layout_smem_a())
        .add(cublasdx::alignment_of_v_b<BLAS>, sizeof(BValueType), BLAS::suggest_layout_smem_b())
        .get();

    auto kernel = gemm_kernel<BLAS, AccumulationIterations, GEMMShape, GEMMArr, GEMMLD, Alpha, AValueType, BValueType, Beta, CValueType>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    auto run_cublasdx_gemm = [&](cudaStream_t str) {
        kernel<<<grid_dim, BLAS::block_dim, shared_memory_size, str>>>(gemm_shape, gemm_arr, gemm_ld, alpha, a, b, beta, c);
    };

    // Perform computations with cuBLASDx BLAS operator.
    run_cublasdx_gemm(stream);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(), c, results.size() * sizeof(CValueType), cudaMemcpyDeviceToHost));

    return results;
}

// This is an example of using cuBLASDx for computing partial sums of a GEMM and summing them in a
// higher precision storage for enhanced accuracy of the final result.
//
//              C = alpha * A * B + beta * C
//
// A, B, and C are matrices. Mixed precisions are supported.
// Note that alpha and beta are expected to have the same precision and type (real or complex value)
// as the matrix C elements.
//
// The partial accumulation is performed every N iterations of the K dimension of the GEMM, which can
// be configured via the partial_sum_iterations template parameter.
template<unsigned int Arch>
int gemm_device_partial_sums() {

    // ===================================
    // Configurable Global GEMM properties
    // ===================================

    // Global GEMM Size --> MNK, where:
    // - A matrix is M x K
    // - B matrix is K x N
    // - C matrix is M x N

    // A matrix which is small but deep, a lot of roundings will be needed
    const auto m = 768; // Global M GEMM Size
    const auto n = 768; // Global N GEMM Size
    const auto k = 8192; // Global K GEMM Size
    auto global_shape = cute::make_tuple(m, n, k);

    // Compute precision (use Tensor Cores of this precision)
    // and cuBLAS input precision
    using a_compute_precision = __half;
    using b_compute_precision = __half;
    using c_compute_precision = float;

    // Higher precision to perform partial sums in
    // NOTE: either this type must be implicitly convertible to 
    // compute type, or converter should be provided in appropriate places.
    // please refer to simple_gemm_fp32_decoupled.cu example for more details.
    using c_accumulation_precision = double;
    // What should be the interval for accumulating in higher precision
    constexpr int partial_sum_iterations = 64;

    if(k % partial_sum_iterations != 0) {
        std::cerr << "K must be divisible by partial_sum_iterations" << std::endl;
        return 1;
    }

    // Global GEMM Arrangement:
    // - cubladsx::row_major, row major data arrangement
    // - cubladsx::col_major, col major data arrangement
    // Note: these values need to be constexpr
    constexpr auto global_arrangement_a = cublasdx::row_major;
    constexpr auto global_arrangement_b = cublasdx::col_major;
    constexpr auto global_arrangement_c = cublasdx::row_major;

    // Leading Dimensions to be used for global data
    // Note: for matrix of size X x Y, the LD must be:
    // - greater or equal than X if matrix is col-major
    // - greater or equal than Y if matrix is row-major
    // Note: these values can be dynamic
    const auto global_lda = (global_arrangement_a == cublasdx::col_major) ? m : k;
    const auto global_ldb = (global_arrangement_b == cublasdx::col_major) ? k : n;
    const auto global_ldc = (global_arrangement_c == cublasdx::col_major) ? m : n;

    // Number type, either real or complex
    constexpr auto type = cublasdx::type::real;

    // Create data type, based on:
    // - precision
    // - type (real / complex)
    // this will be either precision or cublasdx::complex<precision>
    using a_compute_value_type = example::get_value_type_t<a_compute_precision, type>;
    using b_compute_value_type = example::get_value_type_t<b_compute_precision, type>;
    using c_compute_value_type = example::get_value_type_t<c_compute_precision, type>;
    using c_accumulation_value_type = example::get_value_type_t<c_accumulation_precision, type>;

    // Scalar multipliers
    // C = alpha * A * B + beta * C
    c_compute_value_type alpha = example::make_value<c_compute_value_type>(1.1);
    c_compute_value_type beta  = example::make_value<c_compute_value_type>(1.2);

    // ======================================
    // Configurable cuBLASDx tile properties
    // ======================================

    // tile size, this describes smaller GEMM,
    // which will be computed by each threadblock
    // using cuBLASDx
    constexpr unsigned int tile_m = 128;
    constexpr unsigned int tile_n = 128;
    constexpr unsigned int tile_k = 32;

    // Number of threads to compute the tile described above
    constexpr unsigned int tile_threads = 128;

    // Arrangement of data in a per-threadblock tile of data
    constexpr auto tile_arr_a = global_arrangement_a;
    constexpr auto tile_arr_b = global_arrangement_b;
    constexpr auto tile_arr_c = global_arrangement_c;

    // Input used to be converted to the final compute precision,
    // this can be used to simulate either in-flight quantization
    // or flexible upcasting of data to save on bandwidth
    using a_io_value_type = a_compute_value_type;
    using b_io_value_type = b_compute_value_type;

    // This is the accumulation precision
    using c_io_value_type = c_accumulation_value_type;

    // Maximal alignment to be used for shared memory data.
    // Effectively this limits maximal vectorization level
    // for loads and stores.
    constexpr unsigned int maximal_alignment = 64;
    constexpr unsigned int cublasdx_alignment = maximal_alignment;

    // ================================
    // Verify configuration correctness
    // ================================

    const bool divisible = (m % tile_m == 0 and n % tile_n == 0 and k % tile_k == 0);
    if(not divisible) {
        std::cerr << "M, N, K dimensions must be divisible by tile_m, tile_n, tile_k" << std::endl;
        return 1;
    }

    // ================================
    // Prepare inputs
    // ================================

    // Use tuples to avoid passing 20 arguments to a function
    constexpr auto global_arrangement = cute::make_tuple(
        // These must be passed as integral constants to properly dispatch static striding
        std::integral_constant<cublasdx::arrangement, global_arrangement_a>{},
        std::integral_constant<cublasdx::arrangement, global_arrangement_b>{},
        std::integral_constant<cublasdx::arrangement, global_arrangement_c>{}
    );

    const auto global_ld = cute::make_tuple(
        global_lda, global_ldb, global_ldc
    );

    // Test implementation
    a_io_value_type *a_cublasdx = nullptr;
    b_io_value_type *b_cublasdx = nullptr;
    c_io_value_type *c_cublasdx = nullptr;

    a_compute_value_type *a_cublas = nullptr;
    b_compute_value_type *b_cublas = nullptr;
    c_compute_value_type *c_cublas = nullptr;

    // Use nullptr tensors to make it easier to calculate memory requirements
    auto [global_a, global_b, global_c] =
        example::make_device_global_tensors(global_shape, global_arrangement, global_ld, a_cublas, b_cublas, c_cublas);

    CUDA_CHECK_AND_EXIT(cudaMalloc(&a_cublasdx, cublasdx::cosize(global_a.layout()) * sizeof(a_io_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&a_cublas,   cublasdx::cosize(global_a.layout()) * sizeof(a_compute_value_type)));

    CUDA_CHECK_AND_EXIT(cudaMalloc(&b_cublasdx, cublasdx::cosize(global_b.layout()) * sizeof(b_io_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&b_cublas,   cublasdx::cosize(global_b.layout()) * sizeof(b_compute_value_type)));

    CUDA_CHECK_AND_EXIT(cudaMalloc(&c_cublasdx, cublasdx::cosize(global_c.layout()) * sizeof(c_io_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(&c_cublas,   cublasdx::cosize(global_c.layout()) * sizeof(c_compute_value_type)));

    // Fill the A, B, C matrices with random values.
    {
        auto host_a_io = example::get_random_data<a_io_value_type>(-0.1f, 0.1f, m * k);
        auto host_b_io = example::get_random_data<b_io_value_type>(-0.1f, 0.1f, k * n);
        auto host_c_io = example::get_random_data<c_io_value_type>(-0.1f, 0.1f, m * n);

        static_assert(std::is_convertible_v<a_io_value_type, a_compute_value_type>);
        auto host_a_compute = std::vector<a_compute_value_type>(host_a_io.begin(), host_a_io.end());

        static_assert(std::is_convertible_v<b_io_value_type, b_compute_value_type>);
        auto host_b_compute = std::vector<b_compute_value_type>(host_b_io.begin(), host_b_io.end());

        static_assert(std::is_convertible_v<c_io_value_type, c_compute_value_type>);
        auto host_c_compute = std::vector<c_compute_value_type>(host_c_io.begin(), host_c_io.end());

        CUDA_CHECK_AND_EXIT(cudaMemcpy(a_cublasdx, host_a_io.data(), m * k * sizeof(a_io_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(b_cublasdx, host_b_io.data(), k * n * sizeof(b_io_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(c_cublasdx, host_c_io.data(), m * n * sizeof(c_io_value_type), cudaMemcpyHostToDevice));

        CUDA_CHECK_AND_EXIT(cudaMemcpy(a_cublas, host_a_compute.data(), m * k * sizeof(a_compute_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(b_cublas, host_b_compute.data(), k * n * sizeof(b_compute_value_type), cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(c_cublas, host_c_compute.data(), m * n * sizeof(c_compute_value_type), cudaMemcpyHostToDevice));
        // destroy host vectors
    }

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))

    // cuBLASDx type creation
    using BLAS = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                          cublasdx::Precision<a_compute_precision, b_compute_precision, c_compute_precision>() +
                          cublasdx::Type<type>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<tile_arr_a, tile_arr_b, tile_arr_c>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<tile_threads>() +
                          cublasdx::Alignment<cublasdx_alignment, cublasdx_alignment, cublasdx_alignment>() +
                          cublasdx::SM<Arch>());

    // =============================
    // Execute cuBLASDx and cuBLASLt
    // =============================

    auto host_dx_results =
        measure_cublasdx<BLAS, partial_sum_iterations>(global_shape, global_arrangement, global_ld, alpha, a_cublasdx, b_cublasdx, beta, c_cublasdx, stream);

    // Measure cuBLAS performance.
    auto host_blas_results =
        example::cublaslt_runner<a_compute_value_type, b_compute_value_type, c_compute_value_type>(global_shape, global_arrangement, global_ld)
            .execute_with_results(alpha, a_cublas, b_cublas, beta, c_cublas, stream);

    // Write performance data.
    using cublasdx::size_of;
    std::cout << "m, n, k: " << m << ", " << n << ", " << k
              << std::endl;
    std::cout << "Compute Type A: " << example::type_string<a_compute_value_type>() << std::endl;
    std::cout << "Compute Type B: " << example::type_string<b_compute_value_type>() << std::endl;
    std::cout << "Compute Type C: " << example::type_string<c_compute_value_type>() << std::endl;
    std::cout << "Accumulation Type: " << example::type_string<c_accumulation_value_type>() << std::endl;
    std::cout << "Accumulation Every N Iterations: " << partial_sum_iterations << std::endl;
    
    std::cout << "Dx Input Precision A: " << example::precision_string<a_io_value_type>() << std::endl;
    std::cout << "Dx Input Precision B: " << example::precision_string<b_io_value_type>() << std::endl;
    std::cout << "Dx Input Precision C: " << example::precision_string<c_io_value_type>() << std::endl;

    std::cout << "cuBLAS Input Precision A: " << example::precision_string<a_compute_value_type>() << std::endl;
    std::cout << "cuBLAS Input Precision B: " << example::precision_string<b_compute_value_type>() << std::endl;
    std::cout << "cuBLAS Input Precision C: " << example::precision_string<c_compute_value_type>() << std::endl;

    auto error = example::calculate_error(host_dx_results, host_blas_results);
    std::cout << "Error = " << error << "\n";
    
    if(example::is_error_acceptable<a_compute_value_type, b_compute_value_type, c_compute_value_type>(error)) {
        std::cout << "Success!\n";
    } else {
        std::cout << "Failure!\n";
    }

    // Free resources.
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(a_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(b_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(c_cublasdx));
    CUDA_CHECK_AND_EXIT(cudaFree(a_cublas));
    CUDA_CHECK_AND_EXIT(cudaFree(b_cublas));
    CUDA_CHECK_AND_EXIT(cudaFree(c_cublas));

    return 0;
}

struct gemm_device_partial_sums_functor {

    template<int Arch>
    int operator()(std::integral_constant<int, Arch>) {
        return gemm_device_partial_sums<Arch>();
    }
};

int main() {
    return example::sm_runner(gemm_device_partial_sums_functor{});
}