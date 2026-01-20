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

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cuda/std/type_traits>
#include <cuda/std/tuple>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/common.hpp"
#include "../reference/cublaslt_runner.hpp"
#include "../reference/check_error.hpp"

#include "debug_printer.hpp"
#include "slicing.hpp"
#include "emulation_kernels.hpp"
#include "tensor_helpers.hpp"

// This example demonstrates the Ozaki scheme for emulating double precision GEMM
// using multiple lower precision GEMM operations. The Ozaki scheme works by:
//  1. Decomposing double precision matrices into multiple int8_t "slices"
//  2. Performing GEMM on each combination of slices
//  3. Reconstructing the final double precision result
//
// Mathematical foundation:
//   For double precision values a and b, we can represent them as:
//   a = Σ(i=0 to slices-1) a_i * 2^(shift_a - i*bits_per_slice)
//   b = Σ(j=0 to slices-1) b_j * 2^(shift_b - j*bits_per_slice)
//
//   Then a*b = ΣΣ a_i * b_j * 2^(shift_a + shift_b - (i+j)*bits_per_slice)
//
//   This allows us to compute the product using multiple int8_t GEMM operations
//   and then combine the results with appropriate scaling.

namespace {

    // An utility structure which combines configuration elements used during example execution
    template<typename TileShape, //
             typename CtaShape,  //
             int Slices,         //
             int RandomSeed = 0>
    struct emulation_params {
        using tile_shape = TileShape;
        using cta_shape  = CtaShape;

        static constexpr int slices      = Slices;     // Number of slices for Ozaki decomposition
        static constexpr int random_seed = RandomSeed; // Seed for reproducible random data

        // Performance comparison parameters
        static constexpr unsigned int kernel_repeats         = 10;
        static constexpr unsigned int kernel_warm_up_repeats = 5;
    };

    // Data format: M x N x K
    using problem_shape = cuda::std::tuple<int32_t, int32_t, int32_t>;

} // anonymous namespace

// Main cuBLASDx DGEMM emulation function using Ozaki scheme
// This function orchestrates the entire emulation process:
//   1. Preprocessing: Extract scaling factors from input matrices
//   2. Slicing: Decompose double precision matrices into int8_t slices
//   3. Matrix multiplication: Perform GEMM on slice combinations
//   4. Reconstruction: Combine results back to double precision
template<int Arch, cublasdx::sm_modifier Modifier, class Params, class GEMMShape, class GEMMArr>
auto cublasdx_dgemm_emulation(double                                alpha,
                              const example::device_vector<double>& device_a,
                              const example::device_vector<double>& device_b,
                              double                                beta,
                              example::device_vector<double>&       device_c,
                              GEMMShape                             gemm_shape,
                              GEMMArr                               gemm_arrangement,
                              cudaStream_t                          stream = 0) {
    // ================================
    // Type definitions for emulation
    // ================================

    using a_value_type = double; // Input matrix A precision
    using b_value_type = double; // Input matrix B precision
    using c_value_type = double; // Output matrix C precision

    using slice_value_type       = int8_t;  // Precision for individual slices
    using accumulator_value_type = int32_t; // Precision for accumulation

    /* The code requires signed magnitudes for slices */
    static_assert(std::is_signed<slice_value_type>());
    static_assert(std::is_signed<accumulator_value_type>());

    /* preconditions */
    static_assert(sizeof(accumulator_value_type) > sizeof(slice_value_type));

    float total_time = 0.0f;

    // ====================================
    // Global tensor creation and layout
    // ====================================

    /* Define matrices input/output matrices A[m-by-k], B[k-by-n], C[m-by-n] */
    const auto [m, n, k] = gemm_shape;

    auto d_tensor_a = cublasdx::make_gmem_tensor<cuda::std::get<0>(gemm_arrangement)>(device_a.data(), m, k);
    auto d_tensor_b = cublasdx::make_gmem_tensor<cuda::std::get<1>(gemm_arrangement)>(device_b.data(), k, n);
    auto d_tensor_c = cublasdx::make_gmem_tensor<cuda::std::get<2>(gemm_arrangement)>(device_c.data(), m, n);

    /* ============================================================== */
    /*                    OZAKI SCHEME STEP 1: SETUP                  */
    /*                     Prepare slice tensors                      */
    /* ============================================================== */

    // Extract GEMM dimensions
    auto slice_m = cuda::std::get<0>(gemm_shape);
    auto slice_n = cuda::std::get<1>(gemm_shape);
    auto slice_k = cuda::std::get<2>(gemm_shape);

    // Extract tile dimensions (must divide evenly into GEMM dimensions)
    constexpr auto tile_m = cuda::std::get<0>(typename Params::tile_shape {});
    constexpr auto tile_n = cuda::std::get<1>(typename Params::tile_shape {});
    constexpr auto tile_k = cuda::std::get<2>(typename Params::tile_shape {});

    // Verify that tile dimensions divide evenly into problem dimensions
    if (slice_m % tile_m != 0 or slice_n % tile_n != 0 or slice_k % tile_k != 0) {
        std::cerr << "GEMM shape must be divisible by tile shape" << std::endl;
        exit(-1);
    }

    // Create slice tensor A: [slices, m, k] - stores int8_t slices of matrix A
    // Each slice represents a portion of the original double precision values
    example::device_vector<slice_value_type> d_slice_a(m * k * Params::slices);

    auto const shape_slice_a  = cuda::std::make_tuple(m, k, cuda::std::integral_constant<int, Params::slices> {});
    auto const stride_slice_a = example::conditional_return<cuda::std::get<0>(gemm_arrangement) == cublasdx::col_major>(
        cuda::std::make_tuple(cuda::std::integral_constant<int, 1> {}, m, m * k),
        cuda::std::make_tuple(k, cuda::std::integral_constant<int, 1> {}, m * k));

    auto d_tensor_slice_a = example::make_gmem_tensor_from_tuples(d_slice_a.data(), shape_slice_a, stride_slice_a);

    // Create slice tensor B: [slices, k, n] - stores int8_t slices of matrix B
    example::device_vector<slice_value_type> d_slice_b(k * n * Params::slices);

    auto const shape_slice_b  = cuda::std::make_tuple(k, n, cuda::std::integral_constant<int, Params::slices> {});
    auto const stride_slice_b = example::conditional_return<cuda::std::get<1>(gemm_arrangement) == cublasdx::col_major>(
        cuda::std::make_tuple(cuda::std::integral_constant<int, 1> {}, k, k * n),
        cuda::std::make_tuple(n, cuda::std::integral_constant<int, 1> {}, k * n));

    auto d_tensor_slice_b = example::make_gmem_tensor_from_tuples(d_slice_b.data(), shape_slice_b, stride_slice_b);

    /* ============================================================== */
    /*                OZAKI SCHEME STEP 2: PREPROCESSING              */
    /*           Extract max exponent of rows(A) and cols(B)          */
    /* ============================================================== */

    // The Ozaki scheme requires finding the maximum absolute value in each
    // row of A and each column of B to determine appropriate scaling factors.
    // These scaling factors ensure that when we slice the double precision
    // values into int8_t components, we don't lose significant precision.

    // Storage for scaling factors (exponent shifts)
    example::device_vector<int32_t> d_shift_a(cuda::std::get<0>(gemm_shape)); // One shift per row of A
    example::device_vector<int32_t> d_shift_b(cuda::std::get<1>(gemm_shape)); // One shift per column of B

    // Create tensors for the shift values with proper tiling structure
    auto d_tensor_shift_a = example::make_gmem_tensor_from_tuples(
        d_shift_a.data(),
        cuda::std::make_tuple(cuda::std::integral_constant<int, tile_m> {}, slice_m / tile_m),
        cuda::std::make_tuple(cuda::std::integral_constant<int, 1> {}, tile_m));

    auto d_tensor_shift_b = example::make_gmem_tensor_from_tuples(
        d_shift_b.data(),
        cuda::std::make_tuple(cuda::std::integral_constant<int, tile_n> {}, slice_n / tile_n),
        cuda::std::make_tuple(cuda::std::integral_constant<int, 1> {}, tile_n));

    // Execute preprocessing kernels to find maximum values and compute scaling factors
    {
        constexpr int reduction_block_size = 64;
        auto          run_preprocessing    = [&](auto str) {
            // Find max absolute value in each row of A and convert to exponent shift
            max_reduce_kernel<reduction_block_size, slice_matrix::a>
                <<<slice_m, reduction_block_size, 0, str>>>(d_tensor_a, d_tensor_shift_a);
            // Find max absolute value in each column of B and convert to exponent shift
            max_reduce_kernel<reduction_block_size, slice_matrix::b>
                <<<slice_n, reduction_block_size, 0, str>>>(d_tensor_b, d_tensor_shift_b);
        };

        auto time_ms = example::measure::execution(
            run_preprocessing, Params::kernel_warm_up_repeats, Params::kernel_repeats, stream);
        total_time += time_ms / Params::kernel_repeats;
        std::cout << "----> cuBLASDx Preprocess time: " << time_ms / Params::kernel_repeats << " ms" << std::endl;

        CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    }

    /* ============================================================== */
    /*                    OZAKI SCHEME STEP 3: SLICING                */
    /*                  Slice up input A and B matrices               */
    /* ============================================================== */

    // This step decomposes each double precision value into multiple int8_t slices.
    // For a double precision value x with scaling factor s, we create slices such that:
    //   x ≈ Σ(i=0 to slices-1) slice_i * 2^(s - i*8)
    // where each slice_i is an int8_t value.

    {
        constexpr auto slice_kernel_block_size = 64;

        auto run_slicing = [&](auto str) {
            // Slice matrix A: each double precision element becomes 'slices' int8_t values
            slice_kernel<slice_kernel_block_size, Params::slices, slice_matrix::a>
                <<<(slice_m * slice_k) / slice_kernel_block_size, slice_kernel_block_size, 0, str>>>(
                    d_tensor_a, d_tensor_shift_a, d_tensor_slice_a, slice_k);
            // Slice matrix B: each double precision element becomes 'slices' int8_t values
            slice_kernel<slice_kernel_block_size, Params::slices, slice_matrix::b>
                <<<(slice_k * slice_n) / slice_kernel_block_size, slice_kernel_block_size, 0, str>>>(
                    d_tensor_b, d_tensor_shift_b, d_tensor_slice_b, slice_k);
        };

        auto time_ms =
            example::measure::execution(run_slicing, Params::kernel_warm_up_repeats, Params::kernel_repeats, stream);
        total_time += time_ms / Params::kernel_repeats;
        std::cout << "----> cuBLASDx Slice time: " << time_ms / Params::kernel_repeats << " ms" << std::endl;

        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    }

    /* ============================================================== */
    /*            OZAKI SCHEME STEP 4: MATRIX MULTIPLICATION          */
    /*                      Product of slices                         */
    /* ============================================================== */

    // This is the core of the Ozaki scheme. We need to compute the product:
    //   C = A * B = (Σ A_i * 2^shift_A_i) * (Σ B_j * 2^shift_B_j)
    //     = ΣΣ A_i * B_j * 2^(shift_A_i + shift_B_j)
    //
    // We compute this as multiple GEMM operations between slice combinations,
    // with each result scaled appropriately and accumulated into the final result.

    auto [time_matmul_kernel_ms, results] = slice_matmul_and_epilogue<Arch, Modifier, Params>(gemm_shape,
                                                                                              gemm_arrangement,
                                                                                              d_tensor_shift_a,
                                                                                              d_tensor_shift_b,
                                                                                              alpha,
                                                                                              d_tensor_slice_a,
                                                                                              d_tensor_slice_b,
                                                                                              beta,
                                                                                              d_tensor_c,
                                                                                              stream);

    // ================================
    // Performance reporting
    // ================================

    const double avg_time_cublasdx_ms = time_matmul_kernel_ms / Params::kernel_repeats;
    total_time += avg_time_cublasdx_ms;

    // Calculate TFLOPS for matrix multiplication kernel only
    double cublasdx_matmul_tflops =
        example::gemm_flops<a_value_type, b_value_type, c_value_type>(slice_m, slice_n, slice_k) /
        (avg_time_cublasdx_ms * 1e9);

    // Calculate TFLOPS for end-to-end including preprocessing and slicing
    double cublasdx_e2e_tflops =
        example::gemm_flops<a_value_type, b_value_type, c_value_type>(slice_m, slice_n, slice_k) / (total_time * 1e9);

    std::cout << "----> cuBLASDx Matmul Kernel time: " << avg_time_cublasdx_ms
              << " ms TFLOPs = " << cublasdx_matmul_tflops << std::endl;
    std::cout << "----> cuBLASDx E2E time: " << total_time << " ms TFLOPs = " << cublasdx_e2e_tflops << std::endl;

    return results;
}

// Main driver function for DGEMM emulation testing
// Tests the Ozaki scheme emulation against native cuBLAS DGEMM for correctness and performance
template<int Arch, cublasdx::sm_modifier Modifier, typename Params>
int dgemm_emulation(const std::vector<problem_shape>& problems, const bool& debug) {

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream));

    /* ===================================================================== */
    /*            Performance evaluation of multiple problem sizes           */
    /* ===================================================================== */

    for (const auto& shape : problems) {
        const int32_t m          = cuda::std::get<0>(shape);
        const int32_t n          = cuda::std::get<1>(shape);
        const int32_t k          = cuda::std::get<2>(shape);
        auto          gemm_shape = cuda::std::make_tuple(m, n, k);

        // Validation of problem size - for educational purposes, only square matrices supported
        if (m != n or m != k) {
            std::cerr << "For education purposes prepared example supports only square matrices, provided problem has "
                      << m << "x" << n << "x" << k << " size" << std::endl;
            return -1;
        }

        // ===================================
        // Global GEMM arrangement configuration
        // ===================================

        // Global GEMM Arrangement:
        // - cubladsx::row_major, row major data arrangement
        // - cubladsx::col_major, col major data arrangement
        // Note: these values need to be constexpr
        constexpr auto global_arrangement_a = cublasdx::row_major;
        constexpr auto global_arrangement_b = cublasdx::col_major;
        constexpr auto global_arrangement_c = cublasdx::col_major;

        // Use tuples to avoid passing 20 arguments to a function
        constexpr auto global_arrangement = cuda::std::make_tuple(
            // These must be passed as integral constants to properly dispatch static striding
            cuda::std::integral_constant<cublasdx::arrangement, global_arrangement_a> {},
            cuda::std::integral_constant<cublasdx::arrangement, global_arrangement_b> {},
            cuda::std::integral_constant<cublasdx::arrangement, global_arrangement_c> {});

        // ===================================
        // Data type definitions
        // ===================================

        using a_value_type = double;
        using b_value_type = double;
        using c_value_type = double;

        using alpha_value_type = double;
        using beta_value_type  = double;

        /* ============================================================== */
        /*                     Input FP64 (host) tensors                  */
        /* ============================================================== */

        // Scalar multipliers for GEMM: C = alpha * A * B + beta * C
        alpha_value_type alpha = 1.1;
        beta_value_type  beta  = 0.9;

        // Generate random test data, normal distribution with mean = 0.0 and sd = 1.0
        static const float range_lower_bound = 1.0f / 3.14f;
        static const float range_upper_bound = 52.0f / 3.14f;

        auto host_a = example::get_random_uniform_data<a_value_type>(
            m * k, range_lower_bound, range_upper_bound, Params::random_seed);
        auto host_b = example::get_random_uniform_data<b_value_type>(
            n * k, range_lower_bound, range_upper_bound, Params::random_seed);
        auto host_c = example::get_random_uniform_data<c_value_type>(
            m * n, range_lower_bound, range_upper_bound, Params::random_seed);

        // Copy to device
        example::device_vector<a_value_type> d_a = host_a;
        example::device_vector<b_value_type> d_b = host_a;

        /* ============================================================== */
        /*                       Compute Reference Result                 */
        /* ============================================================== */

        // Compute reference result using native cuBLAS double precision GEMM
        const auto cublas_results = [&]() {
            example::device_vector<c_value_type> d_c = host_c;
            auto [time_cublas_ms, host_blas_results] =
                example::cublaslt_runner<a_value_type, b_value_type, c_value_type>(gemm_shape, global_arrangement)
                    .execute_with_time_and_results(alpha,
                                                   d_a.data(),
                                                   d_b.data(),
                                                   beta,
                                                   d_c.data(),
                                                   Params::kernel_warm_up_repeats,
                                                   Params::kernel_repeats,
                                                   stream);

            const double avg_time_cublas_ms = time_cublas_ms / Params::kernel_repeats;
            double       cublas_tflops =
                example::gemm_flops<a_value_type, b_value_type, c_value_type>(m, n, k) / (avg_time_cublas_ms * 1e9);

            std::cout << "----> cuBLAS DGEMM execution time: " << avg_time_cublas_ms << " ms TFLOPs = " << cublas_tflops
                      << std::endl;

            return host_blas_results;
        }();

        /* ============================================================== */
        /*                     Compute Emulation Result                   */
        /* ============================================================== */

        // Compute emulated result using Ozaki scheme with cuBLASDx
        example::device_vector<c_value_type> cublasdx_c       = host_c;
        auto                                 cublasdx_results = cublasdx_dgemm_emulation<Arch, Modifier, Params>(
            alpha, d_a, d_b, beta, cublasdx_c, gemm_shape, global_arrangement, stream);

        // ===================================
        // Error analysis and reporting
        // ===================================

        double tot_rel_cublasdx_emulation_error =
            example::calculate_error(cublasdx_results, cublas_results, debug, debug);

        std::cout << "{" << cuda::std::get<0>(gemm_shape) << "," << cuda::std::get<1>(gemm_shape) << ","
                  << cuda::std::get<2>(gemm_shape) << "} "
                  << "cuBLASDx emulation to cuBLAS native DGEMM total relative error = "
                  << tot_rel_cublasdx_emulation_error << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    return 0;
}

template<typename Params>
struct dgemm_emulation_functor {
    template<int Arch, cublasdx::sm_modifier Modifier, class... Args>
    int operator()(std::integral_constant<int, Arch>,
                   std::integral_constant<cublasdx::sm_modifier, Modifier>,
                   const Args&... args) {
        return dgemm_emulation<Arch, Modifier, Params>(args...);
    }
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {

    // ===================================
    // Ozaki scheme configuration
    // ===================================

    // The number of slices used in emulation algorithm
    // More slices = higher precision but more computation
    constexpr unsigned slices = 7;

    // ===================================
    // cuBLASDx tile configuration
    // ===================================

    // For good performance results on Blackwell (RTX Pro) architecture the following tile combinations
    // should provide performant kernel configuration
    //
    // 128x64x128 -- 128 threads
    // 128x128x64 -- 128 threads
    // 128x256x64 -- 256 threads
    // 256x128x64 -- 256 threads
    // 128x128x128 -- 256 threads, good for bigger sizes

    // For good performance results on Blackwell (B200) architecture it is adviced to pick smaller tiles
    // for smaller problems and with problem's grow pick larger tiles. All this to ensure high GPU
    // occupancy where large number of SMs performs computations in parallel.
    //
    // Example configurations:
    // - problem 1024x1024x1024:    128x64x256  -- 128 threads
    // - problem 2048x2048x2048:    128x128x128 -- 128 threads
    // - problem 4096x4096x4096:    128x128x128 -- 128 threads
    // - problem 8192x8192x8192:    128x128x128 -- 128 threads
    // - problem 16384x16384x16384: 128x256x128 -- 128 threads


    // The shape of data tile processed by a single CTA block
    constexpr int tile_shape_m = 128;
    constexpr int tile_shape_n = 128;
    constexpr int tile_shape_k = 128;

    // The shape of CTA block (number of threads)
    constexpr int cta_shape_x = 128;
    constexpr int cta_shape_y = 1;
    constexpr int cta_shape_z = 1;

    using tile_shape = cuda::std::tuple<cuda::std::integral_constant<int, tile_shape_m>,
                                        cuda::std::integral_constant<int, tile_shape_n>,
                                        cuda::std::integral_constant<int, tile_shape_k>>;

    using cta_shape = cuda::std::tuple<cuda::std::integral_constant<int, cta_shape_x>,
                                       cuda::std::integral_constant<int, cta_shape_y>,
                                       cuda::std::integral_constant<int, cta_shape_z>>;

    using params = emulation_params<tile_shape, cta_shape, slices>;

    bool debug = false;

    if (debug) {
        print_device_properties();
    }

    // ===================================
    // Problem sizes for testing
    // ===================================

    // Format of problem shape: M x N x K
    std::vector<problem_shape> problems = {{2048, 2048, 2048}, {4096, 4096, 4096}};

    return example::sm_runner(dgemm_emulation_functor<params> {}, problems, debug);
}

