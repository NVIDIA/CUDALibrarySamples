/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUBLASDX_EXAMPLE_REFERENCE_TRSM_HPP
#define CUBLASDX_EXAMPLE_REFERENCE_TRSM_HPP

#include <cublas_v2.h>
#include <cuComplex.h>
#include <cublasdx.hpp>

#include <type_traits>
#include <vector>
#include <tuple>

#include "../common/common.hpp"

namespace example {

    namespace detail {

        // Transpose a batch of matrices stored contiguously.
        // Input layout: dim_fast-major (e.g. col-major row→fast).
        // After transpose: dim_slow-major.
        template<class T>
        void trsm_transpose_batch(std::vector<T>& data, unsigned dim_fast, unsigned dim_slow, unsigned batches) {
            std::vector<T> tmp(data);
            for (unsigned b = 0; b < batches; ++b) {
                for (unsigned j = 0; j < dim_slow; ++j) {
                    for (unsigned k = 0; k < dim_fast; ++k) {
                        data[b * dim_fast * dim_slow + j * dim_fast + k] =
                            tmp[b * dim_slow * dim_fast + k * dim_slow + j];
                    }
                }
            }
        }

        template<class T>
        T trsm_alpha_one() {
            return T(1);
        }
        template<>
        inline cublasdx::complex<float> trsm_alpha_one<cublasdx::complex<float>>() {
            return {1.f, 0.f};
        }
        template<>
        inline cublasdx::complex<double> trsm_alpha_one<cublasdx::complex<double>>() {
            return {1.0, 0.0};
        }

    } // namespace detail

    // -------------------------------------------------------------------------
    // reference_trsm (raw parameter version)
    //
    // Solves op(A) * X = B  (left) or  X * op(A) = B  (right) with alpha = 1.
    // Neither h_A nor h_B are modified.
    //
    // Returns cute::make_tuple(solution, total_ms) where:
    //   solution  - std::vector<T> containing X (same shape as h_B)
    //   total_ms  - total cuBLAS time over `runs` launches (0.0f if runs == 0);
    //               divide by `runs` to get the per-call average.
    //
    // Handles col-major and row-major layouts transparently.
    // T : scalar type - float, double, cublasdx::complex<float/double>
    // -------------------------------------------------------------------------
    template<class T>
    auto reference_trsm(const std::vector<T>& h_A,
                        const std::vector<T>& h_B,
                        unsigned              m,
                        unsigned              n,
                        unsigned              batches,
                        bool                  is_left,
                        bool                  is_lower,
                        bool                  is_diag_unit,
                        bool                  is_col_major_a,
                        bool                  is_col_major_b,
                        cublasOperation_t     op           = CUBLAS_OP_N,
                        unsigned              warm_up_runs = 0,
                        unsigned              runs         = 0,
                        cudaStream_t          stream       = nullptr) {
        const unsigned a_m  = is_left ? m : n;
        const unsigned a_sz = a_m * a_m;
        const unsigned b_sz = m * n;

        // For row-major inputs, work on transposed internal copies so the cuBLAS
        // calls always see col-major data.  h_A and h_B are never mutated.
        std::vector<T> a_work;
        const T*       a_data = h_A.data();
        if (!is_col_major_a) {
            a_work = h_A;
            detail::trsm_transpose_batch<T>(a_work, a_m, a_m, batches);
            a_data = a_work.data();
        }

        std::vector<T> b_work(h_B);
        if (!is_col_major_b) {
            detail::trsm_transpose_batch<T>(b_work, m, n, batches);
        }

        // Save the (possibly transposed) RHS so d_B can be restored before timing.
        const std::vector<T> b_orig(runs > 0 ? b_work : std::vector<T> {});

        const cublasSideMode_t  side  = is_left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
        const cublasFillMode_t  uplo  = is_lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        const cublasOperation_t trans = op;
        const cublasDiagType_t  diag  = is_diag_unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
        const int               lda   = static_cast<int>(a_m);
        const int               ldb   = static_cast<int>(m);
        const T                 alpha = detail::trsm_alpha_one<T>();

        // Allocate all batches on device at once.
        T* d_A = nullptr;
        T* d_B = nullptr;
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * a_sz * batches));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(T) * b_sz * batches));

        // Device pointer arrays for the batched cuBLAS API.
        std::vector<T*> h_A_ptrs(batches), h_B_ptrs(batches);
        for (unsigned i = 0; i < batches; ++i) {
            h_A_ptrs[i] = d_A + i * a_sz;
            h_B_ptrs[i] = d_B + i * b_sz;
        }
        T** d_A_ptrs = nullptr;
        T** d_B_ptrs = nullptr;
        CUDA_CHECK_AND_EXIT(cudaMalloc(&d_A_ptrs, sizeof(T*) * batches));
        CUDA_CHECK_AND_EXIT(cudaMalloc(&d_B_ptrs, sizeof(T*) * batches));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(d_A_ptrs, h_A_ptrs.data(), sizeof(T*) * batches, cudaMemcpyHostToDevice));
        CUDA_CHECK_AND_EXIT(cudaMemcpy(d_B_ptrs, h_B_ptrs.data(), sizeof(T*) * batches, cudaMemcpyHostToDevice));

        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK_AND_EXIT(cublasCreate(&handle));

        bool own_stream = (stream == nullptr);
        if (own_stream) {
            CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
        CUBLAS_CHECK_AND_EXIT(cublasSetStream(handle, stream));

        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, a_data, sizeof(T) * a_sz * batches, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpyAsync(d_B, b_work.data(), sizeof(T) * b_sz * batches, cudaMemcpyHostToDevice, stream));

        constexpr bool is_float = std::is_same_v<T, float> || std::is_same_v<T, cublasdx::complex<float>>;
        constexpr bool is_complex =
            std::is_same_v<T, cublasdx::complex<float>> || std::is_same_v<T, cublasdx::complex<double>>;

        auto call_trsm = [&]() {
            if constexpr (is_float && !is_complex) {
                const float a = *reinterpret_cast<const float*>(&alpha);
                CUBLAS_CHECK_AND_EXIT(cublasStrsmBatched(handle,
                                                         side,
                                                         uplo,
                                                         trans,
                                                         diag,
                                                         static_cast<int>(m),
                                                         static_cast<int>(n),
                                                         &a,
                                                         (const float* const*)d_A_ptrs,
                                                         lda,
                                                         (float* const*)d_B_ptrs,
                                                         ldb,
                                                         static_cast<int>(batches)));
            } else if constexpr (is_float && is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasCtrsmBatched(handle,
                                                         side,
                                                         uplo,
                                                         trans,
                                                         diag,
                                                         static_cast<int>(m),
                                                         static_cast<int>(n),
                                                         reinterpret_cast<const cuComplex*>(&alpha),
                                                         (const cuComplex* const*)d_A_ptrs,
                                                         lda,
                                                         (cuComplex* const*)d_B_ptrs,
                                                         ldb,
                                                         static_cast<int>(batches)));
            } else if constexpr (!is_float && !is_complex) {
                const double a = *reinterpret_cast<const double*>(&alpha);
                CUBLAS_CHECK_AND_EXIT(cublasDtrsmBatched(handle,
                                                         side,
                                                         uplo,
                                                         trans,
                                                         diag,
                                                         static_cast<int>(m),
                                                         static_cast<int>(n),
                                                         &a,
                                                         (const double* const*)d_A_ptrs,
                                                         lda,
                                                         (double* const*)d_B_ptrs,
                                                         ldb,
                                                         static_cast<int>(batches)));
            } else {
                CUBLAS_CHECK_AND_EXIT(cublasZtrsmBatched(handle,
                                                         side,
                                                         uplo,
                                                         trans,
                                                         diag,
                                                         static_cast<int>(m),
                                                         static_cast<int>(n),
                                                         reinterpret_cast<const cuDoubleComplex*>(&alpha),
                                                         (const cuDoubleComplex* const*)d_A_ptrs,
                                                         lda,
                                                         (cuDoubleComplex* const*)d_B_ptrs,
                                                         ldb,
                                                         static_cast<int>(batches)));
            }
        };

        // Correctness run: one call, result copied back into b_work.
        call_trsm();
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
        CUDA_CHECK_AND_EXIT(
            cudaMemcpyAsync(b_work.data(), d_B, sizeof(T) * b_sz * batches, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

        // Performance timing (optional).
        float total_ms = 0.0f;
        if (runs > 0) {
            // Restore d_B to the original RHS before timing.
            CUDA_CHECK_AND_EXIT(
                cudaMemcpyAsync(d_B, b_orig.data(), sizeof(T) * b_sz * batches, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            total_ms = example::measure::execution(
                [&](cudaStream_t s) {
                    CUBLAS_CHECK_AND_EXIT(cublasSetStream(handle, s));
                    call_trsm();
                },
                warm_up_runs,
                runs,
                stream);
        }

        CUBLAS_CHECK_AND_EXIT(cublasDestroy(handle));
        CUDA_CHECK_AND_EXIT(cudaFree(d_A));
        CUDA_CHECK_AND_EXIT(cudaFree(d_B));
        CUDA_CHECK_AND_EXIT(cudaFree(d_A_ptrs));
        CUDA_CHECK_AND_EXIT(cudaFree(d_B_ptrs));
        if (own_stream) {
            CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
        }

        // Undo the row-major transpose on the solution before returning.
        if (!is_col_major_b) {
            detail::trsm_transpose_batch<T>(b_work, n, m, batches);
        }

        return std::make_tuple(b_work, total_ms);
    }

    // -------------------------------------------------------------------------
    // reference_trsm (BLAS descriptor version)
    //
    // Convenience overload: all TRSM parameters are extracted from the BLAS
    // descriptor at compile time.
    // -------------------------------------------------------------------------
    template<class BLAS>
    auto reference_trsm(const std::vector<typename BLAS::a_value_type>& h_A,
                        const std::vector<typename BLAS::b_value_type>& h_B,
                        unsigned                                        batches,
                        unsigned                                        warm_up_runs = 0,
                        unsigned                                        runs         = 0,
                        cudaStream_t                                    stream       = nullptr) {
        using T = typename BLAS::a_value_type;
        static_assert(std::is_same_v<T, typename BLAS::b_value_type>,
                      "A and B value types must match for TRSM reference");

        constexpr bool     is_left  = (cublasdx::side_of_v<BLAS> == cublasdx::side::left);
        constexpr bool     is_lower = (cublasdx::fill_mode_of_v<BLAS> == cublasdx::fill_mode::lower);
        constexpr bool     is_unit  = (cublasdx::diag_of_v<BLAS> == cublasdx::diag::unit);
        constexpr bool     is_col_a = (cublasdx::arrangement_of_v_a<BLAS> == cublasdx::arrangement::col_major);
        constexpr bool     is_col_b = (cublasdx::arrangement_of_v_b<BLAS> == cublasdx::arrangement::col_major);
        constexpr unsigned M        = cublasdx::size_of<BLAS>::m;
        constexpr unsigned N        = cublasdx::size_of<BLAS>::n;

        return reference_trsm<T>(h_A,
                                 h_B,
                                 M,
                                 N,
                                 batches,
                                 is_left,
                                 is_lower,
                                 is_unit,
                                 is_col_a,
                                 is_col_b,
                                 CUBLAS_OP_N,
                                 warm_up_runs,
                                 runs,
                                 stream);
    }

} // namespace example

#endif // CUBLASDX_EXAMPLE_REFERENCE_TRSM_HPP
