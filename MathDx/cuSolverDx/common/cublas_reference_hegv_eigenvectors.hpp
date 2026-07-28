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

#ifndef CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUBLAS_REFERENCE_HEGV_EIGENVECTORS_HPP
#define CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUBLAS_REFERENCE_HEGV_EIGENVECTORS_HPP

#include <cublas_v2.h>
#include <vector>

#include <cusolverdx.hpp>

#include "cudart.hpp"
#include "random.hpp"
#include "error_checking.hpp"
#include "numeric.hpp"

namespace common {

    namespace detail {

        // HEGV only references the stored triangle of A and B; mirror it so cuBLAS uses the same Hermitian A and B.
        template<typename T, cusolverdx::fill_mode FillMode>
        void complete_hermitian_from_stored_triangle(std::vector<T>&    M,
                                                      const unsigned int m,
                                                      const unsigned int lda,
                                                      const unsigned int batches) {
            constexpr bool upper = FillMode == cusolverdx::fill_mode::upper;
            for (unsigned int b = 0; b < batches; ++b) {
                T* batch = M.data() + static_cast<size_t>(b) * m * m;
                for (unsigned int col = 0; col < m; ++col) {
                    for (unsigned int row = 0; row < m; ++row) {
                        const unsigned int off          = row + col * lda;
                        const bool         in_stored_triangle = upper ? (row <= col) : (row >= col);
                        if (!in_stored_triangle) {
                            const unsigned int off_up = col + row * lda;
                            batch[off]                = common::conj(batch[off_up]);
                        }
                        // LAPACK/cuSolverDx Hermitian convention: diagonal imaginary parts are zero.
                        if constexpr (common::is_complex<T>()) {
                            if (row == col) {
                                batch[off].imag(0);
                            }
                        }
                    }
                }
            }
        }

        template<typename T, typename cuda_data_type>
        void reference_hegv_cublas_gemm(cublasHandle_t        cublasH,
                                        cublasOperation_t     transa,
                                        cublasOperation_t     transb,
                                        const int             m,
                                        const cuda_data_type* A,
                                        const int             lda,
                                        const cuda_data_type* B,
                                        const int             ldb,
                                        cuda_data_type*       C,
                                        const int             ldc,
                                        const cuda_data_type  alpha,
                                        const cuda_data_type  beta) {
            constexpr bool is_complex = common::is_complex<T>();
            constexpr bool is_float   = std::is_same_v<typename common::get_precision<T>::type, float>;
            if constexpr (is_float && !is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasSgemm(cublasH, transa, transb, m, m, m, &alpha, A, lda, B, ldb, &beta, C, ldc));
            } else if constexpr (is_float && is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasCgemm(cublasH, transa, transb, m, m, m, &alpha, A, lda, B, ldb, &beta, C, ldc));
            } else if constexpr (!is_float && !is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasDgemm(cublasH, transa, transb, m, m, m, &alpha, A, lda, B, ldb, &beta, C, ldc));
            } else if constexpr (!is_float && is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasZgemm(cublasH, transa, transb, m, m, m, &alpha, A, lda, B, ldb, &beta, C, ldc));
            }
        }

        // VL = V * diag(lambda) in column-major storage (same as blas::dimm in test_hegv.cu).
        // Eigenvalues are real for Hermitian-definite problems.
        template<typename T, typename cuda_data_type, typename precision_type>
        void reference_hegv_form_vl(cudaStream_t            stream,
                                    cublasHandle_t          cublasH,
                                    const int               m,
                                    const size_t            bytes,
                                    const precision_type*   lambda,
                                    const cuda_data_type*   V,
                                    cuda_data_type*         VL) {
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(VL, V, bytes, cudaMemcpyDeviceToDevice, stream));
            CUBLAS_CHECK_AND_EXIT(cublasSetPointerMode(cublasH, CUBLAS_POINTER_MODE_HOST));
            for (int col = 0; col < m; ++col) {
                const precision_type scale = lambda[col];
                if constexpr (std::is_same_v<cuda_data_type, float>) {
                    CUBLAS_CHECK_AND_EXIT(cublasSscal(cublasH, m, &scale, VL + col * m, 1));
                } else if constexpr (std::is_same_v<cuda_data_type, double>) {
                    CUBLAS_CHECK_AND_EXIT(cublasDscal(cublasH, m, &scale, VL + col * m, 1));
                } else if constexpr (std::is_same_v<cuda_data_type, cuComplex>) {
                    CUBLAS_CHECK_AND_EXIT(cublasCsscal(cublasH, m, &scale, VL + col * m, 1));
                } else {
                    CUBLAS_CHECK_AND_EXIT(cublasZdscal(cublasH, m, &scale, VL + col * m, 1));
                }
            }
        }

    } // namespace detail

    // Compare left- and right-hand sides of the generalized eigenvector identity (test_hegv.cu).
    //   type 1: AV  vs B*VL
    //   type 2: A*B*V vs VL
    //   type 3: B*A*V vs VL
    template<int                     EigTypeValue,
             cusolverdx::fill_mode   FillMode,
             cusolverdx::arrangement ArrangementA,
             cusolverdx::arrangement ArrangementB,
             typename T,
             typename Precision = typename common::get_precision<T>::type>
    double cublas_reference_hegv_eigenvector_verification(std::vector<T>&               A,
                                                          std::vector<T>&               B,
                                                          std::vector<T>&               V,
                                                          const std::vector<Precision>& lambda,
                                                          const unsigned int            m,
                                                          const unsigned int            batches) {
        static_assert(EigTypeValue == 1 || EigTypeValue == 2 || EigTypeValue == 3);

        using cuda_data_type = typename cusolverdx::convert_to_cuda_type<T>::type;

        const unsigned int batch_size = m * m;
        const int          lda        = m;
        const int          ldb        = m;

        cublasHandle_t cublasH = nullptr;
        CUBLAS_CHECK_AND_EXIT(cublasCreate(&cublasH));

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK_AND_EXIT(cublasSetStream(cublasH, stream));

        const cuda_data_type alpha_one = common::traits<cuda_data_type>::one;
        const cuda_data_type beta_zero = common::traits<cuda_data_type>::zero;

        const size_t bytes = sizeof(cuda_data_type) * batch_size;

        const unsigned int stored_batches = static_cast<unsigned int>(A.size() / batch_size);

        // Row-major host data -> column-major for cuBLAS (same as reference_cusolver_hegv).
        if constexpr (ArrangementA == cusolverdx::arrangement::row_major) {
            common::transpose_matrix<T>(A, lda, m, stored_batches);
            common::transpose_matrix<T>(V, lda, m, stored_batches);
        }
        if constexpr (ArrangementB == cusolverdx::arrangement::row_major) {
            common::transpose_matrix<T>(B, ldb, m, stored_batches);
        }

        // A ana B are generated with fillup_random_matrix symm=false), so it needs to be processed before running GEMM
        detail::complete_hermitian_from_stored_triangle<T, FillMode>(B, m, ldb, stored_batches);
        detail::complete_hermitian_from_stored_triangle<T, FillMode>(A, m, lda, stored_batches);

        std::vector<T> h_left(batch_size);
        std::vector<T> h_right(batch_size);

        void* d_pool = nullptr;
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_pool), 6 * bytes));

        auto* d_A     = reinterpret_cast<cuda_data_type*>(d_pool);
        auto* d_B     = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + bytes);
        auto* d_V     = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + 2 * bytes);
        auto* d_VL    = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + 3 * bytes);
        auto* d_left  = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + 4 * bytes);
        auto* d_right = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + 5 * bytes);

        double worst_relative = 0.0;

        for (unsigned int batch = 0; batch < batches; ++batch) {
            const T*         a_batch = A.data() + static_cast<size_t>(batch) * batch_size;
            const T*         b_batch = B.data() + static_cast<size_t>(batch) * batch_size;
            const T*         v_batch = V.data() + static_cast<size_t>(batch) * batch_size;
            const Precision* lam_b   = lambda.data() + static_cast<size_t>(batch) * m;

            std::vector<Precision> h_lambda(lam_b, lam_b + m);

            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, a_batch, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, b_batch, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_V, v_batch, bytes, cudaMemcpyHostToDevice, stream));

            detail::reference_hegv_form_vl<T, cuda_data_type, Precision>(
                stream, cublasH, m, bytes, h_lambda.data(), d_V, d_VL);

            if constexpr (EigTypeValue == 1) {
                // left = A*V, right = B*VL
                detail::reference_hegv_cublas_gemm<T, cuda_data_type>(
                    cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, d_A, lda, d_V, lda, d_left, lda, alpha_one, beta_zero);
                detail::reference_hegv_cublas_gemm<T, cuda_data_type>(
                    cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, d_B, ldb, d_VL, ldb, d_right, ldb, alpha_one, beta_zero);
            } else if constexpr (EigTypeValue == 2) {
                // left = A*B*V, right = VL (d_right holds B*V until replaced by VL)
                detail::reference_hegv_cublas_gemm<T, cuda_data_type>(
                    cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, d_B, ldb, d_V, ldb, d_right, ldb, alpha_one, beta_zero);
                detail::reference_hegv_cublas_gemm<T, cuda_data_type>(
                    cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, d_A, lda, d_right, ldb, d_left, lda, alpha_one, beta_zero);
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_right, d_VL, bytes, cudaMemcpyDeviceToDevice, stream));
            } else {
                // left = B*A*V, right = VL (d_right holds A*V until replaced by VL)
                detail::reference_hegv_cublas_gemm<T, cuda_data_type>(
                    cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, d_A, lda, d_V, lda, d_right, ldb, alpha_one, beta_zero);
                detail::reference_hegv_cublas_gemm<T, cuda_data_type>(
                    cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, d_B, ldb, d_right, ldb, d_left, ldb, alpha_one, beta_zero);
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_right, d_VL, bytes, cudaMemcpyDeviceToDevice, stream));
            }

            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(h_left.data(), d_left, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(h_right.data(), d_right, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            // Elementwise relative error between left- and right-hand side matrices (not a custom norm bound).
            const double relative = common::check_error<T, T>(h_left.data(), h_right.data(), 1);
            worst_relative        = std::max(worst_relative, relative);
        }

        CUDA_CHECK_AND_EXIT(cudaFree(d_pool));
        CUBLAS_CHECK_AND_EXIT(cublasDestroy(cublasH));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

        return worst_relative;
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUBLAS_REFERENCE_HEGV_EIGENVECTORS_HPP
