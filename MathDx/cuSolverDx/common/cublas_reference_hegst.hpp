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

#ifndef CUSOLVERDX_EXAMPLE_COMMON_REFERENCE_HEGST_HPP_
#define CUSOLVERDX_EXAMPLE_COMMON_REFERENCE_HEGST_HPP_

#include <cublas_v2.h>
#include <type_traits>
#include <vector>

#include <cusolverdx.hpp>

#include "cudart.hpp"
#include "numeric.hpp"

namespace common {

    namespace detail {

        // Unpack packed Hermitian storage into a dense column-major m×m matrix for cuBLAS.
        // Complex diagonals are forced real (imag 0), matching cuSolverDx/LAPACK Hermitian convention
        template<cusolverdx::fill_mode FillMode, cusolverdx::arrangement ArrangementA, typename T>
        void reference_hegst_unpack_hermitian_col_major(const T* batch_src, std::vector<T>& cm, const unsigned int m) {
            const unsigned int batch_size = m * m;
            cm.resize(batch_size);
            auto index_a = [&](unsigned int row, unsigned int col) {
                if constexpr (ArrangementA == cusolverdx::arrangement::col_major) {
                    return row + col * m;
                } else {
                    return row * m + col;
                }
            };
            for (unsigned int row = 0; row < m; row++) {
                for (unsigned int col = 0; col < m; col++) {
                    T value {};
                    if constexpr (FillMode == cusolverdx::fill_mode::lower) {
                        const auto src_row = (row >= col) ? row : col;
                        const auto src_col = (row >= col) ? col : row;
                        value              = batch_src[index_a(src_row, src_col)];
                        if (row < col) {
                            value = common::conj(value);
                        }
                    } else {
                        const auto src_row = (row <= col) ? row : col;
                        const auto src_col = (row <= col) ? col : row;
                        value              = batch_src[index_a(src_row, src_col)];
                        if (row > col) {
                            value = common::conj(value);
                        }
                    }
                    if constexpr (common::is_complex<T>()) {
                        if (row == col) {
                            value = T{value.real(), 0};
                        }
                    }
                    cm[row + col * m] = value;
                }
            }
        }

        // Unpack packed triangular factor into dense column-major m×m (unused triangle zero).
        template<cusolverdx::fill_mode FillMode, cusolverdx::arrangement ArrangementB, typename T>
        void reference_hegst_unpack_triangular_col_major(const T* batch_src, std::vector<T>& cm, const unsigned int m) {
            const unsigned int batch_size = m * m;
            cm.assign(batch_size, common::convert<T>(0.0));
            auto index_b = [&](unsigned int row, unsigned int col) {
                if constexpr (ArrangementB == cusolverdx::arrangement::col_major) {
                    return row + col * m;
                } else {
                    return row * m + col;
                }
            };
            for (unsigned int row = 0; row < m; row++) {
                for (unsigned int col = 0; col < m; col++) {
                    if constexpr (FillMode == cusolverdx::fill_mode::lower) {
                        if (row >= col) {
                            cm[row + col * m] = batch_src[index_b(row, col)];
                        }
                    } else {
                        if (row <= col) {
                            cm[row + col * m] = batch_src[index_b(row, col)];
                        }
                    }
                }
            }
        }

        // cuBLAS API supports only col-major. Pack dense column-major cm into batch_dst.
        template<cusolverdx::arrangement ArrangementA, typename T>
        void reference_hegst_pack_col_major_to_arrangement(const std::vector<T>& cm, T* batch_dst, const unsigned int m) {
            if constexpr (ArrangementA == cusolverdx::arrangement::col_major) {
                const size_t nbytes = sizeof(T) * static_cast<size_t>(m) * static_cast<size_t>(m);
                std::memcpy(batch_dst, cm.data(), nbytes);
            } else {
                for (unsigned int row = 0; row < m; row++) {
                    for (unsigned int col = 0; col < m; col++) {
                        batch_dst[row * m + col] = cm[row + col * m];
                    }
                }
            }
        }

        // cublas<t>trsm wrapper
        template<typename T, typename cuda_data_type>
        void reference_hegst_cublas_trsm(cublasHandle_t        cublasH,
                                         cublasSideMode_t      side,
                                         cublasFillMode_t      uplo,
                                         cublasOperation_t     trans,
                                         cublasDiagType_t      diag,
                                         const int             mm,
                                         const int             nn,
                                         const cuda_data_type* A,
                                         const int             lda,
                                         cuda_data_type*       B,
                                         const int             ldb) {
            constexpr bool       is_complex = common::is_complex<T>();
            constexpr bool       is_float   = std::is_same_v<typename common::get_precision<T>::type, float>;
            const cuda_data_type alpha      = common::traits<cuda_data_type>::one;
            if constexpr (is_float && !is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasStrsm(cublasH, side, uplo, trans, diag, mm, nn, &alpha, A, lda, B, ldb));
            } else if constexpr (is_float && is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasCtrsm(cublasH, side, uplo, trans, diag, mm, nn, &alpha, A, lda, B, ldb));
            } else if constexpr (!is_float && !is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasDtrsm(cublasH, side, uplo, trans, diag, mm, nn, &alpha, A, lda, B, ldb));
            } else if constexpr (!is_float && is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasZtrsm(cublasH, side, uplo, trans, diag, mm, nn, &alpha, A, lda, B, ldb));
            }
        }

        // cublas<t>trmm wrapper
        template<typename T, typename cuda_data_type>
        void reference_hegst_cublas_trmm(cublasHandle_t        cublasH,
                                         cublasSideMode_t      side,
                                         cublasFillMode_t      uplo,
                                         cublasOperation_t     trans,
                                         cublasDiagType_t      diag,
                                         const int             m,
                                         const int             n,
                                         const cuda_data_type* A,
                                         const int             lda,
                                         const cuda_data_type* B,
                                         const int             ldb,
                                         cuda_data_type*       C,
                                         const int             ldc) {
            constexpr bool       is_complex = common::is_complex<T>();
            constexpr bool       is_float   = std::is_same_v<typename common::get_precision<T>::type, float>;
            const cuda_data_type alpha      = common::traits<cuda_data_type>::one;
            if constexpr (is_float && !is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasStrmm(cublasH, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc));
            } else if constexpr (is_float && is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasCtrmm(cublasH, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc));
            } else if constexpr (!is_float && !is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasDtrmm(cublasH, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc));
            } else if constexpr (!is_float && is_complex) {
                CUBLAS_CHECK_AND_EXIT(cublasZtrmm(cublasH, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb, C, ldc));
            }
        }
    } // namespace detail

    //==========================================================================================================
    // Host cuBLAS reference for HEGST (LAPACK itype EigTypeValue: 1, 2, or 3).
    // EigType 1: two TRSMs. EigType 2/3: two TRMMs (same reduced matrix as prior split helpers).
    //==========================================================================================================
    template<int                    EigTypeValue,
             cusolverdx::fill_mode   FillMode,
             cusolverdx::arrangement ArrangementA,
             cusolverdx::arrangement ArrangementB,
             typename T>
    void cublas_reference_hegst(const std::vector<T>& A_input,
                                const std::vector<T>& B_factor,
                                std::vector<T>&       C_reference,
                                const unsigned int    m,
                                const unsigned int    batches) {
        static_assert(EigTypeValue == 1 || EigTypeValue == 2 || EigTypeValue == 3);

        using cuda_data_type = typename cusolverdx::convert_to_cuda_type<T>::type;

        const unsigned int batch_size = m * m;
        C_reference.resize(batch_size * batches);
        if (batches == 0 || m == 0) {
            return;
        }

        // lda: HEGST matrix A side. ldb: Cholesky factor on B (d_L_or_U).
        const unsigned int lda = m;
        const unsigned int ldb = m;

        cublasHandle_t cublasH = nullptr;
        CUBLAS_CHECK_AND_EXIT(cublasCreate(&cublasH));

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_CHECK_AND_EXIT(cublasSetStream(cublasH, stream));

        const size_t bytes = sizeof(T) * batch_size;

        std::vector<T> h_cm_A;
        std::vector<T> h_cm_fac;
        std::vector<T> h_cm_result(batch_size);
        void* d_pool = nullptr;

        if constexpr (EigTypeValue == 1) {
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_pool), 2 * bytes));
            auto* d_L_or_U = reinterpret_cast<cuda_data_type*>(d_pool);
            auto* d_work   = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + bytes);

            for (unsigned int batch = 0; batch < batches; batch++) {
                const T* in_batch  = A_input.data() + static_cast<size_t>(batch) * batch_size;
                const T* fac_batch = B_factor.data() + static_cast<size_t>(batch) * batch_size;
                T*       ref_batch = C_reference.data() + static_cast<size_t>(batch) * batch_size;

                detail::reference_hegst_unpack_hermitian_col_major<FillMode, ArrangementA, T>(in_batch, h_cm_A, m);
                if constexpr (ArrangementB == cusolverdx::arrangement::col_major) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_L_or_U, fac_batch, bytes, cudaMemcpyHostToDevice, stream));
                } else {
                    detail::reference_hegst_unpack_triangular_col_major<FillMode, ArrangementB, T>(fac_batch, h_cm_fac, m);
                    CUDA_CHECK_AND_EXIT(
                        cudaMemcpyAsync(d_L_or_U, h_cm_fac.data(), bytes, cudaMemcpyHostToDevice, stream));
                }

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_work, h_cm_A.data(), bytes, cudaMemcpyHostToDevice, stream));

                if constexpr (FillMode == cusolverdx::fill_mode::lower) {
                    detail::reference_hegst_cublas_trsm<T, cuda_data_type>(cublasH,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        m, 
                        m,
                        d_L_or_U,
                        ldb,
                        d_work,
                        lda);
                    detail::reference_hegst_cublas_trsm<T, cuda_data_type>(cublasH,
                        CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_LOWER,
                        common::is_complex<T>() ? CUBLAS_OP_C : CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT,
                        m,
                        m,
                        d_L_or_U,
                        ldb,
                        d_work,
                        lda);
                } else {
                    detail::reference_hegst_cublas_trsm<T, cuda_data_type>(cublasH,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_FILL_MODE_UPPER,
                        common::is_complex<T>() ? CUBLAS_OP_C : CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT,
                        m,
                        m,
                        d_L_or_U,
                        ldb,
                        d_work,
                        lda);
                    detail::reference_hegst_cublas_trsm<T, cuda_data_type>(cublasH,
                        CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        m,
                        m,
                        d_L_or_U,
                        ldb,
                        d_work,
                        lda);
                }

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(h_cm_result.data(), d_work, bytes, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

                detail::reference_hegst_pack_col_major_to_arrangement<ArrangementA, T>(h_cm_result, ref_batch, m);
            }

        } else {
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_pool), 4 * bytes));
            auto* d_L_or_U = reinterpret_cast<cuda_data_type*>(d_pool);
            auto* d_A      = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + bytes);
            auto* d_temp   = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + 2 * bytes);
            auto* d_out    = reinterpret_cast<cuda_data_type*>(static_cast<char*>(d_pool) + 3 * bytes);

            for (unsigned int batch = 0; batch < batches; batch++) {
                const T* in_batch  = A_input.data() + static_cast<size_t>(batch) * batch_size;
                const T* fac_batch = B_factor.data() + static_cast<size_t>(batch) * batch_size;
                T*       ref_batch = C_reference.data() + static_cast<size_t>(batch) * batch_size;

                detail::reference_hegst_unpack_hermitian_col_major<FillMode, ArrangementA, T>(in_batch, h_cm_A, m);
                if constexpr (ArrangementB == cusolverdx::arrangement::col_major) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_L_or_U, fac_batch, bytes, cudaMemcpyHostToDevice, stream));
                } else {
                    detail::reference_hegst_unpack_triangular_col_major<FillMode, ArrangementB, T>(fac_batch, h_cm_fac, m);
                    CUDA_CHECK_AND_EXIT(
                        cudaMemcpyAsync(d_L_or_U, h_cm_fac.data(), bytes, cudaMemcpyHostToDevice, stream));
                }

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, h_cm_A.data(), bytes, cudaMemcpyHostToDevice, stream));

                if constexpr (FillMode == cusolverdx::fill_mode::lower) {
                    detail::reference_hegst_cublas_trmm<T, cuda_data_type>(cublasH,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_FILL_MODE_LOWER,
                        common::is_complex<T>() ? CUBLAS_OP_C : CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT,
                        m,
                        m,
                        d_L_or_U,
                        ldb,
                        d_A,
                        lda,
                        d_temp,
                        lda);
                    detail::reference_hegst_cublas_trmm<T, cuda_data_type>(cublasH,
                        CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        m,
                        m,
                        d_L_or_U,
                        ldb,
                        d_temp,
                        lda,
                        d_out,
                        lda);
                } else {
                    detail::reference_hegst_cublas_trmm<T, cuda_data_type>(cublasH,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_NON_UNIT,
                        m,
                        m,
                        d_L_or_U,
                        ldb,
                        d_A,
                        lda,
                        d_temp,
                        lda);
                    detail::reference_hegst_cublas_trmm<T, cuda_data_type>(cublasH,
                        CUBLAS_SIDE_RIGHT,
                        CUBLAS_FILL_MODE_UPPER,
                        common::is_complex<T>() ? CUBLAS_OP_C : CUBLAS_OP_T,
                        CUBLAS_DIAG_NON_UNIT,
                        m,
                        m,
                        d_L_or_U,
                        ldb,
                        d_temp,
                        lda,
                        d_out,
                        lda);
                }

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(h_cm_result.data(), d_out, bytes, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

                detail::reference_hegst_pack_col_major_to_arrangement<ArrangementA, T>(h_cm_result, ref_batch, m);
            }

        }
        CUDA_CHECK_AND_EXIT(cudaFree(d_pool));

        CUBLAS_CHECK_AND_EXIT(cublasDestroy(cublasH));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_REFERENCE_HEGST_HPP_
