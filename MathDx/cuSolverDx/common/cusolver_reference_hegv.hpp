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

#ifndef CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_HEGV_HPP
#define CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_HEGV_HPP

#include "measure.hpp"
#include "random.hpp"

namespace common {

    template<typename T,
             typename cuda_data_type,
             typename precision_type = typename common::get_precision<T>::type,
             bool use_sygvj          = true>
    bool reference_cusolver_hegv(std::vector<T>&              A,
                                 std::vector<T>&              B,
                                 std::vector<precision_type>& lambda,
                                 int*                         info,
                                 const unsigned int           m,
                                 const int                    eig_type,
                                 const unsigned int           padded_batches  = 1,
                                 bool                         is_lower_fill   = true,
                                 bool                         is_col_major_a  = true,
                                 bool                         is_col_major_b  = true,
                                 bool                         compute_vectors = false,
                                 const unsigned int           actual_batches  = 0) {

        const unsigned int a_size = A.size() / padded_batches;
        const unsigned int b_size = B.size() / padded_batches;
        const unsigned int lda    = m;
        const unsigned int ldb    = m;

        unsigned int batches = (actual_batches == 0) ? padded_batches : actual_batches;

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        cusolverDnHandle_t cusolverH = nullptr;
        CUSOLVER_CHECK_AND_EXIT(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnSetStream(cusolverH, stream));

        const cublasFillMode_t  uplo  = is_lower_fill ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        const cusolverEigMode_t jobz  = compute_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        const cusolverEigType_t itype = static_cast<cusolverEigType_t>(eig_type);

        if (!is_col_major_a) {
            transpose_matrix<T>(A, lda, m, batches);
        }
        if (!is_col_major_b) {
            transpose_matrix<T>(B, ldb, m, batches);
        }

        cuda_data_type* d_A      = nullptr;
        cuda_data_type* d_B      = nullptr;
        precision_type* d_lambda = nullptr;
        int*            d_info   = nullptr;
        int             lwork    = 0;
        cuda_data_type* d_work   = nullptr;
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * a_size));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(T) * b_size));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_lambda), sizeof(precision_type) * m));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));

        constexpr bool is_complex = common::is_complex<T>();
        constexpr bool is_float   = std::is_same_v<precision_type, float>;

        if constexpr (use_sygvj) { // Use cuSolverDn<t>sygvj
            const int   max_sweeps   = 15;
            const int   sort_eig     = 1;
            syevjInfo_t syevj_params = nullptr;
            CUSOLVER_CHECK_AND_EXIT(cusolverDnCreateSyevjInfo(&syevj_params));
            CUSOLVER_CHECK_AND_EXIT(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));
            CUSOLVER_CHECK_AND_EXIT(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig));

            if constexpr (is_float && !is_complex) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnSsygvj_bufferSize(
                    cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, &lwork, syevj_params));
            } else if constexpr (is_float && is_complex) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnChegvj_bufferSize(
                    cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, &lwork, syevj_params));
            } else if constexpr (!is_float && !is_complex) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnDsygvj_bufferSize(
                    cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, &lwork, syevj_params));
            } else if constexpr (!is_float && is_complex) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnZhegvj_bufferSize(
                    cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, &lwork, syevj_params));
            }

            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(cuda_data_type) * lwork));

            auto execute_one_batch = [&](cudaStream_t str) {
                if constexpr (is_float && !is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnSsygvj(
                        cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, d_work, lwork, d_info, syevj_params));
                } else if constexpr (is_float && is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnChegvj(
                        cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, d_work, lwork, d_info, syevj_params));
                } else if constexpr (!is_float && !is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnDsygvj(
                        cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, d_work, lwork, d_info, syevj_params));
                } else if constexpr (!is_float && is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnZhegvj(
                        cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, d_work, lwork, d_info, syevj_params));
                }
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
            };

            for (unsigned int b = 0; b < batches; ++b) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
                    d_A, A.data() + static_cast<size_t>(b) * a_size, sizeof(T) * a_size, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
                    d_B, B.data() + static_cast<size_t>(b) * b_size, sizeof(T) * b_size, cudaMemcpyHostToDevice, stream));

                execute_one_batch(stream);

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(lambda.data() + static_cast<size_t>(b) * m,
                                                    d_lambda,
                                                    sizeof(precision_type) * m,
                                                    cudaMemcpyDeviceToHost,
                                                    stream));
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info + b, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
                if (compute_vectors) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
                        A.data() + static_cast<size_t>(b) * a_size, d_A, sizeof(T) * a_size, cudaMemcpyDeviceToHost, stream));
                }
            }

            CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroySyevjInfo(syevj_params));

        } else { // Use cuSolverDn<t>sygvd
            if constexpr (is_float && !is_complex) {
                CUSOLVER_CHECK_AND_EXIT(
                    cusolverDnSsygvd_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, &lwork));
            } else if constexpr (is_float && is_complex) {
                CUSOLVER_CHECK_AND_EXIT(
                    cusolverDnChegvd_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, &lwork));
            } else if constexpr (!is_float && !is_complex) {
                CUSOLVER_CHECK_AND_EXIT(
                    cusolverDnDsygvd_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, &lwork));
            } else if constexpr (!is_float && is_complex) {
                CUSOLVER_CHECK_AND_EXIT(
                    cusolverDnZhegvd_bufferSize(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, &lwork));
            }

            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(cuda_data_type) * lwork));

            auto execute_one_batch = [&](cudaStream_t str) {
                if constexpr (is_float && !is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(
                        cusolverDnSsygvd(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, d_work, lwork, d_info));
                } else if constexpr (is_float && is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(
                        cusolverDnChegvd(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, d_work, lwork, d_info));
                } else if constexpr (!is_float && !is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(
                        cusolverDnDsygvd(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, d_work, lwork, d_info));
                } else if constexpr (!is_float && is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(
                        cusolverDnZhegvd(cusolverH, itype, jobz, uplo, m, d_A, lda, d_B, ldb, d_lambda, d_work, lwork, d_info));
                }
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
            };

            for (unsigned int b = 0; b < batches; ++b) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
                    d_A, A.data() + static_cast<size_t>(b) * a_size, sizeof(T) * a_size, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
                    d_B, B.data() + static_cast<size_t>(b) * b_size, sizeof(T) * b_size, cudaMemcpyHostToDevice, stream));

                execute_one_batch(stream);

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(lambda.data() + static_cast<size_t>(b) * m,
                                                    d_lambda,
                                                    sizeof(precision_type) * m,
                                                    cudaMemcpyDeviceToHost,
                                                    stream));
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info + b, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
                if (compute_vectors) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(
                        A.data() + static_cast<size_t>(b) * a_size, d_A, sizeof(T) * a_size, cudaMemcpyDeviceToHost, stream));
                }
            }
        }

        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

        CUDA_CHECK_AND_EXIT(cudaFree(d_A));
        CUDA_CHECK_AND_EXIT(cudaFree(d_B));
        CUDA_CHECK_AND_EXIT(cudaFree(d_lambda));
        CUDA_CHECK_AND_EXIT(cudaFree(d_info));
        CUDA_CHECK_AND_EXIT(cudaFree(d_work));

        for (unsigned int i = 0; i < batches; ++i) {
            if (info[i] != 0) {
                if constexpr (use_sygvj) {
                    if constexpr (common::is_complex<T>()) {
                        std::cout << "non-zero info returned with cuSolver hegvj for batch #" << i << " (info=" << info[i] << ")"
                                  << std::endl;
                    } else {
                        std::cout << "non-zero info returned with cuSolver sygvj for batch #" << i << " (info=" << info[i] << ")"
                                  << std::endl;
                    }
                } else {
                    if constexpr (common::is_complex<T>()) {
                        std::cout << "non-zero info returned with cuSolver hegvd for batch #" << i << " (info=" << info[i] << ")"
                                  << std::endl;
                    } else {
                        std::cout << "non-zero info returned with cuSolver sygvd for batch #" << i << " (info=" << info[i] << ")"
                                  << std::endl;
                    }
                }
                CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroy(cusolverH));
                CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
                return false;
            }
        }

        if (!is_col_major_a) {
            transpose_matrix<T>(A, m, lda, batches);
        }
        if (!is_col_major_b) {
            transpose_matrix<T>(B, m, ldb, batches);
        }

        CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroy(cusolverH));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
        return true;
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_HEGV_HPP
