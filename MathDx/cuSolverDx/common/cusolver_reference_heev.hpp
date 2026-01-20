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

#ifndef CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_HEEV_HPP
#define CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_HEEV_HPP

#include "measure.hpp"

namespace common {
    template<typename T, typename cuda_data_type, typename precision_type = typename common::get_precision<T>::type, bool use_syevj = false, bool check_perf = false>
    bool reference_cusolver_heev(std::vector<T>&              A,
                                 std::vector<precision_type>& lambda,
                                 int*                         info,
                                 const unsigned int           m,
                                 const unsigned int           padded_batches = 1,
                                 bool                         is_lower_fill  = true,
                                 bool                         is_col_major_a = true,
                                 bool                         compute_vectors = false,
                                 const unsigned int           actual_batches = 0) {

        const unsigned int a_size = A.size() / padded_batches;
        const unsigned int lda    = m;

        unsigned int batches = (actual_batches == 0) ? padded_batches : actual_batches;

        // note that cusolverDnXsyevBatched only support n*lda*batchSize<=INT32_MAX, so we need to adjust the batch size if it is too large
        if constexpr (!use_syevj && check_perf) {
            if (m * lda * batches > INT_MAX) {
                batches = INT_MAX / (m * lda);
                std::cout << "m * lda * batches > INT_MAX, adjust the batch size to " << batches << std::endl;
            }
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        cusolverDnHandle_t cusolverH = nullptr;
        CUSOLVER_CHECK_AND_EXIT(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnSetStream(cusolverH, stream));

        const cublasFillMode_t  uplo = (is_lower_fill) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        const cusolverEigMode_t jobz = (compute_vectors) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR; /* compute eigenvectors */

        [[maybe_unused]] double ms_syev_cusolver;

        // if row major, transpose the input A
        if (!is_col_major_a) {
            // For row major, transpose the matrix before calling cuSolver
            transpose_matrix<T>(A, lda, m, batches);
        }

        //============================================
        // Use cuSolver syevjBatched API for both single and multiple batches
        if constexpr (use_syevj) {
            cuda_data_type* d_A      = nullptr;
            precision_type* d_lambda = nullptr;
            int*            d_info   = nullptr;
            int             lwork    = 0;

            /* configuration of syevj  */
            const int    max_sweeps = 15;
            const int    sort_eig   = 1; /* sort eigenvalues */

            syevjInfo_t syevj_params = NULL;
            CUSOLVER_CHECK_AND_EXIT(cusolverDnCreateSyevjInfo(&syevj_params));
            CUSOLVER_CHECK_AND_EXIT(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));
            CUSOLVER_CHECK_AND_EXIT(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig));

            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * a_size * batches));
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_lambda), sizeof(precision_type) * m * batches));
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * batches));

            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * a_size * batches, cudaMemcpyHostToDevice, stream));

            // Query workspace size for syevjBatched
            constexpr bool is_complex = common::is_complex<T>();
            constexpr bool is_float   = std::is_same_v<precision_type, float>;

            if constexpr (is_float && !is_complex) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnSsyevjBatched_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_lambda, &lwork, syevj_params, batches));
            } else if constexpr (is_float && is_complex) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnCheevjBatched_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_lambda, &lwork, syevj_params, batches));
            } else if constexpr (!is_float && !is_complex) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnDsyevjBatched_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_lambda, &lwork, syevj_params, batches));
            } else if constexpr (!is_float && is_complex) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnZheevjBatched_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_lambda, &lwork, syevj_params, batches));
            }

            // Allocate workspace
            cuda_data_type* d_work = nullptr;
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(cuda_data_type) * lwork));

            // Execute batched syevj (Jacobi-based eigenvalue solver)
            auto execute_syevj_api = [&](cudaStream_t str) {
                constexpr bool is_complex = common::is_complex<T>();
                constexpr bool is_float   = std::is_same_v<precision_type, float>;
                if constexpr (is_float && !is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnSsyevjBatched(cusolverH, jobz, uplo, m, d_A, lda, d_lambda, d_work, lwork, d_info, syevj_params, batches));
                } else if constexpr (is_float && is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnCheevjBatched(cusolverH, jobz, uplo, m, d_A, lda, d_lambda, d_work, lwork, d_info, syevj_params, batches));
                } else if constexpr (!is_float && !is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnDsyevjBatched(cusolverH, jobz, uplo, m, d_A, lda, d_lambda, d_work, lwork, d_info, syevj_params, batches));
                } else if constexpr (!is_float && is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnZheevjBatched(cusolverH, jobz, uplo, m, d_A, lda, d_lambda, d_work, lwork, d_info, syevj_params, batches));
                }
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
            };


            if constexpr (!check_perf) {
                execute_syevj_api(stream); // execute syevjBatched
            } else {
                [[maybe_unused]] auto execute_reset_a = [&](cudaStream_t str) { CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * a_size * batches, cudaMemcpyHostToDevice, str)); };

                const unsigned int warmup_repeats = 1;
                const unsigned int repeats        = 1;
                ms_syev_cusolver                  = common::measure::execution(execute_syevj_api, execute_reset_a, warmup_repeats, repeats, stream) / repeats;
            }

            // Copy results back
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(lambda.data(), d_lambda, sizeof(precision_type) * m * batches, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info, d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));
            if (compute_vectors) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A.data(), d_A, sizeof(T) * a_size * batches, cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            // Cleanup
            CUDA_CHECK_AND_EXIT(cudaFree(d_A));
            CUDA_CHECK_AND_EXIT(cudaFree(d_lambda));
            CUDA_CHECK_AND_EXIT(cudaFree(d_info));
            CUDA_CHECK_AND_EXIT(cudaFree(d_work));


            //============================================
            // Use cuSolver 64-bit API cusolverDnXsyevBatched for both single and multiple batches
        } else {
            void*             d_A                      = nullptr;
            void*             d_lambda                 = nullptr;
            int*              d_info                   = nullptr;
            size_t            workspaceInBytesOnDevice = 0;       /* size of workspace */
            void*             d_work                   = nullptr; /* device workspace */
            size_t            workspaceInBytesOnHost   = 0;       /* size of workspace */
            std::vector<char> h_work;                             /* host workspace */

            CUDA_CHECK_AND_EXIT(cudaMalloc(&d_A, sizeof(cuda_data_type) * a_size * batches));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * a_size * batches, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaMalloc(&d_lambda, sizeof(precision_type) * m * batches));
            CUDA_CHECK_AND_EXIT(cudaMalloc(&d_info, sizeof(int) * batches));

            cusolverDnParams_t params = nullptr;
            CUSOLVER_CHECK_AND_EXIT(cusolverDnCreateParams(&params));

            // query working space
            CUSOLVER_CHECK_AND_EXIT(cusolverDnXsyevBatched_bufferSize(cusolverH,
                                                                      params,
                                                                      jobz,
                                                                      uplo,
                                                                      (int64_t)m,
                                                                      common::traits<cuda_data_type>::cuda_data_type,
                                                                      d_A,
                                                                      (int64_t)lda,
                                                                      common::traits<precision_type>::cuda_data_type,
                                                                      d_lambda,
                                                                      common::traits<cuda_data_type>::cuda_data_type, // computeType
                                                                      &workspaceInBytesOnDevice,
                                                                      &workspaceInBytesOnHost,
                                                                      (int64_t)batches));

            CUDA_CHECK_AND_EXIT(cudaMalloc(&d_work, workspaceInBytesOnDevice));

            if (0 < workspaceInBytesOnHost) {
                h_work.resize(workspaceInBytesOnHost);
            }

            // Execute syevBatched
            auto execute_syev_api = [&](cudaStream_t str) {
                CUSOLVER_CHECK_AND_EXIT(cusolverDnXsyevBatched(cusolverH,
                                                               params,
                                                               jobz,
                                                               uplo,
                                                               (int64_t)m,
                                                               common::traits<cuda_data_type>::cuda_data_type,
                                                               d_A,
                                                               (int64_t)lda,
                                                               common::traits<precision_type>::cuda_data_type,
                                                               d_lambda,
                                                               common::traits<cuda_data_type>::cuda_data_type,
                                                               d_work,
                                                               workspaceInBytesOnDevice,
                                                               h_work.data(),
                                                               workspaceInBytesOnHost,
                                                               d_info,
                                                               (int64_t)batches));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
            };

            if constexpr (!check_perf) {
                execute_syev_api(stream); // execute syevBatched
            } else {
                [[maybe_unused]] auto execute_reset_a = [&](cudaStream_t str) { CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * a_size * batches, cudaMemcpyHostToDevice, str)); };

                const unsigned int warmup_repeats = 1;
                const unsigned int repeats        = 1;
                ms_syev_cusolver                  = common::measure::execution(execute_syev_api, execute_reset_a, warmup_repeats, repeats, stream) / repeats;
            }

            // Copy results back
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(lambda.data(), d_lambda, sizeof(precision_type) * m * batches, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info, d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));
            if (compute_vectors) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A.data(), d_A, sizeof(T) * a_size * batches, cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            CUDA_CHECK_AND_EXIT(cudaFree(d_A));
            CUDA_CHECK_AND_EXIT(cudaFree(d_lambda));
            CUDA_CHECK_AND_EXIT(cudaFree(d_info));
            CUDA_CHECK_AND_EXIT(cudaFree(d_work));
            CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroyParams(params));
        }

        // Check for errors
        for (auto i = 0; i < batches; i++) {
            if (*(info + i) != 0) {
                std::cout << "non-zero d_info returned with cuSolver syevjBatched for batch #" << i << std::endl;
                CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroy(cusolverH));
                CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
                return false;
            }
        }

        if constexpr (check_perf) {
            double seconds_per_giga_batch = ms_syev_cusolver / 1e3 / batches * 1e9;
            double gb_s                   = (m * (m + 1) * sizeof(T) + m * sizeof(precision_type)) / seconds_per_giga_batch;                                 // A read, half write, and lambda write
            common::print_perf((use_syevj) ? "Ref_cuSolverDn<t>syevjBatched" : "Ref_cuSolverDnXsyevBatched", batches, m, m, 0, 0, gb_s, ms_syev_cusolver, 0); // dummy 0 for nrhs, gflops, and blockDim
        }

        // if row major, transpose the result A back
        if (!is_col_major_a) {
            transpose_matrix<T>(A, m, lda, batches);
        }

        CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroy(cusolverH));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
        return true;
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_HEEV_HPP
