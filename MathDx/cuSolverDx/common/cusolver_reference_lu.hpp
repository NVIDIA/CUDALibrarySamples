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

#ifndef CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_LU_HPP
#define CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_LU_HPP

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "measure.hpp"

namespace common {
    template<typename T, typename cuda_data_type, bool do_solver = false, bool check_blas_getrf_perf = false>
    bool reference_cusolver_lu(std::vector<T>&    A,
                               std::vector<T>&    B,
                               int*               info,
                               const unsigned int m,
                               const unsigned int n,
                               const unsigned int nrhs           = 1,
                               const unsigned int padded_batches = 1,
                               const bool         is_pivot       = false,
                               bool               is_col_major_a = true,
                               bool               is_col_major_b = true,
                               bool               is_trans_a     = false,
                               int64_t*           ipiv           = nullptr,
                               const unsigned int actual_batches = 0) {

        const unsigned int a_size = A.size() / padded_batches;
        const unsigned int lda    = a_size / n;
        const unsigned int mn     = min(m, n);

        [[maybe_unused]] const unsigned int b_size = B.size() / padded_batches;
        [[maybe_unused]] const unsigned int ldb    = b_size / nrhs; // ldb if b is column major

        const unsigned batches = (actual_batches == 0) ? padded_batches : actual_batches;

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // if row major, transpose the input A
        if (!is_col_major_a) {
            transpose_matrix<T>(A, lda, n, batches); // fast, second_fast, batch -> after transpose, swap fast and second_fast
        }
        if (!is_col_major_b && do_solver && nrhs > 1) {
            transpose_matrix<T>(B, ldb, nrhs, batches); // fast, second_fast, batch -> after transpose, swap fast and second_fast
        }

        [[maybe_unused]] cublasOperation_t trans = (is_trans_a) ? (common::is_complex<T>() ? CUBLAS_OP_C : CUBLAS_OP_T) : CUBLAS_OP_N;

        //================== Check performance using cuBlas<t>getrfBatched
        if constexpr (check_blas_getrf_perf) {
            if ( m != n) {
                printf("cublas<t>getrfBatched only supported for square matrices\n");
                return false;
            }

            // create cublas handle, bind a stream
            cublasHandle_t cublasH = nullptr;
            CUBLAS_CHECK_AND_EXIT(cublasCreate(&cublasH));
            CUBLAS_CHECK_AND_EXIT(cublasSetStream(cublasH, stream));

            // d_info
            int* d_info = nullptr;
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * batches));
            int* d_ipiv = nullptr;
            if (is_pivot) {
                CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_ipiv), sizeof(int) * mn * batches));
            }

            cuda_data_type** d_A_array = nullptr;

            std::vector<cuda_data_type*> d_A(batches, nullptr);

            for (int i = 0; i < batches; i++) {
                CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(cuda_data_type) * a_size));
            }

            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(cuda_data_type*) * batches));

            for (int i = 0; i < batches; i++) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A[i], A.data() + i * a_size, sizeof(cuda_data_type) * a_size, cudaMemcpyHostToDevice, stream));
            }

            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(cuda_data_type*) * batches, cudaMemcpyHostToDevice, stream));

            auto execute_cublas_getrf_api = [&](cudaStream_t str) {
                constexpr bool is_complex = common::is_complex<T>();
                constexpr bool is_float   = std::is_same_v<typename common::get_precision<T>::type, float>;
                if constexpr (is_float && !is_complex) {
                    CUBLAS_CHECK_AND_EXIT(cublasSgetrfBatched(cublasH, m, d_A_array, lda, d_ipiv, d_info, batches));
                } else if constexpr (is_float && is_complex) {
                    CUBLAS_CHECK_AND_EXIT(cublasCgetrfBatched(cublasH, m, d_A_array, lda, d_ipiv, d_info, batches));
                } else if constexpr (!is_float && !is_complex) {
                    CUBLAS_CHECK_AND_EXIT(cublasDgetrfBatched(cublasH, m, d_A_array, lda, d_ipiv, d_info, batches));
                } else if constexpr (!is_float && is_complex) {
                    CUBLAS_CHECK_AND_EXIT(cublasZgetrfBatched(cublasH, m, d_A_array, lda, d_ipiv, d_info, batches));
                }
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
            };

            // cuBlasXgetrfBatched forces in-place A, so need to reset A after execution to get appropriate performance data
            auto execute_reset_a = [&](cudaStream_t str) {
                for (int i = 0; i < batches; i++) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A[i], A.data() + i * a_size, sizeof(cuda_data_type) * a_size, cudaMemcpyHostToDevice, str));
                }
            };
            // measure and report cusolver API performance
            const unsigned int warmup_repeats = 1;
            const unsigned int repeats        = 1;

            double ms_getrf = common::measure::execution(execute_cublas_getrf_api, execute_reset_a, warmup_repeats, repeats, stream) / repeats;

            for (int i = 0; i < batches; i++) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A.data() + i * a_size, d_A[i], sizeof(cuda_data_type) * a_size, cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info, d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            CUDA_CHECK_AND_EXIT(cudaFree(d_A_array));
            for (int i = 0; i < batches; i++) {
                CUDA_CHECK_AND_EXIT(cudaFree(d_A[i]));
            }
            if (is_pivot) {
                CUDA_CHECK_AND_EXIT(cudaFree(d_ipiv));
            }
            CUDA_CHECK_AND_EXIT(cudaFree(d_info));
            CUBLAS_CHECK_AND_EXIT(cublasDestroy(cublasH));

            // check for errors
            for (int i = 0; i < batches; i++) {
                if (info[i] != 0) {
                    printf("cublas<t>getrfBatched %d-th batch is wrong, info = %d \n", i, info[i]);
                    return false;
                }
            }

            // report the timing
            double seconds_per_giga_batch = ms_getrf / 1e3 / batches * 1e9;
            double gb_s                   = a_size * 2 * sizeof(T) / seconds_per_giga_batch; // A read and write
            double gflops                 = common::get_flops_getrf<T>(m, n) / seconds_per_giga_batch;
            common::print_perf("Ref_cublasXgetrfBatched", batches, m, n, 0, gflops, gb_s, ms_getrf, 0); // dummy 0 for k andblockDim


            //================== Check correctness using cuSolverDnXgetrf and cuSolverDnXgetrs
        } else {
            cusolverDnHandle_t cusolverH = NULL;
            cusolverDnParams_t params    = nullptr;
            CUSOLVER_CHECK_AND_EXIT(cusolverDnCreate(&cusolverH));
            CUSOLVER_CHECK_AND_EXIT(cusolverDnSetStream(cusolverH, stream));
            CUSOLVER_CHECK_AND_EXIT(cusolverDnCreateParams(&params));

            // d_info
            int* d_info = nullptr;
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
            CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_info, 3, sizeof(int), stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            T* d_A = nullptr; /* device copy of A */
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * a_size));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * a_size, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            [[maybe_unused]] T* d_B = nullptr;
            if constexpr (do_solver) {
                CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(T) * b_size));
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(T) * b_size, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
            }

            int64_t* d_ipiv_64 = nullptr;
            if (is_pivot) {
                CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_ipiv_64), sizeof(int64_t) * mn));
            }

            size_t            workspaceInBytesOnDevice = 0;       /* size of workspace */
            void*             d_work                   = nullptr; /* device workspace for getrf */
            size_t            workspaceInBytesOnHost   = 0;       /* size of workspace */
            std::vector<char> h_work;                             /* host workspace for getrf */

            // query working space
            CUSOLVER_CHECK_AND_EXIT(cusolverDnXgetrf_bufferSize(cusolverH,
                                                                params,
                                                                int64_t(m),
                                                                int64_t(n),
                                                                common::traits<cuda_data_type>::cuda_data_type,
                                                                d_A,
                                                                lda,
                                                                common::traits<cuda_data_type>::cuda_data_type,
                                                                &workspaceInBytesOnDevice,
                                                                &workspaceInBytesOnHost));

            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_work), workspaceInBytesOnDevice));
            if (0 < workspaceInBytesOnHost) {
                h_work.resize(workspaceInBytesOnHost);
            }


            // LU factorization one batch at a time
            for (unsigned int batch = 0; batch < batches; batch++) {
                if (batch > 0) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, &(A[a_size * batch]), sizeof(T) * a_size, cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
                }
                CUSOLVER_CHECK_AND_EXIT(cusolverDnXgetrf(cusolverH,
                                                         params,
                                                         int64_t(m),
                                                         int64_t(n),
                                                         common::traits<cuda_data_type>::cuda_data_type,
                                                         d_A,
                                                         lda,
                                                         d_ipiv_64,
                                                         common::traits<cuda_data_type>::cuda_data_type,
                                                         d_work,
                                                         workspaceInBytesOnDevice,
                                                         h_work.data(),
                                                         workspaceInBytesOnHost,
                                                         d_info));

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&(A[a_size * batch]), d_A, sizeof(T) * a_size, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&info[batch], d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
                if (is_pivot) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&ipiv[batch * mn], d_ipiv_64, sizeof(int64_t) * mn, cudaMemcpyDeviceToHost, stream));
                }

                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

                if (0 > info[batch]) {
                    printf("%d-th parameter is wrong \n", -info[batch]);
                    return false;
                }

                if constexpr (do_solver) {
                    if (batch > 0) {
                        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, &B[b_size * batch], sizeof(T) * b_size, cudaMemcpyHostToDevice, stream));
                        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
                    }
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnXgetrs(cusolverH,
                                                             params,
                                                             trans,
                                                             int64_t(m),
                                                             nrhs,
                                                             common::traits<cuda_data_type>::cuda_data_type,
                                                             d_A,
                                                             lda,
                                                             d_ipiv_64,
                                                             common::traits<cuda_data_type>::cuda_data_type,
                                                             d_B,
                                                             ldb,
                                                             d_info));

                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&B[b_size * batch], d_B, sizeof(T) * b_size, cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
                }
            } // end batch loop

            /* free resources */
            CUDA_CHECK_AND_EXIT(cudaFree(d_A));
            CUDA_CHECK_AND_EXIT(cudaFree(d_info));
            CUDA_CHECK_AND_EXIT(cudaFree(d_work));
            if (is_pivot) {
                CUDA_CHECK_AND_EXIT(cudaFree(d_ipiv_64));
            }
            if constexpr (do_solver) {
                CUDA_CHECK_AND_EXIT(cudaFree(d_B));
            }
            CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroyParams(params));
            CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroy(cusolverH));
        }

        // if row major, transpose the result A
        if (!is_col_major_a) {
            transpose_matrix<T>(A, n, lda, batches); // fast, second fast, batch
        }
        if (!is_col_major_b && do_solver && nrhs > 1) {
            transpose_matrix<T>(B, nrhs, ldb, batches); // fast, second fast, batch
        }

        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
        return true;
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_LU_HPP
