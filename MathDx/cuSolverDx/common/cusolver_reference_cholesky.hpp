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

#ifndef CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_CHOLESKY_HPP
#define CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_CHOLESKY_HPP

namespace common {
    template<typename T, typename cuda_data_type, bool do_solver = false, bool check_factor_perf = false, bool check_solve_perf = false>
    bool reference_cusolver_cholesky(std::vector<T>&    A,
                                     std::vector<T>&    B,
                                     int*               info,
                                     const unsigned int m,
                                     const unsigned int nrhs           = 1,
                                     const unsigned int padded_batches = 1,
                                     bool               is_lower_fill  = true,
                                     bool               is_col_major_a = true,
                                     bool               is_col_major_b = true,
                                     const unsigned int actual_batches = 0) {

        const unsigned int a_size = A.size() / padded_batches;
        const unsigned int lda    = a_size / m;

        [[maybe_unused]] const unsigned int b_size = B.size() / padded_batches;
        [[maybe_unused]] const unsigned int ldb    = b_size / nrhs; // ldb if b is column major

        const unsigned batches = (actual_batches == 0) ? padded_batches : actual_batches;

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        cusolverDnHandle_t cusolverH = NULL;
        CUSOLVER_CHECK_AND_EXIT(cusolverDnCreate(&cusolverH));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnSetStream(cusolverH, stream));
        cublasFillMode_t uplo;
        uplo = (is_lower_fill) ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

        // if row major, transpose the input A
        if (!is_col_major_a) {
            transpose_matrix<T>(A, lda, m, batches); // fast, second fast, batch
            // printf("after transpose A = \n");
            // common::print_matrix(m, m * batches, A.data(), lda);
        }
        if (!is_col_major_b && do_solver && nrhs > 1) {
            transpose_matrix<T>(B, ldb, nrhs, batches); // fast, second fast, batch
        }

        // d_info
        int* d_info = nullptr;
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * batches));
        CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_info, 3, sizeof(int) * batches, stream));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

        //============================================
        // Use cuSolver batched cholesky API, which only supports nrhs=1
        if (batches > 1 && nrhs == 1) {

            std::vector<cuda_data_type*> Aarray(batches, nullptr);
            cuda_data_type**             d_Aarray = nullptr;

            [[maybe_unused]] std::vector<cuda_data_type*> Barray(batches, nullptr);
            [[maybe_unused]] cuda_data_type**             d_Barray = nullptr;

            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&Aarray[0]), sizeof(T) * a_size * batches));
            for (auto j = 1; j < batches; j++) {
                Aarray[j] = Aarray[j - 1] + a_size;
            }
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(Aarray[0], A.data(), sizeof(T) * a_size * batches, cudaMemcpyHostToDevice, stream));

            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_Aarray), sizeof(T*) * batches));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_Aarray, Aarray.data(), sizeof(T*) * batches, cudaMemcpyHostToDevice, stream));

            if constexpr (do_solver) {
                CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&Barray[0]), sizeof(T) * b_size * batches));
                for (auto j = 1; j < batches; j++) {
                    Barray[j] = Barray[j - 1] + b_size;
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(Barray[0], B.data(), sizeof(T) * b_size * batches, cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_Barray), sizeof(T*) * batches));
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_Barray, Barray.data(), sizeof(T*) * batches, cudaMemcpyHostToDevice, stream));
                }
            }

            auto execute_potrf_api = [&](cudaStream_t str) {
                constexpr bool is_complex = common::is_complex<T>();
                constexpr bool is_float   = std::is_same_v<typename common::get_precision<T>::type, float>;
                if constexpr (is_float && !is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnSpotrfBatched(cusolverH, uplo, m, d_Aarray, lda, d_info, batches));
                } else if constexpr (is_float && is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnCpotrfBatched(cusolverH, uplo, m, d_Aarray, lda, d_info, batches));
                } else if constexpr (!is_float && !is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnDpotrfBatched(cusolverH, uplo, m, d_Aarray, lda, d_info, batches));
                } else if constexpr (!is_float && is_complex) {
                    CUSOLVER_CHECK_AND_EXIT(cusolverDnZpotrfBatched(cusolverH, uplo, m, d_Aarray, lda, d_info, batches));
                }
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
            };
            [[maybe_unused]] auto execute_reset_a = [&](cudaStream_t str) { CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(Aarray[0], A.data(), sizeof(T) * a_size * batches, cudaMemcpyHostToDevice, str)); };

            [[maybe_unused]] double ms_factor, ms_solve;

            if constexpr (!check_factor_perf) {
                execute_potrf_api(stream);
            } else {
                // measure and report cusolver API performance
                // cuSolver API forces in-place A and B, so no warmup and repeat 1, specially for complex data
                // otherwise factorization returns non-zero d_info
                const unsigned int warmup_repeats = 1;
                const unsigned int repeats        = 1;

                ms_factor = common::measure::execution(execute_potrf_api, execute_reset_a, warmup_repeats, repeats, stream) / repeats;
            }

            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(Aarray.data(), d_Aarray, sizeof(T*) * Aarray.size(), cudaMemcpyDeviceToHost, stream));

            // cuSolver's batched solver uses the off-diagonal part as workspace.  So, we only copy the desired triangle.
            for (int k = 0; k < batches; k++) {
                T* A_k = A.data() + k * a_size;
                if (is_lower_fill) {
                    // Lower fill and column major, or upper fill and row major
                    for (int j = 0; j < m; ++j) {
                        int    offset = j + j * lda;
                        size_t size   = sizeof(T) * (m - j);
                        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A_k + offset, Aarray[k] + offset, size, cudaMemcpyDeviceToHost, stream));
                    }
                } else {
                    // Upper fill and column major, or lower fill and row major
                    for (int j = 0; j < m; ++j) {
                        int    offset = j * lda;
                        size_t size   = sizeof(T) * (j + 1);
                        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A_k + offset, Aarray[k] + offset, size, cudaMemcpyDeviceToHost, stream));
                    }
                }
            }

            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info, d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            for (auto i = 0; i < batches; i++) {
                if (*(info + i) != 0) {
                    std::cout << "non-zero d_info returned with cusolverDnXpotrfBatched for batch #" << i << std::endl;
                    CUDA_CHECK_AND_EXIT(cudaFree(d_Aarray));
                    return false;
                }
            }

            if constexpr (do_solver) {
                auto execute_potrs_api = [&](cudaStream_t str) {
                    constexpr bool is_complex = common::is_complex<T>();
                    constexpr bool is_float   = std::is_same_v<typename common::get_precision<T>::type, float>;
                    if constexpr (is_float && !is_complex) {
                        CUSOLVER_CHECK_AND_EXIT(cusolverDnSpotrsBatched(cusolverH, uplo, m, nrhs, d_Aarray, lda, d_Barray, ldb, d_info, batches));
                    } else if constexpr (is_float && is_complex) {
                        CUSOLVER_CHECK_AND_EXIT(cusolverDnCpotrsBatched(cusolverH, uplo, m, nrhs, d_Aarray, lda, d_Barray, ldb, d_info, batches));
                    } else if constexpr (!is_float && !is_complex) {
                        CUSOLVER_CHECK_AND_EXIT(cusolverDnDpotrsBatched(cusolverH, uplo, m, nrhs, d_Aarray, lda, d_Barray, ldb, d_info, batches));
                    } else if constexpr (!is_float && is_complex) {
                        CUSOLVER_CHECK_AND_EXIT(cusolverDnZpotrsBatched(cusolverH, uplo, m, nrhs, d_Aarray, lda, d_Barray, ldb, d_info, batches));
                    }
                    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
                };
                [[maybe_unused]] auto execute_reset_b = [&](cudaStream_t str) { CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(Barray[0], B.data(), sizeof(T) * b_size * batches, cudaMemcpyHostToDevice, str)); };

                if constexpr (!check_solve_perf) {
                    execute_potrs_api(stream);
                } else { // measure and report cusolver API performance
                    const unsigned int warmup_repeats = 1;
                    const unsigned int repeats        = 1;

                    ms_solve = common::measure::execution(execute_potrs_api, execute_reset_b, warmup_repeats, repeats, stream) / repeats;
                }

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(Barray.data(), d_Barray, sizeof(T*) * Barray.size(), cudaMemcpyDeviceToHost, stream));

                for (int j = 0; j < batches; j++) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&(B[j * b_size]), Barray[j], sizeof(T) * b_size, cudaMemcpyDeviceToHost, stream));
                }

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info, d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

                for (auto i = 0; i < batches; i++) {
                    if (*(info + i) != 0) {
                        std::cout << "non-zero d_info returned with cusolverDnXpotrsBatched for batch #" << i << std::endl;
                        CUDA_CHECK_AND_EXIT(cudaFree(d_Barray));
                        return false;
                    }
                }
                CUDA_CHECK_AND_EXIT(cudaFree(Barray[0]));
                CUDA_CHECK_AND_EXIT(cudaFree(d_Barray));
            }

            // if performance check, report the timing
            if constexpr (check_factor_perf && check_solve_perf) {
                double seconds_per_giga_batch = (ms_factor + ms_solve) / 1e3 / batches * 1e9;
                double gb_s                   = (a_size + b_size) * 2 * sizeof(T) / seconds_per_giga_batch; // A and B both read and write
                double gflops                 = (common::get_flops_potrs<T>(m, nrhs) + common::get_flops_potrf<T>(m)) / seconds_per_giga_batch;
                common::print_perf("Ref_cuSolverDnXpotrf+XportsBatched", batches, m, m, nrhs, gflops, gb_s, ms_factor + ms_solve, 0); // dummy 0 for blockDim
            } else if constexpr (check_factor_perf && !check_solve_perf) {
                double seconds_per_giga_batch = ms_factor / 1e3 / batches * 1e9;
                double gb_s                   = a_size * 2 * sizeof(T) / seconds_per_giga_batch; // A read and write
                double gflops                 = common::get_flops_potrf<T>(m) / seconds_per_giga_batch;
                common::print_perf("Ref_cuSolverDnXpotrfBatched", batches, m, m, nrhs, gflops, gb_s, ms_factor, 0); // dummy 0 for blockDim
            } else if constexpr (!check_factor_perf && check_solve_perf) {
                double seconds_per_giga_batch = ms_solve / 1e3 / batches * 1e9;
                double gb_s                   = (a_size + b_size * 2) * sizeof(T) / seconds_per_giga_batch; // A read only,B read and write
                double gflops                 = common::get_flops_potrs<T>(m, nrhs) / seconds_per_giga_batch;
                common::print_perf("Ref_cuSolverDnXpotrsBatched", batches, m, m, nrhs, gflops, gb_s, ms_solve, 0); // dummy 0 for blockDim
            }


            /* free resources */
            CUDA_CHECK_AND_EXIT(cudaFree(Aarray[0]));
            CUDA_CHECK_AND_EXIT(cudaFree(d_Aarray));

            //============================================
            // Use cuSolver non-batched API, doing one batch at a time
        } else {
            cusolverDnParams_t params = nullptr;

            T* d_A = nullptr;
            CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * a_size));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, &A[0], sizeof(T) * a_size, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

            [[maybe_unused]] T* d_B = nullptr;
            if constexpr (do_solver) {
                CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(T) * b_size));
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, &B[0], sizeof(T) * b_size, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
            }

            size_t            workspaceInBytesOnDevice = 0;       /* size of workspace */
            void*             d_work                   = nullptr; /* device workspace */
            size_t            workspaceInBytesOnHost   = 0;       /* size of workspace */
            std::vector<char> h_work;                             /* host workspace */

            CUSOLVER_CHECK_AND_EXIT(cusolverDnCreateParams(&params));

            // query working space
            CUSOLVER_CHECK_AND_EXIT(cusolverDnXpotrf_bufferSize(cusolverH,
                                                                params,
                                                                uplo,
                                                                m,
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

            // Cholesky factorization one batch at a time
            for (unsigned int batch = 0; batch < batches; batch++) {
                if (batch > 0) {
                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, &(A[a_size * batch]), sizeof(T) * a_size, cudaMemcpyHostToDevice, stream));
                    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
                }
                CUSOLVER_CHECK_AND_EXIT(cusolverDnXpotrf(cusolverH,
                                                         params,
                                                         uplo,
                                                         m,
                                                         common::traits<cuda_data_type>::cuda_data_type,
                                                         d_A,
                                                         lda,
                                                         common::traits<cuda_data_type>::cuda_data_type,
                                                         d_work,
                                                         workspaceInBytesOnDevice,
                                                         h_work.data(),
                                                         workspaceInBytesOnHost,
                                                         d_info));

                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&(A[a_size * batch]), d_A, sizeof(T) * a_size, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&info[batch], d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

                if (0 != info[batch]) {
                    std::printf("%d-th parameter is wrong \n", -info[0]);
                    return false;
                }

                if constexpr (do_solver) {
                    if (batch > 0) {
                        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, &B[b_size * batch], sizeof(T) * b_size, cudaMemcpyHostToDevice, stream));
                        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
                    }
                    // cholesky solver
                    CUSOLVER_CHECK_AND_EXIT(
                        cusolverDnXpotrs(cusolverH, params, uplo, m, nrhs, common::traits<cuda_data_type>::cuda_data_type, d_A, lda, common::traits<cuda_data_type>::cuda_data_type, d_B, ldb, d_info));

                    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&B[b_size * batch], d_B, sizeof(T) * b_size, cudaMemcpyDeviceToHost, stream));
                    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
                }
            } // end of batches loop

            /* free resources */
            CUDA_CHECK_AND_EXIT(cudaFree(d_A));
            CUDA_CHECK_AND_EXIT(cudaFree(d_work));
            CUDA_CHECK_AND_EXIT(cudaFree(d_B));
        }

        // if row major, transpose the result A
        if (!is_col_major_a) {
            transpose_matrix<T>(A, m, lda, batches); // fast, second fast, flow
            // printf("after transpose A = \n");
            // common::print_matrix(m, m * batches, A.data(), lda);
        }
        if (!is_col_major_b && do_solver && nrhs > 1) {
            transpose_matrix<T>(B, nrhs, ldb, batches); // fast, second fast, batch
        }

        CUDA_CHECK_AND_EXIT(cudaFree(d_info));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroy(cusolverH));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
        return true;
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_CHOLESKY_HPP