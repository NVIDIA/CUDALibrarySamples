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

#ifndef CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_GTSV_HPP
#define CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_GTSV_HPP

#include <cusparse.h>

namespace common {
    template<typename T, typename cuda_data_type, bool check_perf = false>
    bool reference_cusolver_gtsv(const std::vector<T>& dl,
                                 const std::vector<T>& d,
                                 const std::vector<T>& du,
                                 std::vector<T>&       B,
                                 const unsigned int    m,
                                 const unsigned int    k,
                                 const unsigned int    padded_batches = 1,
                                 const bool            is_col_major_b = true,
                                 const unsigned int    actual_batches = 0) {

        const unsigned int b_size = B.size() / padded_batches;

        unsigned int batches = (actual_batches == 0) ? padded_batches : actual_batches;

        cudaStream_t stream = nullptr;
        CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        cusparseHandle_t cusparseH = nullptr;
        CUSPARSE_CHECK_AND_EXIT(cusparseCreate(&cusparseH));
        CUSPARSE_CHECK_AND_EXIT(cusparseSetStream(cusparseH, stream));

        [[maybe_unused]] double ms_gtsv_cusolver = 0.0;

        // if row major, transpose the input A
        if (!is_col_major_b) {
            // For row major, transpose the matrix before calling cuSolver
            transpose_matrix<T>(B, m, k, batches);
        }

        //============================================
        // Use cuSparse<t>gtsv2StridedBatch API for both single and multiple batches
        cuda_data_type* d_dl  = nullptr; /* device copy of dl */
        cuda_data_type* d_d   = nullptr; /* device copy of d */
        cuda_data_type* d_du  = nullptr; /* device copy of du */
        cuda_data_type* d_B   = nullptr;
        size_t          lwork = 0;

        // cuSparse API expects dl/d/du having the same batchStride, i.e., the first element of each batch of dl is 0,
        // and the last element of each batch of du is 0 as well
        // Also the cuSpare API only support nrhs == 1
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_dl), sizeof(cuda_data_type) * m * batches));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_d), sizeof(cuda_data_type) * m * batches));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_du), sizeof(cuda_data_type) * m * batches));
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(cuda_data_type) * m * batches));

        // set the unneeded elements to 0
        CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_dl, 0, sizeof(cuda_data_type) * m * batches, stream));
        CUDA_CHECK_AND_EXIT(cudaMemsetAsync(d_du, 0, sizeof(cuda_data_type) * m * batches, stream));

        // for B, only copy the first right hand side
        for (int b = 0; b < batches; b++) {
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_dl + 1 + b * m, dl.data() + b * (m - 1), sizeof(cuda_data_type) * (m - 1), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_du + b * m, du.data() + b * (m - 1), sizeof(cuda_data_type) * (m - 1), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B + b * m, B.data() + b * m * k, sizeof(cuda_data_type) * m, cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_d, d.data(), sizeof(cuda_data_type) * m * batches, cudaMemcpyHostToDevice, stream));


        // Query workspace size for syevjBatched
        constexpr bool is_complex = common::is_complex<T>();
        constexpr bool is_float   = std::is_same_v<typename common::get_precision<T>::type, float>;

        if constexpr (is_float && !is_complex) {
            CUSPARSE_CHECK_AND_EXIT(cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseH, m, d_dl, d_d, d_du, d_B, batches, m /* batchStride */, &lwork));
        } else if constexpr (is_float && is_complex) {
            CUSPARSE_CHECK_AND_EXIT(cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseH, m, d_dl, d_d, d_du, d_B, batches, m /* batchStride */, &lwork));
        } else if constexpr (!is_float && !is_complex) {
            CUSPARSE_CHECK_AND_EXIT(cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseH, m, d_dl, d_d, d_du, d_B, batches, m /* batchStride */, &lwork));
        } else if constexpr (!is_float && is_complex) {
            CUSPARSE_CHECK_AND_EXIT(cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseH, m, d_dl, d_d, d_du, d_B, batches, m /* batchStride */, &lwork));
        }

        // Allocate workspace
        void* d_work = nullptr;
        CUDA_CHECK_AND_EXIT(cudaMalloc(&d_work, lwork));

        for (int nrhs = 0; nrhs < k; nrhs++) {
            // copy B to d_B
            for (int b = 0; b < batches; b++) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B + b * m, B.data() + b * m * k + nrhs * m, sizeof(cuda_data_type) * m, cudaMemcpyHostToDevice, stream));
            }

            // Execute batched gtsv
            auto execute_gtsv_api = [&](cudaStream_t str) {
                constexpr bool is_complex = common::is_complex<T>();
                constexpr bool is_float   = std::is_same_v<typename common::get_precision<T>::type, float>;
                if constexpr (is_float && !is_complex) {
                    CUSPARSE_CHECK_AND_EXIT(cusparseSgtsv2StridedBatch(cusparseH, m, d_dl, d_d, d_du, d_B, batches, m /* batchStride */, d_work));
                } else if constexpr (is_float && is_complex) {
                    CUSPARSE_CHECK_AND_EXIT(cusparseCgtsv2StridedBatch(cusparseH, m, d_dl, d_d, d_du, d_B, batches, m /* batchStride */, d_work));
                } else if constexpr (!is_float && !is_complex) {
                    CUSPARSE_CHECK_AND_EXIT(cusparseDgtsv2StridedBatch(cusparseH, m, d_dl, d_d, d_du, d_B, batches, m /* batchStride */, d_work));
                } else if constexpr (!is_float && is_complex) {
                    CUSPARSE_CHECK_AND_EXIT(cusparseZgtsv2StridedBatch(cusparseH, m, d_dl, d_d, d_du, d_B, batches, m /* batchStride */, d_work));
                }
                CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
            };


            if constexpr (!check_perf) {
                execute_gtsv_api(stream); // execute gtsvBatched
            } else {
                if (nrhs == 0) { // check perf only for the first right hand side
                    const unsigned int warmup_repeats = 1;
                    const unsigned int repeats        = 1;
                    ms_gtsv_cusolver                  = common::measure::execution(execute_gtsv_api, warmup_repeats, repeats, stream) / repeats;
                }
            }

            // Copy results back
            for (int b = 0; b < batches; b++) {
                CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(B.data() + b * m * k + nrhs * m, d_B + b * m, sizeof(cuda_data_type) * m, cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
        }

        // Cleanup
        CUDA_CHECK_AND_EXIT(cudaFree(d_dl));
        CUDA_CHECK_AND_EXIT(cudaFree(d_d));
        CUDA_CHECK_AND_EXIT(cudaFree(d_du));
        CUDA_CHECK_AND_EXIT(cudaFree(d_B));
        CUDA_CHECK_AND_EXIT(cudaFree(d_work));

        CUSPARSE_CHECK_AND_EXIT(cusparseDestroy(cusparseH));
        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

        if constexpr (check_perf) {
            double seconds_per_giga_batch = ms_gtsv_cusolver / 1e3 / batches * 1e9;
            double gb_s = (3 * m - 2 + m * 1 * 2) * sizeof(T) / seconds_per_giga_batch; // A read and B read/write, Note cusparase API only supports nrhs=1                           // A read, half write, and lambda write
            common::print_perf("Ref_cuSparse<t>gtsv2StrideBatch", batches, m, m, 1, 0, gb_s, ms_gtsv_cusolver, 0); // dummy 0 for nrhs, gflops, and blockDim
        }

        // if row major, transpose the result B back
        if (!is_col_major_b) {
            transpose_matrix<T>(B, k, m, batches);
        }

        return true;
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_CUSOLVERDX_COMMON_CUSOLVER_REFERENCE_HEEV_HPP
