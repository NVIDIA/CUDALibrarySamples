/*
 * Copyright 2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int group_count = 2;
    const int m_array[group_count] = {2, 3};
    const int n_array[group_count] = {2, 3};
    const int k_array[group_count] = {2, 3};
    const int lda_array[group_count] = {2, 3};
    const int ldb_array[group_count] = {2, 3};
    const int ldc_array[group_count] = {2, 3};
    const int group_size[group_count] = {2, 1};

    /*
     * Group 0:
     *   A = | 1.0 | 2.0 | 5.0 | 6.0 |
     *       | 3.0 | 4.0 | 7.0 | 8.0 |
     *
     *   B = | 5.0 | 6.0 |  9.0 | 10.0 |
     *       | 7.0 | 8.0 | 11.0 | 12.0 |
     *
     * Group 1:
     *   A = | 1.0 | 2.0 | 3.0 |
     *       | 4.0 | 5.0 | 6.0 |
     *       | 7.0 | 8.0 | 9.0 |
     *
     *   B = | 4.0  | 5.0  | 6.0  |
     *       | 7.0  | 8.0  | 9.0  |
     *       | 10.0 | 11.0 | 12.0 |
     */

    const std::vector<std::vector<data_type>> A_array = {
        {1.0, 3.0, 2.0, 4.0},
        {5.0, 7.0, 6.0, 8.0},
        {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0}};
    const std::vector<std::vector<data_type>> B_array = {
        {5.0, 7.0, 6.0, 8.0},
        {9.0, 11.0, 10.0, 12.0},
        {4.0, 7.0, 10.0, 5.0, 8.0, 11.0, 6.0, 9.0, 12.0}};

    const data_type NaN = std::numeric_limits<data_type>::quiet_NaN();
    std::vector<std::vector<data_type>> C_array = {
        {NaN, NaN, NaN, NaN},
        {NaN, NaN, NaN, NaN},
        {NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN}};

    const data_type alpha_array[group_count] = {1.0, 1.0};
    const data_type beta_array[group_count] = {0.0, 0.0};

    data_type **d_A_array = nullptr;
    data_type **d_B_array = nullptr;
    data_type **d_C_array = nullptr;

    const int gemm_count = A_array.size();
    std::vector<data_type *> d_A(gemm_count, nullptr);
    std::vector<data_type *> d_B(gemm_count, nullptr);
    std::vector<data_type *> d_C(gemm_count, nullptr);

    cublasOperation_t transa_array[group_count] = {CUBLAS_OP_N, CUBLAS_OP_N};
    cublasOperation_t transb_array[group_count] = {CUBLAS_OP_N, CUBLAS_OP_N};

    int problem_idx = 0;
    for (int i = 0; i < group_count; i++) {
        printf("Group %d:\n", i);
        for (int j = 0; j < group_size[i]; j++) {
            printf("A[%d]\n", j);
            print_matrix(m_array[i], k_array[i], A_array[problem_idx].data(),
                         lda_array[i]);
            printf("=====\n");

            printf("B[%d]\n", j);
            print_matrix(k_array[i], n_array[i], B_array[problem_idx].data(),
                         ldb_array[i]);
            printf("=====\n");

            problem_idx++;
        }
        printf("\n");
    }

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < gemm_count; i++) {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_A[i]), sizeof(data_type) * A_array[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_B[i]), sizeof(data_type) * B_array[i].size()));
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void **>(&d_C[i]), sizeof(data_type) * C_array[i].size()));
    }

    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_A_array), sizeof(data_type *) * gemm_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_B_array), sizeof(data_type *) * gemm_count));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_C_array), sizeof(data_type *) * gemm_count));

    for (int i = 0; i < gemm_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], A_array[i].data(), sizeof(data_type) * A_array[i].size(),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], B_array[i].data(), sizeof(data_type) * B_array[i].size(),
                                   cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(data_type *) * gemm_count,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(data_type *) * gemm_count,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(data_type *) * gemm_count,
                               cudaMemcpyHostToDevice, stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasDgemmGroupedBatched(
        cublasH, transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, d_A_array, lda_array, d_B_array, ldb_array, beta_array,
        d_C_array, ldc_array, group_count, group_size));

    /* step 4: copy data to host */
    for (int i = 0; i < gemm_count; i++) {
        CUDA_CHECK(cudaMemcpyAsync(C_array[i].data(), d_C[i], sizeof(data_type) * C_array[i].size(),
                                   cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     * Group 0:
     *   C = | 19.0 | 22.0 | 111.0 | 122.0 |
     *       | 43.0 | 50.0 | 151.0 | 166.0 |
     *
     * Group 1:
     *   C = |  48.0 |  54.0 |  60.0 |
     *       | 111.0 | 126.0 | 141.0 |
     *       | 174.0 | 198.0 | 222.0 |
     */
    problem_idx = 0;
    for (int i = 0; i < group_count; i++) {
        printf("Group %d:\n", i);
        for (int j = 0; j < group_size[i]; j++) {
            printf("C[%d]\n", j);
            print_matrix(m_array[i], n_array[i], C_array[problem_idx].data(),
                         ldc_array[i]);
            printf("=====\n");

            problem_idx++;
        }
        if (i < group_count - 1) {
            printf("\n");
        }
    }

    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < gemm_count; i++) {
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
