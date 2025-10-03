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


#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 4;
    const int n = 4;
    const int k = 4;
    const int lda = 4;
    const int ldb = 4;
    const int ldc = 4;
    /*
     *   A = |  1.0 |  5.0 |  9.0 | 13.0 |
     *       |  2.0 |  6.0 | 10.0 | 14.0 |
     *       |  3.0 |  7.0 | 11.0 | 15.0 |
     *       |  4.0 |  8.0 | 12.0 | 16.0 |
     *
     *   B = |  1.0 |  2.0 |  3.0 |  4.0 |
     *       |  5.0 |  6.0 |  7.0 |  8.0 |
     *       |  9.0 | 10.0 | 11.0 | 12.0 |
     *       | 13.0 | 14.0 | 15.0 | 16.0 |
     */

    const std::vector<float> A = { 1.0f,  2.0f,  3.0f,  4.0f,
                                   5.0f,  6.0f,  7.0f,  8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f,
                                  13.0f, 14.0f, 15.0f, 16.0f}; 
    const std::vector<float> B = { 1.0f,  5.0f,  9.0f, 13.0f,
                                   2.0f,  6.0f, 10.0f, 14.0f,
                                   3.0f,  7.0f, 11.0f, 15.0f,
                                   4.0f,  8.0f, 12.0f, 16.0f}; 
    std::vector<float> C(m * n);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_Cnative = nullptr;
    float *d_Cemulated = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Cnative), sizeof(float) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Cemulated), sizeof(float) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute in native FP32 */
    CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_Cnative, ldc));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_Cnative, sizeof(float) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 276.0 | 304.0 | 332.0 | 360.0 |
     *       | 304.0 | 336.0 | 368.0 | 400.0 |
     *       | 332.0 | 368.0 | 404.0 | 440.0 |
     *       | 360.0 | 400.0 | 440.0 | 480.0 |
     */

    printf("C (fp32)\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    /* step 5: compute in emulated FP32 */
    CUBLAS_CHECK(
        cublasSetMathMode(cublasH, CUBLAS_FP32_EMULATED_BF16X9_MATH));

    /* 
     * For building confidence in numerics, we can use the eager strategy to leverage an emulated fp32 kernel when possible
     * even if it doesn't make performant sense to do so, like in the case of this very small problem
     */
    CUBLAS_CHECK(
        cublasSetEmulationStrategy(cublasH, CUBLAS_EMULATION_STRATEGY_EAGER));

    CUBLAS_CHECK(
        cublasSgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_Cemulated, ldc));

    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_Cemulated, sizeof(float) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("C (bf16x9)\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Cnative));
    CUDA_CHECK(cudaFree(d_Cemulated));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}