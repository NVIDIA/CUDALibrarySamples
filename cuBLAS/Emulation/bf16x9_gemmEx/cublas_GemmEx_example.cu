/*
 * Copyright 2025 NVIDIA Corporation.  All rights reserved.
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
    CUBLAS_CHECK(cublasGemmEx(
        cublasH, transa, transb, m, n, k, &alpha, d_A, CUDA_R_32F, lda, d_B, CUDA_R_32F, ldb,
        &beta, d_Cnative, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

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

    /* 
     * For building confidence in numerics, we can use the eager strategy to leverage an emulated fp32 kernel when possible
     * even if it doesn't make performant sense to do so, like in the case of this very small problem
     */
    CUBLAS_CHECK(
        cublasSetEmulationStrategy(cublasH, CUBLAS_EMULATION_STRATEGY_EAGER));

    CUBLAS_CHECK(cublasGemmEx(
        cublasH, transa, transb, m, n, k, &alpha, d_A, CUDA_R_32F, lda, d_B, CUDA_R_32F, ldb,
        &beta, d_Cemulated, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT));

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
