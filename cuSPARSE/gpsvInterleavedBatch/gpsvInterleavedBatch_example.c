/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
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
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: %d\n",                \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

 // compute: |b - A*x|_inf
void residual_eval(int    n,
                   float* h_S,
                   float* h_L,
                   float* h_M,
                   float* h_U,
                   float* h_W,
                   float* h_b,
                   float* h_X,
                   float* r_nrminf_ptr) {
    float r_nrminf = 0;
    for (int i = 0; i < n; i++) {
        float dot = 0;
        if (i > 1)
            dot += h_S[i] * h_X[i - 2];
        if (i > 0)
            dot += h_L[i] * h_X[i - 1];
        dot += h_M[i] * h_X[i];
        if (i < (n - 1))
            dot += h_U[i] * h_X[i + 1];
        if (i < (n - 2))
            dot += h_W[i] * h_X[i + 2];
        float ri = h_b[i] - dot;
        r_nrminf = (r_nrminf > fabs(ri)) ? r_nrminf : fabs(ri);
    }
    *r_nrminf_ptr = r_nrminf;
}

int main(void) {
    int n         = 4;
    int batchSize = 2;
    int full_size = n * batchSize;
    //
    //     |  1    8   13   0  |       | 1 |       | -0.0592 |
    // A1 =|  5    2    9  14  |, b1 = | 2 |, x1 = |  0.3428 |
    //     | 11    6    3  10  |       | 3 |       | -0.1295 |
    //     |  0   12    7   4  |       | 4 |       |  0.1982 |
    //
    //     | 15   22   27   0  |       | 5 |       | -0.0012 |
    // A2 =| 19   16   23  28  |, b2 = | 6 |, x2 = |  0.2792 |
    //     | 25   20   17  24  |       | 7 |       | -0.0416 |
    //     |  0   26   21  18  |       | 8 |       |  0.0898 |
    //
    // A = (h_S, h_L, h_M, h_U, h_W), h_B and h_X are in aggregate format
    float h_S[] = {0, 0, 11, 12, 0, 0, 25, 26};
    float h_L[] = {0, 5, 6, 7, 0, 19, 20, 21};
    float h_M[] = {1, 2, 3, 4, 15, 16, 17, 18};
    float h_U[] = {8, 9, 10, 0, 22, 23, 24, 0};
    float h_W[] = {13, 14, 0, 0, 27, 28, 0, 0};
    float h_B[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_X[] = {0, 0, 0, 0, 0, 0, 0, 0};
    //--------------------------------------------------------------------------
    // step 1: allocate device memory
    float *d_S0, *d_L0, *d_M0, *d_U0, *d_W0;
    float  *d_S,  *d_L,  *d_M, *d_U,   *d_W;
    float  *d_B,  *d_X;
    // device memory
    //   (d_S0, d_L0, d_M0, d_U0, d_W0) is aggregate format
    //   (d_S, d_L, d_M, d_U, d_W) is interleaved format
    CHECK_CUDA( cudaMalloc((void**) &d_S0, full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_L0, full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_M0, full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_U0, full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_W0, full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_S,  full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_L,  full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_M,  full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_U,  full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_W,  full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_B,  full_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &d_X,  full_size * sizeof(float)) )
    //--------------------------------------------------------------------------
    // step 2: copy data to device
    CHECK_CUDA( cudaMemcpy(d_S0, h_S, full_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_L0, h_L, full_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_M0, h_M, full_size * sizeof(float),
                           cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(d_U0, h_U, full_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_W0, h_W, full_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_B, h_B, full_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // step 3: create cuSPARSE and cuBLAS handles
    cusparseHandle_t cusparseHandle = NULL;
    cublasHandle_t   cublasHandle   = NULL;
    CHECK_CUSPARSE( cusparseCreate(&cusparseHandle) )
    CHECK_CUBLAS(   cublasCreate(&cublasHandle) )
    //--------------------------------------------------------------------------
    // step 4: prepare data in device, interleaved format
    float h_one = 1;
    float h_zero = 0;
    // convert h_S to interleaved format h_S = transpose(ds0)
    CHECK_CUBLAS( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                              batchSize,        // number of rows of h_S
                              n,                // number of columns of h_S
                              &h_one, d_S0,     // ds0 is n-by-batchSize
                              n,                // leading dimension of ds0
                              &h_zero, NULL, n, // don't care
                              d_S,              // h_S is batchSize-by-n
                              batchSize) )      // leading dimension of h_S

    // convert h_L to interleaved format  h_L = transpose(dl0)
    CHECK_CUBLAS( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                              batchSize,        // number of rows of h_L
                              n,                // number of columns of h_L
                              &h_one, d_L0,     // dl0 is n-by-batchSize
                              n,                // leading dimension of dl0
                              &h_zero, NULL, n, // don't care
                              d_L,              // h_L is batchSize-by-n
                              batchSize) )      // leading dimension of h_L

    // convert h_M to interleaved format h_M = transpose(d0)
    CHECK_CUBLAS( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                              batchSize,        // number of rows of h_M
                              n,                // number of columns of h_M
                              &h_one, d_M0,     // d0 is n-by-batchSize
                              n,                // leading dimension of d0
                              &h_zero, NULL, n, // don't care
                              d_M,              // h_M is batchSize-by-n
                              batchSize) )      // leading dimension of h_M

    // convert h_U to interleaved format h_U = transpose(du0)
    CHECK_CUBLAS( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                              batchSize,        // number of rows of h_U
                              n,                // number of columns of h_U
                              &h_one, d_U0,     // du0 is n-by-batchSize
                              n,                // leading dimension of du0
                              &h_zero, NULL, n, // don't care
                              d_U,              // h_U is batchSize-by-n
                              batchSize) )      // leading dimension of h_U

    // convert h_W to interleaved format h_W = transpose(dw0)
    CHECK_CUBLAS( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                              batchSize,        // number of rows of h_W
                              n,                // number of columns of h_W
                              &h_one, d_W0,     // dw0 is n-by-batchSize
                              n,                // leading dimension of dw0
                              &h_zero, NULL, n, // don't care
                              d_W,              // h_W is batchSize-by-n
                              batchSize) )      // leading dimension of h_W

    // convert h_B to interleaved format h_X = transpose(h_B)
    CHECK_CUBLAS( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                              batchSize,        // number of rows of h_X
                              n,                // number of columns of h_X
                              &h_one, d_B,      // h_B is n-by-batchSize
                              n,                // leading dimension of h_B
                              &h_zero, NULL, n, // don't care
                              d_X,              // h_X is batchSize-by-n
                              batchSize) )      // leading dimension of h_X
    //--------------------------------------------------------------------------
    // step 5: prepare workspace
    size_t bufferSize;
    void*  d_buffer;
    int    algo = 0; // QR factorization
    CHECK_CUSPARSE( cusparseSgpsvInterleavedBatch_bufferSizeExt(
                        cusparseHandle, algo, n, d_S, d_L, d_M, d_U, d_W, d_X,
                        batchSize, &bufferSize) )

    printf("bufferSize = %lld\n", (long long) bufferSize);

    CHECK_CUDA( cudaMalloc((void**) &d_buffer, bufferSize) )
    //--------------------------------------------------------------------------
    // step 6: solve Aj*xj = bj
    CHECK_CUSPARSE( cusparseSgpsvInterleavedBatch(
                        cusparseHandle, algo, n, d_S, d_L, d_M, d_U, d_W, d_X,
                        batchSize, d_buffer) )
    //--------------------------------------------------------------------------
    // step 7: convert h_X back to aggregate format
    // h_B = transpose(h_X)
    CHECK_CUBLAS( cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                              n,                // number of rows of h_B
                              batchSize,        // number of columns of h_B
                              &h_one, d_X,      // h_X is batchSize-by-n
                              batchSize,        // leading dimension of h_X
                              &h_zero, NULL, n, // don't cae
                              d_B,              // h_B is n-by-batchSize
                              n));              // leading dimension of h_B

    CHECK_CUSPARSE( cusparseDestroy(cusparseHandle) )
    CHECK_CUBLAS( cublasDestroy(cublasHandle) )
    //--------------------------------------------------------------------------
    // step 8: residual evaluation
    CHECK_CUDA( cudaMemcpy(h_X, d_B, full_size * sizeof(float),
                           cudaMemcpyDeviceToHost));
    //--------------------------------------------------------------------------
    // step 9: Check results
    printf("==== x1 = inv(A1)*b1 \n");
    for (int j = 0; j < n; j++)
        printf("x1[%d] = %f\n", j, h_X[j]);

    float r1_nrminf;
    residual_eval(n, h_S, h_L, h_M, h_U, h_W, h_B, h_X, &r1_nrminf);
    printf("|b1 - A1 * x1| = %E\n", r1_nrminf);

    printf("\n==== x2 = inv(A2)*b2 \n");
    for (int j = 0; j < n; j++)
        printf("x2[%d] = %f\n", j, h_X[n + j]);

    float r2_nrminf;
    residual_eval(n,
                  h_S + n,
                  h_L + n,
                  h_M + n,
                  h_U + n,
                  h_W + n,
                  h_B + n,
                  h_X + n,
                  &r2_nrminf);
    printf("|b2 - A2*x2| = %E\n", r2_nrminf);
    //--------------------------------------------------------------------------
    // step 10: free resources
    CHECK_CUDA( cudaFree(d_S0) )
    CHECK_CUDA( cudaFree(d_L0) )
    CHECK_CUDA( cudaFree(d_M0) )
    CHECK_CUDA( cudaFree(d_U0) )
    CHECK_CUDA( cudaFree(d_W0) )
    CHECK_CUDA( cudaFree(d_S) )
    CHECK_CUDA( cudaFree(d_L) )
    CHECK_CUDA( cudaFree(d_M) )
    CHECK_CUDA( cudaFree(d_U) )
    CHECK_CUDA( cudaFree(d_W) )
    CHECK_CUDA( cudaFree(d_B) )
    CHECK_CUDA( cudaFree(d_X) )
    return EXIT_SUCCESS;
}
