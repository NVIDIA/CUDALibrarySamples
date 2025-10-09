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

/*
 * This sample demonstrates emulated DGEMM in fixed mode.  Fixed mode allows users to specify how many mantissa bits will be
 * retained by the emulation algorithm.
 * 
 * Emulated DGEMM/ZGEMM environment variables:
 *
 *  - CUBLAS_EMULATE_DOUBLE_PRECISION: A value of 1 will allow cuBLAS to utilize FP64 emulation algorithms in double precision
 *                                     and double complex precision routines.
 *
 *  - CUBLAS_EMULATION_STRATEGY: This supports two values: (1) performant -- the default value which enables a layer
 *                               of heuristics to pick between emulation and native algorithms to choose the most
 *                               performant option (2) eager -- a value which will leverage emulation whenever possible.
 *
 *  - CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT: Number of mantissa bits to be used for fixed emulation.  When set,
 *                                                    if an emulation algorithm is used, it will use fixed emulation instead
 *                                                    of dynamic emulation with the number of mantissa bits specified by the user.
 *                                                    If an invalid value is set, the user-provided value will be ignored.
 */
int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    // Compute with 7 INT8 slices
    int mantissaBitCount = 55;

    // Disables the automatic dynamic precision tuning framework
    cudaEmulationMantissaControl_t mControl = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;

    /*
     *   A = | 1.0 | 3.0 |
     *       | 2.0 | 4.0 |
     *
     *   B = | 5.0 | 7.0 |
     *       | 6.0 | 8.0 |
     */
    const std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> B = {5.0, 6.0, 7.0, 8.0};
    std::vector<double> C(m * n);
    const double alpha = 1.0;
    const double beta = 0.0;

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_C = nullptr;

    void* workspace;
    size_t workspaceSizeInBytes;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");

    /* Step 1: Create cuBLAS handle and bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* Step 2: Initialize cuBLAS workspace
     *
     * Fixed-point emulation requires significant workspace memory in order to be performant. 
     * Without user-provided workspace, cuBLAS will fall back to the
     * cudaMallocAsync API for workspace allocation.  This is functional but has
     * a performance overhead that can be significant on some application.
     *
     * If feasible, it is recommended to pre-allocate workspace memory for the cuBLAS
     * handle before running applications.
     */
    workspaceSizeInBytes = getFixedPointWorkspaceSizeInBytes(m, n, k, 1, false, mControl, mantissaBitCount);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&workspace), workspaceSizeInBytes));
    CUBLAS_CHECK(cublasSetWorkspace(cublasH, workspace, workspaceSizeInBytes));

    /* Step 3: Copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* Step 4: Setup emulation configurations */

    // Allow FP64 emulation algorithms to be used
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));

    // Run emulation whenever possible
    CUBLAS_CHECK(cublasSetEmulationStrategy(cublasH, CUBLAS_EMULATION_STRATEGY_EAGER));

    // Choose to run with fixed mantissa control
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(cublasH, mControl));

    // Set the number of mantissa bits to retain
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(cublasH, mantissaBitCount));

    /* Step 5: Compute */
    CUBLAS_CHECK(cublasGemmEx(
                    cublasH, transa, transb, m, n, k, &alpha, d_A, CUDA_R_64F, lda, d_B, CUDA_R_64F,
                    ldb, &beta, d_C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT));

    /* Step 6: Copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(double) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */

    printf("C\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    // Free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
