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

/*
 * This sample demonstrates emulated ZGEMM using dynamic mantissa control. Dynamic mantissa control leverages an 
 * automatic dynamic precision framework to determine how much precision should be retained to have the 
 * same or better accuracy than native FP64.
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

    // Allows for up to 10 INT8 slices
    int maxMantissaBitCount = 79;

    // Allows the automatic dynamic precision tuning framework
    cudaEmulationMantissaControl_t mControl = CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;

    /*
     *   A = | 1.0 + 1.0j | 3.0 - 3.0j |
     *       | 2.0 - 2.0j | 4.0 + 4.0j |
     *
     *   B = | 5.0 + 5.0j | 7.0 - 7.0j |
     *       | 6.0 - 6.0j | 8.0 + 8.0j |
     */
    const std::vector<cuDoubleComplex> A = {
        {1.0, 1.0}, {2.0, -2.0}, {3.0, -3.0}, {4.0, 4.0}};
    const std::vector<cuDoubleComplex> B = {
        {5.0, 5.0}, {6.0, -6.0}, {7.0, -7.0}, {8.0, 8.0}};
    std::vector<cuDoubleComplex> C(m * n);
    const cuDoubleComplex alpha = {1.0, -1.0};
    const cuDoubleComplex beta = {0.0, 0.0};

    cuDoubleComplex *d_A = nullptr;
    cuDoubleComplex *d_B = nullptr;
    cuDoubleComplex *d_C = nullptr;

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
    workspaceSizeInBytes = getApproximateFixedPointEmulationWorkspaceSize(
            CUDA_C_64F, m, n, k, 1, maxMantissaBitCount, mControl);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&workspace), workspaceSizeInBytes));
    CUBLAS_CHECK(cublasSetWorkspace(cublasH, workspace, workspaceSizeInBytes));


    /* Step 3: Copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(cuDoubleComplex) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(cuDoubleComplex) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(cuDoubleComplex) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(cuDoubleComplex) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(cuDoubleComplex) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* Step 4: Setup emulation configurations */

    // Allow FP64 emulation algorithms to be used
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));

    // Run emulation whenever possible
    CUBLAS_CHECK(cublasSetEmulationStrategy(cublasH, CUBLAS_EMULATION_STRATEGY_EAGER));

    // Optionally specify dynamic mantissa control (dynamic is the default value)  
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(cublasH, mControl));

    // Optionaly choose a maximum value before falling back to native FP64.  A default value of 0 allows cuBLAS to
    // choose an optimal value for the current GPU
    CUBLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(cublasH, maxMantissaBitCount));

    // Optionaly reduce or increase precision by adding an offset to the output of the automated precision tuning algorithm.
    // In this case, we leverage 8 fewer bits than recommended, which can tradeoff accuracy for improved performance.
    CUBLAS_CHECK(cublasSetFixedPointEmulationMantissaBitOffset(cublasH, -8));

    /* Step 5: Compute */
    CUBLAS_CHECK(
        cublasZgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    /* Step 6: Copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(cuDoubleComplex) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   C = | -26.0 - 26.0i | 62.0 - 62.0i |
     *       |  68.0 - 68.0i | 36.0 + 36.0i |
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