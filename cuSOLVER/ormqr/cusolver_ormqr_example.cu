/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream{};

    const int m = 3;
    const int lda = m;
    const int ldb = m;
    const int nrhs = 1; // number of right hand side vectors

    /*       | 1 2 3 |
     *   A = | 4 5 6 |
     *       | 2 1 1 |
     *
     *   x = (1 1 1)'
     *   b = (6 15 4)'
     */

    const std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
    // std::vector<double> X = {1.0, 1.0, 1.0}; // exact solution
    const std::vector<double> B = {6.0, 15.0, 4.0};
    std::vector<double> XC(ldb * nrhs, 0); // solution matrix from GPU

    /* device memory */
    double *d_A = nullptr;
    double *d_tau = nullptr;
    double *d_B = nullptr;
    int *d_info = nullptr;
    double *d_work = nullptr;

    int lwork_geqrf = 0;
    int lwork_ormqr = 0;
    int lwork = 0;
    int info = 0;

    const double one = 1;

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, m, A.data(), lda);
    std::printf("=====\n");
    std::printf("B = (matlab base-1)\n");
    print_matrix(m, nrhs, B.data(), ldb);
    std::printf("=====\n");

    /* step 1: create cudense/cublas handle */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A and B to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of geqrf and ormqr */
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork_geqrf));

    CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m,
                                               d_A, lda, d_tau, d_B, ldb, &lwork_ormqr));

    lwork = std::max(lwork_geqrf, lwork_ormqr);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 4: compute QR factorization */
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, d_info));

    /* check if QR is good or not */
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after geqrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    /* step 5: compute Q^T*B */
    CUSOLVER_CHECK(cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda,
                                    d_tau, d_B, ldb, d_work, lwork, d_info));

    /* check if QR is good or not */
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after ormqr: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    /* step 6: compute x = R \ Q^T*B */
    CUBLAS_CHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, lda, d_B, ldb));

    CUDA_CHECK(cudaMemcpyAsync(XC.data(), d_B, sizeof(double) * XC.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("X = (matlab base-1)\n");
    print_matrix(m, nrhs, XC.data(), ldb);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}