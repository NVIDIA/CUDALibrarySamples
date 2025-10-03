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
    const int n = 2;
    const int lda = m;

    /*
     *       | 1 2 |
     *   A = | 4 5 |
     *       | 2 1 |
     */

    const std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    std::vector<double> Q(lda * n, 0); // orthonormal columns
    std::vector<double> R(n * n, 0);   // R = I - Q**T*Q

    /* device memory */
    double *d_A = nullptr;
    double *d_tau = nullptr;
    int *d_info = nullptr;
    double *d_work = nullptr;

    double *d_R = nullptr;

    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    int lwork = 0;
    int info = 0;

    const double h_one = 1;
    const double h_minus_one = -1;

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, n, A.data(), lda);
    std::printf("=====\n");

    /* step 1: create cudense/cublas handle */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A and B to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(double) * R.size()));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of geqrf and orgqr */
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolverH, m, n, d_A, lda, &lwork_geqrf));

    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(cusolverH, m, n, n, d_A, lda, d_tau, &lwork_orgqr));

    lwork = std::max(lwork_geqrf, lwork_orgqr);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 4: compute QR factorization */
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, n, d_A, lda, d_tau, d_work, lwork, d_info));

    /* check if QR is successful or not */
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after geqrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    /* step 5: compute Q */
    CUSOLVER_CHECK(cusolverDnDorgqr(cusolverH, m, n, n, d_A, lda, d_tau, d_work, lwork, d_info));

    /* check if orgqr is successful or not */
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after orgqr: info = %d\n", info);
    if (0 > info) {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    CUDA_CHECK(
        cudaMemcpyAsync(Q.data(), d_A, sizeof(double) * A.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("Q = (matlab base-1)\n");
    print_matrix(m, n, Q.data(), lda);

    // step 6: measure R = I - Q**T*Q

    // Set R = I.
    std::fill(R.begin(), R.end(), 0.0);
    for (int i = 0; i < n; i++) {
        R[i + i * n] = 1.0;
    }
    CUDA_CHECK(
        cudaMemcpyAsync(d_R, R.data(), sizeof(double) * R.size(), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasDgemm(cublasH,
                             CUBLAS_OP_T,  // Q**T
                             CUBLAS_OP_N,  // Q
                             n,            // number of rows of R
                             n,            // number of columns of R
                             m,            // number of columns of Q**T
                             &h_minus_one, /* host pointer */
                             d_A,          // Q**T
                             lda,
                             d_A,         // Q
                             lda, &h_one, /* hostpointer */
                             d_R, n));

    double dR_nrm2 = 0.0;
    CUBLAS_CHECK(cublasDnrm2(cublasH, R.size(), d_R, 1, &dR_nrm2));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("|I - Q**T*Q| = %E\n", dR_nrm2);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_R));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}