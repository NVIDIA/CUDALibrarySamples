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

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    syevjInfo_t syevj_params = NULL;

    const int m = 3;
    const int lda = m;
    /*
     *       | 3.5 0.5 0 |
     *   A = | 0.5 3.5 0 |
     *       | 0   0   2 |
     */

    const std::vector<double> A = {3.5, 0.5, 0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};
    const std::vector<double> lambda = {2.0, 3.0, 4.0};

    std::vector<double> V(lda * m); // eigenvectors
    std::vector<double> W(m);       // eigenvalues

    double *d_A = nullptr;
    double *d_W = nullptr;
    int *devInfo = nullptr;
    double *d_work = nullptr;
    int lwork = 0;
    int info_gpu = 0;

    /* configuration of syevj  */
    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    /* numerical results of syevj  */
    double residual = 0;
    int executed_sweeps = 0;

    printf("tol = %E, default value is machine zero \n", tol);
    printf("max. sweeps = %d, default value is 100\n", max_sweeps);

    printf("A = (matlab base-1)\n");
    print_matrix(m, m, A.data(), lda);
    printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: configuration of syevj */
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));

    /* default value of tolerance is machine zero */
    CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));

    /* default value of max. sweeps is 100 */
    CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));

    /* step 3: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * lda * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * lda * m, cudaMemcpyHostToDevice, stream));

    /* step 4: query working space of syevj */
    CUSOLVER_CHECK(
        cusolverDnDsyevj_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork, syevj_params));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 5: compute eigen-pair   */
    CUSOLVER_CHECK(cusolverDnDsyevj(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo,
                                    syevj_params));

    CUDA_CHECK(
        cudaMemcpyAsync(V.data(), d_A, sizeof(double) * lda * m, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(W.data(), d_W, sizeof(double) * m, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (0 == info_gpu) {
        printf("syevj converges \n");
    } else if (0 > info_gpu) {
        printf("%d-th parameter is wrong \n", -info_gpu);
        exit(1);
    } else {
        printf("WARNING: info = %d : syevj does not converge \n", info_gpu);
    }

    printf("Eigenvalue = (matlab base-1), ascending order\n");
    for (int i = 0; i < m; i++) {
        printf("W[%d] = %E\n", i + 1, W[i]);
    }

    printf("V = (matlab base-1)\n");
    print_matrix(m, m, V.data(), lda);
    printf("=====\n");

    /* step 6: check eigenvalues */
    double lambda_sup = 0;
    for (int i = 0; i < m; i++) {
        double error = fabs(lambda[i] - W[i]);
        lambda_sup = (lambda_sup > error) ? lambda_sup : error;
    }
    printf("|lambda - W| = %E\n", lambda_sup);

    CUSOLVER_CHECK(cusolverDnXsyevjGetSweeps(cusolverH, syevj_params, &executed_sweeps));

    CUSOLVER_CHECK(cusolverDnXsyevjGetResidual(cusolverH, syevj_params, &residual));

    printf("residual |A - V*W*V**H|_F = %E \n", residual);
    printf("number of executed sweeps = %d \n", executed_sweeps);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}