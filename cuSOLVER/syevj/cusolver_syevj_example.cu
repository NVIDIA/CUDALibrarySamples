/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
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
