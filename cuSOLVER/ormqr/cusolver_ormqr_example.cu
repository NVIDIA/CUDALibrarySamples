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

    std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0};
    // std::vector<double> X = {1.0, 1.0, 1.0}; // exact solution
    std::vector<double> B = {6.0, 15.0, 4.0};
    std::vector<double> XC(ldb * nrhs); // solution matrix from GPU

    /* device memory */
    double *d_A = nullptr;
    double *d_tau = nullptr;
    double *d_B = nullptr;
    int *devInfo = nullptr;
    double *d_work = nullptr;

    int lwork_geqrf = 0;
    int lwork_ormqr = 0;
    int lwork = 0;
    int info_gpu = 0;

    const double one = 1;

    printf("A = (matlab base-1)\n");
    print_matrix(m, m, A.data(), lda, CUBLAS_OP_T);
    printf("=====\n");
    printf("B = (matlab base-1)\n");
    print_matrix(m, nrhs, B.data(), ldb, CUBLAS_OP_T);
    printf("=====\n");

    /* step 1: create cudense/cublas handle */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A and B to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * lda * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(double) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * ldb * nrhs));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * lda * m, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice,
                               stream));

    /* step 3: query working space of geqrf and ormqr */
    CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(cusolverH, m, m, d_A, lda, &lwork_geqrf));

    CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m,
                                               d_A, lda, d_tau, d_B, ldb, &lwork_ormqr));

    lwork = std::max(lwork_geqrf, lwork_ormqr);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 4: compute QR factorization */
    CUSOLVER_CHECK(cusolverDnDgeqrf(cusolverH, m, m, d_A, lda, d_tau, d_work, lwork, devInfo));

    /* check if QR is good or not */
    CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("after geqrf: info_gpu = %d\n", info_gpu);
    if (0 != info_gpu) {
        throw std::runtime_error("the i-th parameter is wrong.");
    }

    /* step 5: compute Q^T*B */
    CUSOLVER_CHECK(cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, nrhs, m, d_A, lda,
                                    d_tau, d_B, ldb, d_work, lwork, devInfo));

    /* check if QR is good or not */
    CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("after ormqr: info_gpu = %d\n", info_gpu);
    if (0 != info_gpu) {
        throw std::runtime_error("the i-th parameter is wrong.");
    }

    /* step 6: compute x = R \ Q^T*B */
    CUBLAS_CHECK(cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                             CUBLAS_DIAG_NON_UNIT, m, nrhs, &one, d_A, lda, d_B, ldb));

    CUDA_CHECK(cudaMemcpyAsync(XC.data(), d_B, sizeof(double) * ldb * nrhs, cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("X = (matlab base-1)\n");
    print_matrix(m, nrhs, XC.data(), ldb, CUBLAS_OP_T);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}