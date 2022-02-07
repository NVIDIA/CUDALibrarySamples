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

    /* check if QR is good or not */
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after ormqr: info = %d\n", info);
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
    CUDA_CHECK(
        cudaMemcpyAsync(R.data(), d_R, sizeof(double) * R.size(), cudaMemcpyDeviceToHost, stream));

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
