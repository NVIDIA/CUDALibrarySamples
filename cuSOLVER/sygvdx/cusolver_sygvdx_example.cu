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

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    const int m = 3;
    const int lda = m;
    /*
     *       | 3.5 0.5 0 |
     *   A = | 0.5 3.5 0 |
     *       | 0   0   2 |
     *
     *       | 10  2   3 |
     *   B = | 2  10   5 |
     *       | 3   5  10 |
     */

    const std::vector<double> A = {3.5, 0.5, 0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};
    const std::vector<double> B = {10.0, 2.0, 3.0, 2.0, 10.0, 5.0, 3.0, 5.0, 10.0};
    const std::vector<double> lambda = {0.158660256604, 0.370751508101882, 0.6};

    std::vector<double> V(lda * m, 0); // eigenvectors
    std::vector<double> W(m, 0);       // eigenvalues

    double *d_A = nullptr;
    double *d_B = nullptr;
    double *d_W = nullptr;
    int *d_info = nullptr;
    double *d_work = nullptr;
    int lwork = 0;
    int info = 0;

    double vl = 0.0;
    double vu = 0.0;
    int il = 1;
    int iu = 2;
    int h_meig = 0;

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, m, A.data(), lda);
    std::printf("=====\n");

    std::printf("B = (matlab base-1)\n");
    print_matrix(m, m, B.data(), lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * W.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_B, B.data(), sizeof(double) * B.size(), cudaMemcpyHostToDevice, stream));

    // step 3: query working space of sygvd
    cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1;     // A*x = (lambda)*B*x
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_I;   // eigenvalues/eigenvectors in the half-open
                                                       // interval (vl,vu] will be found
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    /*
    cusolverStatus_t
    cusolverDnDsygvd_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigType_t itype,
        cusolverEigMode_t jobz,
        cublasFillMode_t uplo,
        int n,
        const double *A,
        int lda,
        const double *B,
        int ldb,
        const double *W,
        int *lwork);
    */
    /*
    cusolverStatus_t
    cusolverDnDsygvdx_bufferSize(
        cusolverDnHandle_t handle,
        cusolverEigType_t itype,
        cusolverEigMode_t jobz,
        cusolverEigRange_t range, <- new
        cublasFillMode_t uplo,
        int n,
        const double *A,
        int lda,
        const double *B,
        int ldb,
        double vl, <- new
        double vu, <- new
        int il, <- new
        int iu, <- new
        int *h_meig, <- new
        const double *W,
        int *lwork);
    */

    CUSOLVER_CHECK(cusolverDnDsygvdx_bufferSize(cusolverH, itype, jobz, range, uplo, m, d_A, lda,
                                               d_B, lda, vl, vu, il, iu, &h_meig, d_W, &lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    // step 4: compute spectrum of (A,B)
    CUSOLVER_CHECK(cusolverDnDsygvdx(cusolverH, itype, jobz, range, uplo, m, d_A, lda, d_B, lda, vl,
                                    vu, il, iu, &h_meig, d_W, d_work, lwork, d_info));

    CUDA_CHECK(
        cudaMemcpyAsync(V.data(), d_A, sizeof(double) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(W.data(), d_W, sizeof(double) * W.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after sygvd: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    std::printf("eigenvalue = (matlab base-1), ascending order\n");
    int idx = 1;
    for (auto const &i : W) {
        std::printf("W[%i] = %E\n", idx, i);
        idx++;
    }

    std::printf("V = (matlab base-1)\n");
    print_matrix(m, m, V.data(), lda);
    std::printf("=====\n");

    std::printf("Eigenvalues found = %d\n", h_meig);

    // step 4: check eigenvalues
    double lambda_sup = 0;
    for (int i = 0; i < m; i++) {
        double error = fabs(lambda[i] - W[i]);
        lambda_sup = (lambda_sup > error) ? lambda_sup : error;
    }
    std::printf("|lambda - W| = %E\n", lambda_sup);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
