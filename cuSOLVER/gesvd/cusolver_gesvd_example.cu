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
    cudaStream_t stream = NULL;

    const int m = 3;   /* 1 <= m <= 32 */
    const int n = 2;   /* 1 <= n <= 32 */
    const int lda = m; /* lda >= m */

    /*
     *       | 1 2 |
     *   A = | 4 5 |
     *       | 2 1 |
     */

    const std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    std::vector<double> U(lda * m, 0);  /* m-by-m unitary matrix, left singular vectors  */
    std::vector<double> VT(lda * n, 0); /* n-by-n unitary matrix, right singular vectors */
    std::vector<double> S(n, 0);        /* numerical singular value */
    std::vector<double> S_exact = {7.065283497082729,
                                   1.040081297712078}; /* exact singular values */
    int info_gpu = 0;                                  /* host copy of error info */

    double *d_A = nullptr;
    double *d_S = nullptr;  /* singular values */
    double *d_U = nullptr;  /* left singular vectors */
    double *d_VT = nullptr; /* right singular vectors */
    double *d_W = nullptr;  /* W = S*VT */

    int *devInfo = nullptr;

    int lwork = 0; /* size of workspace */
    double *d_work = nullptr;
    double *d_rwork = nullptr;

    const double h_one = 1;
    const double h_minus_one = -1;

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, n, A.data(), lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(double) * VT.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&devInfo), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 4: compute SVD*/
    signed char jobu = 'A';  // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    CUSOLVER_CHECK(cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda, d_S, d_U,
                                    lda, // ldu
                                    d_VT,
                                    lda, // ldvt,
                                    d_work, lwork, d_rwork, devInfo));

    CUDA_CHECK(
        cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(double) * VT.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(
        cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after gesvd: info_gpu = %d\n", info_gpu);
    if (0 == info_gpu) {
        std::printf("gesvd converges \n");
    } else if (0 > info_gpu) {
        std::printf("%d-th parameter is wrong \n", -info_gpu);
        exit(1);
    } else {
        std::printf("WARNING: info = %d : gesvd does not converge \n", info_gpu);
    }

    std::printf("S = singular values (matlab base-1)\n");
    print_matrix(n, 1, S.data(), n);
    std::printf("=====\n");

    std::printf("U = left singular vectors (matlab base-1)\n");
    print_matrix(m, m, U.data(), lda);
    std::printf("=====\n");

    std::printf("VT = right singular vectors (matlab base-1)\n");
    print_matrix(n, n, VT.data(), lda);
    std::printf("=====\n");

    // step 5: measure error of singular value
    double ds_sup = 0;
    for (int j = 0; j < n; j++) {
        double err = fabs(S[j] - S_exact[j]);
        ds_sup = (ds_sup > err) ? ds_sup : err;
    }
    std::printf("|S - S_exact| = %E \n", ds_sup);

    CUBLAS_CHECK(cublasDdgmm(cublasH, CUBLAS_SIDE_LEFT, n, n, d_VT, lda, d_S, 1, d_W, lda));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * lda * n, cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasDgemm(cublasH,
                             CUBLAS_OP_N,  // U
                             CUBLAS_OP_N,  // W
                             m,            // number of rows of A
                             n,            // number of columns of A
                             n,            // number of columns of U
                             &h_minus_one, /* host pointer */
                             d_U,          // U
                             lda,
                             d_W,         // W
                             lda, &h_one, /* hostpointer */
                             d_A, lda));

    double dR_fro = 0.0;
    CUBLAS_CHECK(cublasDnrm2(cublasH, lda * n, d_A, 1, &dR_fro));

    std::printf("|A - U*S*VT| = %E \n", dR_fro);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_VT));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(devInfo));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_rwork));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
