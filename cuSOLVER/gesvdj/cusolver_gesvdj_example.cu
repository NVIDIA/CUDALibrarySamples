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
    gesvdjInfo_t gesvdj_params = NULL;

    const int m = 3;                   /* 1 <= m <= 32 */
    const int n = 2;                   /* 1 <= n <= 32 */
    const int lda = m;                 /* lda >= m */
    const int ldu = m;                 /* ldu >= m */
    const int ldv = n;                 /* ldv >= n */
    const int minmn = (m < n) ? m : n; /* min(m,n) */

    /*
     *       | 1 2 |
     *   A = | 4 5 |
     *       | 2 1 |
     */

    std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    std::vector<double> U(ldu * m, 0); /* m-by-m unitary matrix, left singular vectors  */
    std::vector<double> V(ldv * n, 0); /* n-by-n unitary matrix, right singular vectors */
    std::vector<double> S(minmn, 0);   /* numerical singular value */
    std::vector<double> S_exact = {7.065283497082729,
                                   1.040081297712078}; /* exact singular values */
    int info = 0;                                      /* host copy of error info */

    double *d_A = nullptr;
    double *d_S = nullptr; /* singular values */
    double *d_U = nullptr; /* left singular vectors */
    double *d_V = nullptr; /* right singular vectors */

    int *d_info = nullptr;

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    /* configuration of gesvdj  */
    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 0;                                      /* econ = 1 for economy size */

    /* numerical results of gesvdj  */
    double residual = 0;
    int executed_sweeps = 0;

    std::printf("m = %d, n = %d \n", m, n);
    std::printf("tol = %E, default value is machine zero \n", tol);
    std::printf("max. sweeps = %d, default value is 100\n", max_sweeps);
    std::printf("econ = %d \n", econ);

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, n, A.data(), lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: configuration of gesvdj */
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

    /* default value of tolerance is machine zero */
    CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));

    /* default value of max. sweeps is 100 */
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

    /* step 3: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(double) * V.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * lda * n, cudaMemcpyHostToDevice, stream));

    /* step 4: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
        cusolverH, jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
                         /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,            /* econ = 1 for economy size */
        m,               /* nubmer of rows of A, 0 <= m */
        n,               /* number of columns of A, 0 <= n  */
        d_A,             /* m-by-n */
        lda,             /* leading dimension of A */
        d_S,             /* min(m,n) */
                         /* the singular values in descending order */
        d_U,             /* m-by-m if econ = 0 */
                         /* m-by-min(m,n) if econ = 1 */
        ldu,             /* leading dimension of U, ldu >= max(1,m) */
        d_V,             /* n-by-n if econ = 0  */
                         /* n-by-min(m,n) if econ = 1  */
        ldv,             /* leading dimension of V, ldv >= max(1,n) */
        &lwork, gesvdj_params));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 5: compute SVD*/
    CUSOLVER_CHECK(cusolverDnDgesvdj(
        cusolverH, jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
                         /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,            /* econ = 1 for economy size */
        m,               /* nubmer of rows of A, 0 <= m */
        n,               /* number of columns of A, 0 <= n  */
        d_A,             /* m-by-n */
        lda,             /* leading dimension of A */
        d_S,             /* min(m,n)  */
                         /* the singular values in descending order */
        d_U,             /* m-by-m if econ = 0 */
                         /* m-by-min(m,n) if econ = 1 */
        ldu,             /* leading dimension of U, ldu >= max(1,m) */
        d_V,             /* n-by-n if econ = 0  */
                         /* n-by-min(m,n) if econ = 1  */
        ldv,             /* leading dimension of V, ldv >= max(1,n) */
        d_work, lwork, d_info, gesvdj_params));

    CUDA_CHECK(
        cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(V.data(), d_V, sizeof(double) * V.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (0 == info) {
        std::printf("gesvdj converges \n");
    } else if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    } else {
        std::printf("WARNING: info = %d : gesvdj does not converge \n", info);
    }

    std::printf("S = singular values (matlab base-1)\n");
    print_matrix(minmn, 1, S.data(), minmn);
    std::printf("=====\n");

    std::printf("U = left singular vectors (matlab base-1)\n");
    print_matrix(m, m, U.data(), ldu);
    std::printf("=====\n");

    std::printf("V = right singular vectors (matlab base-1)\n");
    print_matrix(n, n, V.data(), ldv);
    std::printf("=====\n");

    /* step 6: measure error of singular value */
    double ds_sup = 0;
    for (int j = 0; j < minmn; j++) {
        double err = fabs(S[j] - S_exact[j]);
        ds_sup = (ds_sup > err) ? ds_sup : err;
    }
    std::printf("|S - S_exact|_sup = %E \n", ds_sup);

    CUSOLVER_CHECK(cusolverDnXgesvdjGetSweeps(cusolverH, gesvdj_params, &executed_sweeps));

    CUSOLVER_CHECK(cusolverDnXgesvdjGetResidual(cusolverH, gesvdj_params, &residual));

    std::printf("residual |A - U*S*V**H|_F = %E \n", residual);
    std::printf("number of executed sweeps = %d \n", executed_sweeps);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroyGesvdjInfo(gesvdj_params));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
