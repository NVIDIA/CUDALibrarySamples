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

    const int batchSize = 2;
    const int m = 3;
    const int n = 2;
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int rank = n;
    const long long int strideA = static_cast<long long int>(lda * n);
    const long long int strideS = n;
    const long long int strideU = static_cast<long long int>(ldu * n);
    const long long int strideV = static_cast<long long int>(ldv * n);

    /*
     *        | 1 2  |       | 10 9 |
     *   A0 = | 4 5  |, A1 = |  8 7 |
     *        | 2 1  |       |  6 5 |
     */

    const std::vector<float> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 10.0, 8.0, 6.0, 9.0, 7.0, 5.0};
    std::vector<float> U(strideU * batchSize, 0); /* left singular vectors  */
    std::vector<float> V(strideV * batchSize, 0); /* right singular vectors */
    std::vector<float> S(strideS * batchSize, 0); /* numerical singular value */

    /* exact singular values */
    const std::vector<float> S_exact = {7.065283497082729, 1.040081297712078, 18.839649186929730,
                                        0.260035600289472};

    float *d_A = nullptr;  /* device copy of A */
    float *d_S = nullptr;  /* singular values */
    float *d_U = nullptr;  /* left singular vectors */
    float *d_V = nullptr;  /* right singular vectors */
    int *d_info = nullptr; /* error info */

    int lwork = 0;           /* size of workspace */
    float *d_work = nullptr; /* device workspace for getrf */

    std::vector<int> info(batchSize, 0);     /* host copy of error info */
    std::vector<double> RnrmF(batchSize, 0); /* residual norm */

    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute eigenvectors */

    std::printf("A0 = (matlab base-1)\n");
    print_matrix(m, n, A.data(), lda);
    std::printf("=====\n");

    std::printf("A1 = (matlab base-1)\n");
    print_matrix(m, n, A.data() + strideA, lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(float) * S.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(float) * U.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(float) * V.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
        cusolverH, jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
                         /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank,            /* number of singular values */
        m,               /* nubmer of rows of Aj, 0 <= m */
        n,               /* number of columns of Aj, 0 <= n  */
        d_A,             /* Aj is m-by-n */
        lda,             /* leading dimension of Aj */
        strideA,         /* >= lda*n */
        d_S,             /* Sj is rank-by-1, singular values in descending order */
        strideS,         /* >= rank */
        d_U,             /* Uj is m-by-rank */
        ldu,             /* leading dimension of Uj, ldu >= max(1,m) */
        strideU,         /* >= ldu*rank */
        d_V,             /* Vj is n-by-rank */
        ldv,             /* leading dimension of Vj, ldv >= max(1,n) */
        strideV,         /* >= ldv*rank */
        &lwork, batchSize /* number of matrices */
        ));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));

    /* step 4: compute SVD */
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
        cusolverH, jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
                         /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        rank,            /* number of singular values */
        m,               /* nubmer of rows of Aj, 0 <= m */
        n,               /* number of columns of Aj, 0 <= n  */
        d_A,             /* Aj is m-by-n */
        lda,             /* leading dimension of Aj */
        strideA,         /* >= lda*n */
        d_S,             /* Sj is rank-by-1 */
                         /* the singular values in descending order */
        strideS,         /* >= rank */
        d_U,             /* Uj is m-by-rank */
        ldu,             /* leading dimension of Uj, ldu >= max(1,m) */
        strideU,         /* >= ldu*rank */
        d_V,             /* Vj is n-by-rank */
        ldv,             /* leading dimension of Vj, ldv >= max(1,n) */
        strideV,         /* >= ldv*rank */
        d_work, lwork, d_info, RnrmF.data(), batchSize /* number of matrices */
        ));

    CUDA_CHECK(
        cudaMemcpyAsync(U.data(), d_U, sizeof(float) * U.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(V.data(), d_V, sizeof(float) * V.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(S.data(), d_S, sizeof(float) * S.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (0 > info[0]) {
        std::printf("%d-th parameter is wrong \n", -info[0]);
        exit(1);
    }
    for (int idx = 0; idx < batchSize; idx++) {
        if (0 == info[idx]) {
            std::printf("%d-th matrix, gesvda converges \n", idx);
        } else {
            std::printf("WARNING: info[%d] = %d : gesvda does not converge \n", idx, info[idx]);
        }
    }

    std::printf("S0 = (matlab base-1)\n");
    print_matrix(rank, 1, S.data(), n);
    std::printf("=====\n");

    std::printf("U0 = (matlab base-1)\n");
    print_matrix(m, rank, U.data(), ldu);
    std::printf("=====\n");

    std::printf("V) = (matlab base-1)\n");
    print_matrix(n, rank, V.data(), ldv);
    std::printf("=====\n");

    float ds_sup = 0;
    for (int j = 0; j < n; j++) {
        float err = fabs(S[j] - S_exact[j]);
        ds_sup = (ds_sup > err) ? ds_sup : err;
    }
    std::printf("|S0 - S0_exact|_sup = %E \n", ds_sup);

    std::printf("residual |A0 - U0*S0*V0**H|_F = %E \n", RnrmF[0]);

    std::printf("S1 = (matlab base-1)\n");
    print_matrix(rank, 1, S.data() + strideS, n);
    std::printf("=====\n");

    std::printf("U1 = (matlab base-1)\n");
    print_matrix(m, rank, U.data() + strideU, ldu);
    std::printf("=====\n");

    std::printf("V1 = (matlab base-1)\n");
    print_matrix(n, rank, V.data() + strideV, ldv);
    std::printf("=====\n");

    ds_sup = 0;
    for (int j = 0; j < n; j++) {
        float err = fabs(S[strideS + j] - S_exact[strideS + j]);
        ds_sup = (ds_sup > err) ? ds_sup : err;
    }
    std::printf("|S1 - S1_exact|_sup = %E \n", ds_sup);

    std::printf("residual |A1 - U1*S1*V1**H|_F = %E \n", RnrmF[1]);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
