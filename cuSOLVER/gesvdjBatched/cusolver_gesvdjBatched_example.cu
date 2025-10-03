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
    gesvdjInfo_t gesvdj_params = NULL;

    const int m = 3;   /* 1 <= m <= 32 */
    const int n = 2;   /* 1 <= n <= 32 */
    const int lda = m; /* lda >= m */
    const int ldu = m; /* ldu >= m */
    const int ldv = n; /* ldv >= n */
    const int batchSize = 2;
    const int minmn = (m < n) ? m : n; /* min(m,n) */

    /*
     *        |  1  -1  |
     *   A0 = | -1   2  |
     *        |  0   0  |
     *
     *   A0 = U0 * S0 * V0**T
     *   S0 = diag(2.6180, 0.382)
     *
     *        |  3   4  |
     *   A1 = |  4   7  |
     *        |  0   0  |
     *
     *   A1 = U1 * S1 * V1**T
     *   S1 = diag(9.4721, 0.5279)
     */

    std::vector<double> A(lda * n * batchSize, 0); /* A = [A0 ; A1] */
    std::vector<double> U(ldu * m * batchSize, 0); /* U = [U0 ; U1] */
    std::vector<double> V(ldv * n * batchSize, 0); /* V = [V0 ; V1] */
    std::vector<double> S(minmn * batchSize, 0);   /* S = [S0 ; S1] */
    std::vector<int> info(batchSize, 0);             /* info = [info0 ; info1] */

    double *d_A = nullptr; /* lda-by-n-by-batchSize */
    double *d_U = nullptr; /* ldu-by-m-by-batchSize */
    double *d_V = nullptr; /* ldv-by-n-by-batchSize */
    double *d_S = nullptr; /* minmn-by-batchSize */
    int *d_info = nullptr; /* batchSize */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const int sort_svd = 0;                                  /* don't sort singular values */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute singular vectors */

    double *A0 = A.data();
    double *A1 = A.data() + lda * n; /* Aj is m-by-n */

    /*
     *        |  1  -1  |
     *   A0 = | -1   2  |
     *        |  0   0  |
     *   A0 is column-major
     */
    A0[0 + 0 * lda] = 1.0;
    A0[1 + 0 * lda] = -1.0;
    A0[2 + 0 * lda] = 0.0;

    A0[0 + 1 * lda] = -1.0;
    A0[1 + 1 * lda] = 2.0;
    A0[2 + 1 * lda] = 0.0;

    /*
     *        |  3   4  |
     *   A1 = |  4   7  |
     *        |  0   0  |
     *   A1 is column-major
     */
    A1[0 + 0 * lda] = 3.0;
    A1[1 + 0 * lda] = 4.0;
    A1[2 + 0 * lda] = 0.0;

    A1[0 + 1 * lda] = 4.0;
    A1[1 + 1 * lda] = 7.0;
    A1[2 + 1 * lda] = 0.0;

    std::printf("m = %d, n = %d \n", m, n);
    std::printf("tol = %E, default value is machine zero \n", tol);
    std::printf("max. sweeps = %d, default value is 100\n", max_sweeps);

    std::printf("A0 = (matlab base-1)\n");
    print_matrix(m, n, A.data(), lda);
    std::printf("=====\n");

    std::printf("A1 = (matlab base-1)\n");
    print_matrix(m, n, A.data() + lda * n, lda);
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

    /* disable sorting */
    CUSOLVER_CHECK(cusolverDnXgesvdjSetSortEig(gesvdj_params, sort_svd));

    /* step 3: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(double) * V.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 4: query working space of gesvdjBatched */
    CUSOLVER_CHECK(cusolverDnDgesvdjBatched_bufferSize(cusolverH, jobz, m, n, d_A, lda, d_S, d_U,
                                                       ldu, d_V, ldv, &lwork, gesvdj_params,
                                                       batchSize));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 5: compute singular values of A0 and A1 */
    CUSOLVER_CHECK(cusolverDnDgesvdjBatched(cusolverH, jobz, m, n, d_A, lda, d_S, d_U, ldu, d_V,
                                            ldv, d_work, lwork, d_info, gesvdj_params, batchSize));

    CUDA_CHECK(
        cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(V.data(), d_V, sizeof(double) * V.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < batchSize; i++) {
        if (0 == info[i]) {
            std::printf("matrix %d: gesvdj converges \n", i);
        } else if (0 > info[i]) {
            /* only info[0] shows if some input parameter is wrong.
             * If so, the error is CUSOLVER_STATUS_INVALID_VALUE.
             */
            std::printf("Error: %d-th parameter is wrong \n", -info[i]);
            exit(1);
        } else { /* info = m+1 */
                 /* if info[i] is not zero, Jacobi method does not converge at i-th matrix. */
            std::printf("WARNING: matrix %d, info = %d : gesvdj does not converge \n", i, info[i]);
        }
    }

    /* Step 6: show singular values and singular vectors */
    double *S0 = S.data();
    double *S1 = S.data() + minmn;
    std::printf("==== \n");
    for (int i = 0; i < minmn; i++) {
        std::printf("S0(%d) = %20.16E\n", i + 1, S0[i]);
    }
    std::printf("==== \n");
    for (int i = 0; i < minmn; i++) {
        std::printf("S1(%d) = %20.16E\n", i + 1, S1[i]);
    }
    std::printf("==== \n");

    double *U0 = U.data();
    double *U1 = U.data() + ldu * m; /* Uj is m-by-m */
    std::printf("U0 = (matlab base-1)\n");
    print_matrix(m, m, U0, ldu);
    std::printf("U1 = (matlab base-1)\n");
    print_matrix(m, m, U1, ldu);

    double *V0 = V.data();
    double *V1 = V.data() + ldv * n; /* Vj is n-by-n */
    std::printf("V0 = (matlab base-1)\n");
    print_matrix(n, n, V0, ldv);
    std::printf("V1 = (matlab base-1)\n");
    print_matrix(n, n, V1, ldv);

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