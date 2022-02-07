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
#include <cusolverMg.h>

#include "cusolverMg_utils.h"
#include "cusolver_utils.h"

/* compute |x|_inf */
template <typename T> static T vec_nrm_inf(int n, const T *x) {
    T max_nrm = 0.0;
    for (int row = 1; row <= n; row++) {
        T xi = x[IDX1F(row)];
        max_nrm = (max_nrm > fabs(xi)) ? max_nrm : fabs(xi);
    }
    return max_nrm;
}

/* A is 1D laplacian, return A(N:-1:1, :) */
template <typename T> static void gen_1d_laplacian(int N, T *A, int lda) {
    for (int J = 1; J <= N; J++) {
        /* A(J,J) = 2 */
        A[IDX2F(N - J + 1, J, lda)] = 2.0;
        if ((J - 1) >= 1) {
            /* A(J, J-1) = -1*/
            A[IDX2F(N - J + 1, J - 1, lda)] = -1.0;
        }
        if ((J + 1) <= N) {
            /* A(J, J+1) = -1*/
            A[IDX2F(N - J + 1, J + 1, lda)] = -1.0;
        }
    }
}

int main(int argc, char *argv[]) {
    cusolverMgHandle_t cusolverH = NULL;

    using data_type = double;

    /* maximum number of GPUs */
    const int MAX_NUM_DEVICES = 16;

    int nbGpus = 0;
    std::vector<int> deviceList(MAX_NUM_DEVICES);

    const int N = 611;
    const int IA = 1;
    const int JA = 1;
    const int T_A = 256; /* tile size */
    const int lda = N;

    const int IB = 1;
    const int JB = 1;
    const int T_B = 100; /* tile size of B */
    const int ldb = N;

    int info = 0;

    cudaLibMgMatrixDesc_t descrA;
    cudaLibMgMatrixDesc_t descrB;
    cudaLibMgGrid_t gridA;
    cudaLibMgGrid_t gridB;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    int64_t lwork_getrf = 0;
    int64_t lwork_getrs = 0;
    int64_t lwork = 0; /* workspace: number of elements per device */

    std::printf("Test 1D Laplacian of order %d\n", N);

    std::printf("Step 1: Create Mg handle and select devices \n");
    CUSOLVER_CHECK(cusolverMgCreate(&cusolverH));

    CUDA_CHECK(cudaGetDeviceCount(&nbGpus));

    nbGpus = (nbGpus < MAX_NUM_DEVICES) ? nbGpus : MAX_NUM_DEVICES;
    std::printf("\tThere are %d GPUs \n", nbGpus);
    for (int j = 0; j < nbGpus; j++) {
        deviceList[j] = j;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, j));
        std::printf("\tDevice %d, %s, cc %d.%d \n", j, prop.name, prop.major, prop.minor);
    }

    CUSOLVER_CHECK(cusolverMgDeviceSelect(cusolverH, nbGpus, deviceList.data()));

    std::printf("step 2: Enable peer access.\n");
    enablePeerAccess(nbGpus, deviceList.data());

    std::printf("Step 3: Allocate host memory A \n");
    std::vector<data_type> A(lda * N, 0);
    std::vector<data_type> B(ldb, 0);
    std::vector<data_type> X(ldb, 0);
    std::vector<int> IPIV(N, 0);

    std::printf("Step 4: Prepare 1D Laplacian \n");
    gen_1d_laplacian<data_type>(N, &A[IDX2F(IA, JA, lda)], lda);

#ifdef SHOW_FORMAT
    std::printf("A = matlab base-1\n");
    print_matrix(N, N, A.data(), lda);
#endif

    /* B = ones(N,1) */
    for (int row = 1; row <= N; row++) {
        B[IDX1F(row)] = 1.0;
    }

#ifdef SHOW_FORMAT
    std::printf("B = matlab base-1\n");
    print_matrix(N, 1, B.data(), ldb, CUBLAS_OP_T);
#endif

    std::printf("Step 5: Create matrix descriptors for A and B \n");

    CUSOLVER_CHECK(cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping));
    CUSOLVER_CHECK(cusolverMgCreateDeviceGrid(&gridB, 1, nbGpus, deviceList.data(), mapping));

    /* (global) A is N-by-N */
    CUSOLVER_CHECK(cusolverMgCreateMatrixDesc(&descrA, N, /* nubmer of rows of (global) A */
                                              N,          /* number of columns of (global) A */
                                              N,          /* number or rows in a tile */
                                              T_A,        /* number of columns in a tile */
                                              traits<data_type>::cuda_data_type, gridA));

    /* (global) B is N-by-1 */
    CUSOLVER_CHECK(cusolverMgCreateMatrixDesc(&descrB, N, /* nubmer of rows of (global) B */
                                              1,          /* number of columns of (global) B */
                                              N,          /* number or rows in a tile */
                                              T_B,        /* number of columns in a tile */
                                              traits<data_type>::cuda_data_type, gridB));

    std::printf("Step 6: Allocate distributed matrices A and D \n");

    std::vector<data_type *> array_d_A(nbGpus, nullptr);
    std::vector<data_type *> array_d_B(nbGpus, nullptr);
    std::vector<int *> array_d_IPIV(nbGpus, nullptr);

    /* A := 0 */
    createMat<data_type>(nbGpus, deviceList.data(), N, /* number of columns of global A */
                         T_A,                          /* number of columns per column tile */
                         lda,                          /* leading dimension of local A */
                         array_d_A.data());

    /* B := 0 */
    createMat<data_type>(nbGpus, deviceList.data(), 1, /* number of columns of global B */
                         T_B,                          /* number of columns per column tile */
                         ldb,                          /* leading dimension of local B */
                         array_d_B.data());

    /* IPIV := 0, IPIV is consistent with A */
    createMat<int>(nbGpus, deviceList.data(), N, /* number of columns of global IPIV */
                   T_A,                          /* number of columns per column tile */
                   1,                            /* leading dimension of local IPIV */
                   array_d_IPIV.data());

    std::printf("Step 7: Prepare data on devices \n");
    memcpyH2D<data_type>(nbGpus, deviceList.data(), N, N,
                         /* input */
                         A.data(), lda,
                         /* output */
                         N,                /* number of columns of global A */
                         T_A,              /* number of columns per column tile */
                         lda,              /* leading dimension of local A */
                         array_d_A.data(), /* host pointer array of dimension nbGpus */
                         IA, JA);

    memcpyH2D<data_type>(nbGpus, deviceList.data(), N, 1,
                         /* input */
                         B.data(), ldb,
                         /* output */
                         1,                /* number of columns of global A */
                         T_B,              /* number of columns per column tile */
                         ldb,              /* leading dimension of local A */
                         array_d_B.data(), /* host pointer array of dimension nbGpus */
                         IB, JB);

    std::printf("Step 8: Allocate workspace space \n");
    CUSOLVER_CHECK(cusolverMgGetrf_bufferSize(
        cusolverH, N, N, reinterpret_cast<void **>(array_d_A.data()), IA, /* base-1 */
        JA,                                                               /* base-1 */
        descrA, array_d_IPIV.data(), traits<data_type>::cuda_data_type, &lwork_getrf));

    CUSOLVER_CHECK(cusolverMgGetrs_bufferSize(
        cusolverH, CUBLAS_OP_N, N, 1, /* NRHS */
        reinterpret_cast<void **>(array_d_A.data()), IA, JA, descrA, array_d_IPIV.data(),
        reinterpret_cast<void **>(array_d_B.data()), IB, JB, descrB,
        traits<data_type>::cuda_data_type, &lwork_getrs));

    lwork = std::max(lwork_getrf, lwork_getrs);
    std::printf("\tAllocate device workspace, lwork = %lld \n", static_cast<long long>(lwork));

    std::vector<data_type *> array_d_work(nbGpus, nullptr);

    /* array_d_work[j] points to device workspace of device j */
    workspaceAlloc(nbGpus, deviceList.data(),
                   sizeof(data_type) * lwork, /* number of bytes per device */
                   reinterpret_cast<void **>(array_d_work.data()));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Step 9: Solve A*X = B by GETRF and GETRS \n");
    CUSOLVER_CHECK(
        cusolverMgGetrf(cusolverH, N, N, reinterpret_cast<void **>(array_d_A.data()), IA, JA,
                        descrA, array_d_IPIV.data(), traits<data_type>::cuda_data_type,
                        reinterpret_cast<void **>(array_d_work.data()), lwork, &info /* host */
                        ));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if A is singular */
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    CUSOLVER_CHECK(cusolverMgGetrs(cusolverH, CUBLAS_OP_N, N, 1, /* NRHS */
                                   reinterpret_cast<void **>(array_d_A.data()), IA, JA, descrA,
                                   array_d_IPIV.data(), reinterpret_cast<void **>(array_d_B.data()),
                                   IB, JB, descrB, traits<data_type>::cuda_data_type,
                                   reinterpret_cast<void **>(array_d_work.data()), lwork,
                                   &info /* host */
                                   ));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if parameters are valid */
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    std::printf("Step 10: Retrieve IPIV and solution vector X\n");

    memcpyD2H<data_type>(nbGpus, deviceList.data(), N, 1,
                         /* input */
                         1,   /* number of columns of global B */
                         T_B, /* number of columns per column tile */
                         ldb, /* leading dimension of local B */
                         array_d_B.data(), IB, JB,
                         /* output */
                         X.data(), /* N-by-1 */
                         ldb);

    /* IPIV is consistent with A, use JA and T_A */
    memcpyD2H<int>(nbGpus, deviceList.data(), 1, N,
                   /* input */
                   N,   /* number of columns of global IPIV */
                   T_A, /* number of columns per column tile */
                   1,   /* leading dimension of local IPIV */
                   array_d_IPIV.data(), 1, JA,
                   /* output */
                   IPIV.data(), /* 1-by-N */
                   1);

#ifdef SHOW_FORMAT
    /* X is N-by-1 */
    std::printf("X = matlab base-1\n");
    print_matrix(N, 1, X.data(), ldb, CUBLAS_OP_T);
#endif

#ifdef SHOW_FORMAT
    /* IPIV is 1-by-N */
    std::printf("IPIV = matlab base-1, 1-by-%d matrix\n", N);
    for (int row = 1; row <= N; row++) {
        std::printf("IPIV(%d) = %d \n", row, IPIV[IDX1F(row)]);
    }
#endif

    std::printf("Step 11: Measure residual error |b - A*x| \n");
    data_type max_err = 0;
    for (int row = 1; row <= N; row++) {
        data_type sum = 0.0;
        for (int col = 1; col <= N; col++) {
            data_type Aij = A[IDX2F(row, col, lda)];
            data_type xj = X[IDX1F(col)];
            sum += Aij * xj;
        }
        data_type bi = B[IDX1F(row)];
        data_type err = fabs(bi - sum);

        max_err = (max_err > err) ? max_err : err;
    }
    data_type x_nrm_inf = vec_nrm_inf(N, X.data());
    data_type b_nrm_inf = vec_nrm_inf(N, B.data());

    data_type A_nrm_inf = 4.0;
    data_type rel_err = max_err / (A_nrm_inf * x_nrm_inf + b_nrm_inf);
    std::printf("\n|b - A*x|_inf = %E\n", max_err);
    std::printf("|x|_inf = %E\n", x_nrm_inf);
    std::printf("|b|_inf = %E\n", b_nrm_inf);
    std::printf("|A|_inf = %E\n", A_nrm_inf);
    /* relative error is around machine zero  */
    /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
    std::printf("|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);

    std::printf("step 12: Free resources \n");
    destroyMat(nbGpus, deviceList.data(), N, /* number of columns of global A */
               T_A,                          /* number of columns per column tile */
               reinterpret_cast<void **>(array_d_A.data()));
    destroyMat(nbGpus, deviceList.data(), 1, /* number of columns of global B */
               T_B,                          /* number of columns per column tile */
               reinterpret_cast<void **>(array_d_B.data()));
    destroyMat(nbGpus, deviceList.data(), N, /* number of columns of global IPIV */
               T_A,                          /* number of columns per column tile */
               reinterpret_cast<void **>(array_d_IPIV.data()));

    workspaceFree(nbGpus, deviceList.data(), reinterpret_cast<void **>(array_d_work.data()));

    return EXIT_SUCCESS;
}
