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

template <typename T> static void gen_1d_laplacian(int N, T *A, int lda) {
    memset(A, 0, sizeof(T) * lda * N);
    for (int J = 1; J <= N; J++) {
        /* A(J,J) = 2 */
        A[IDX2F(J, J, lda)] = 2.0;
        if ((J - 1) >= 1) {
            /* A(J, J-1) = -1*/
            A[IDX2F(J, J - 1, lda)] = -1.0;
        }
        if ((J + 1) <= N) {
            /* A(J, J+1) = -1*/
            A[IDX2F(J, J + 1, lda)] = -1.0;
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

    const int N = 2111;
    const int IA = 1;
    const int JA = 1;
    const int T_A = 256; /* tile size */
    const int lda = N;

    int info = 0;

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    cudaLibMgMatrixDesc_t descrA;
    cudaLibMgGrid_t gridA;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

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
    std::vector<data_type> D(N, 0);

    std::printf("Step 4: Prepare 1D Laplacian \n");
    gen_1d_laplacian<data_type>(N, &A[IDX2F(IA, JA, lda)], lda);

#ifdef SHOW_FORMAT
    std::printf("A = matlab base-1\n");
    print_matrix(N, N, A.data(), lda);
#endif

    std::printf("Step 5: Create matrix descriptors for A and D \n");

    CUSOLVER_CHECK(cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList.data(), mapping));

    /* (global) A is N-by-N */
    CUSOLVER_CHECK(cusolverMgCreateMatrixDesc(&descrA, N, /* nubmer of rows of (global) A */
                                              N,          /* number of columns of (global) A */
                                              N,          /* number or rows in a tile */
                                              T_A,        /* number of columns in a tile */
                                              traits<data_type>::cuda_data_type, gridA));

    std::printf("Step 6: Allocate distributed matrices A and D \n");

    std::vector<data_type *> array_d_A(nbGpus, nullptr);

    const int A_num_blks = (N + T_A - 1) / T_A;
    const int blks_per_device = (A_num_blks + nbGpus - 1) / nbGpus;

    for (int p = 0; p < nbGpus; p++) {
        CUDA_CHECK(cudaSetDevice(deviceList[p]));
        CUDA_CHECK(cudaMalloc(&(array_d_A[p]), sizeof(double) * lda * T_A * blks_per_device));
    }

    printf("Step 7: Prepare data on devices \n");
    /* The following setting only works for IA = JA = 1 */
    for (int k = 0; k < A_num_blks; k++) {
        /* k = ibx * nbGpus + p */
        const int p = (k % nbGpus);
        const int ibx = (k / nbGpus);
        double *h_Ak = A.data() + (size_t)lda * T_A * k;
        double *d_Ak = array_d_A[p] + (size_t)lda * T_A * ibx;
        const int width = std::min(T_A, (N - T_A * k));
        CUDA_CHECK(cudaMemcpy(d_Ak, h_Ak, sizeof(double) * lda * width, cudaMemcpyHostToDevice));
    }
    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Step 8: Allocate workspace space \n");
    CUSOLVER_CHECK(cusolverMgSyevd_bufferSize(
        cusolverH, (cusolverEigMode_t)jobz, CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
        N, reinterpret_cast<void **>(array_d_A.data()), IA,         /* base-1 */
        JA,                                                         /* base-1 */
        descrA, reinterpret_cast<void *>(D.data()), traits<data_type>::cuda_data_type,
        traits<data_type>::cuda_data_type, &lwork));

    std::printf("\tAllocate device workspace, lwork = %lld \n", static_cast<long long>(lwork));

    std::vector<data_type *> array_d_work(nbGpus, nullptr);

    /* array_d_work[j] points to device workspace of device j */
    workspaceAlloc(nbGpus, deviceList.data(),
                   sizeof(data_type) * lwork, /* number of bytes per device */
                   reinterpret_cast<void **>(array_d_work.data()));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Step 9: Compute eigenvalues and eigenvectors \n");
    CUSOLVER_CHECK(cusolverMgSyevd(
        cusolverH, (cusolverEigMode_t)jobz, CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
        N, reinterpret_cast<void **>(array_d_A.data()),             /* exit: eigenvectors */
        IA, JA, descrA, reinterpret_cast<void **>(D.data()),        /* exit: eigenvalues */
        traits<data_type>::cuda_data_type, traits<data_type>::cuda_data_type,
        reinterpret_cast<void **>(array_d_work.data()), lwork, &info /* host */
        ));

    /* sync all devices */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* check if SYEVD converges */
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    std::printf("Step 10: Copy eigenvectors to A and eigenvalues to D\n");

    memcpyD2H<data_type>(nbGpus, deviceList.data(), N, N,
                         /* input */
                         N,   /* number of columns of global A */
                         T_A, /* number of columns per column tile */
                         lda, /* leading dimension of local A */
                         array_d_A.data(), IA, JA,
                         /* output */
                         A.data(), /* N-y-N eigenvectors */
                         lda);

#ifdef SHOW_FORMAT
    /* D is 1-by-N */
    std::printf("Eigenvalue D = \n");
    print_matrix(1, N, D.data(), 1);
#endif

    std::printf("Step 11: Verify eigenvales \n");
    std::printf("     lambda(k) = 4 * sin(pi/2 *k/(N+1))^2 for k = 1:N \n");
    data_type max_err_D = 0;
    for (int k = 1; k <= N; k++) {
        const data_type pi = 4 * atan(1.0);
        const data_type h = 1.0 / (static_cast<data_type>(N) + 1);
        const data_type factor = sin(pi / 2.0 * (static_cast<data_type>(k)) * h);
        const data_type lambda = 4.0 * factor * factor;
        const data_type err = fabs(D[IDX1F(k)] - lambda);
        max_err_D = (max_err_D > err) ? max_err_D : err;
    }
    std::printf("\n|D - lambda|_inf = %E\n\n", max_err_D);

    std::printf("step 12: Free resources \n");
    destroyMat(nbGpus, deviceList.data(), N, /* number of columns of global A */
               T_A,                          /* number of columns per column tile */
               reinterpret_cast<void **>(array_d_A.data()));

    workspaceFree(nbGpus, deviceList.data(), reinterpret_cast<void **>(array_d_work.data()));

    return EXIT_SUCCESS;
}
