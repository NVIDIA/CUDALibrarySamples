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

    using data_type = double;

    const int64_t m = 3;
    const int64_t lda = m;
    const int64_t ldb = m;
    /*       | 1 2 3  |
     *   A = | 4 5 6  |
     *       | 7 8 10 |
     *
     * without pivoting: A = L*U
     *       | 1 0 0 |      | 1  2  3 |
     *   L = | 4 1 0 |, U = | 0 -3 -6 |
     *       | 7 2 1 |      | 0  0  1 |
     *
     * with pivoting: P*A = L*U
     *       | 0 0 1 |
     *   P = | 1 0 0 |
     *       | 0 1 0 |
     *
     *       | 1       0     0 |      | 7  8       10     |
     *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
     *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
     */

    const std::vector<data_type> A = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    const std::vector<data_type> B = {1.0, 2.0, 3.0};
    std::vector<data_type> X(m, 0);
    std::vector<data_type> LU(lda * m, 0);
    std::vector<int64_t> Ipiv(m, 0);
    int info = 0;

    data_type *d_A = nullptr;  /* device copy of A */
    data_type *d_B = nullptr;  /* device copy of B */
    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;     /* error info */

    size_t d_lwork = 0;     /* size of workspace */
    void *d_work = nullptr; /* device workspace for getrf */
    size_t h_lwork = 0;     /* size of workspace */
    void *h_work = nullptr; /* host workspace for getrf */

    const int pivot_on = 1;
    const int algo = 0;

    if (pivot_on) {
        std::printf("pivot is on : compute P*A = L*U \n");
    } else {
        std::printf("pivot is off: compute A = L*U (not numerically stable)\n");
    }

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, m, A.data(), lda);
    std::printf("=====\n");

    std::printf("B = (matlab base-1)\n");
    print_matrix(m, 1, B.data(), ldb);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* Create advanced params */
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    if (algo == 0) {
        std::printf("Using New Algo\n");
        CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));
    } else {
        std::printf("Using Legacy Algo\n");
        CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1));
    }

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * Ipiv.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: query working space of getrf */
    CUSOLVER_CHECK(
        cusolverDnXgetrf_bufferSize(cusolverH, params, m, m, traits<data_type>::cuda_data_type, d_A,
                                    lda, traits<data_type>::cuda_data_type, &d_lwork, &h_lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));

    /* step 4: LU factorization */
    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m, traits<data_type>::cuda_data_type,
                                        d_A, lda, d_Ipiv, traits<data_type>::cuda_data_type, d_work,
                                        d_lwork, h_work, h_lwork, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, m, m, traits<data_type>::cuda_data_type,
                                        d_A, lda, nullptr, traits<data_type>::cuda_data_type,
                                        d_work, d_lwork, h_work, h_lwork, d_info));
    }

    if (pivot_on) {
        CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int64_t) * Ipiv.size(),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(LU.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xgetrf: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    if (pivot_on) {
        std::printf("pivoting sequence, matlab base-1\n");
        for (int j = 0; j < m; j++) {
            std::printf("Ipiv(%d) = %lu\n", j + 1, Ipiv[j]);
        }
    }
    std::printf("L and U = (matlab base-1)\n");
    print_matrix(m, m, LU.data(), lda);
    std::printf("=====\n");

    /*
     * step 5: solve A*X = B
     *       | 1 |       | -0.3333 |
     *   B = | 2 |,  X = |  0.6667 |
     *       | 3 |       |  0      |
     *
     */
    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, 1, /* nrhs */
                                        traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
                                        traits<data_type>::cuda_data_type, d_B, ldb, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, m, 1, /* nrhs */
                                        traits<data_type>::cuda_data_type, d_A, lda, nullptr,
                                        traits<data_type>::cuda_data_type, d_B, ldb, d_info));
    }

    CUDA_CHECK(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * X.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("X = (matlab base-1)\n");
    print_matrix(m, 1, X.data(), ldb);
    std::printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
