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

    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    const int batchSize = 2;
    const int nrhs = 1;
    const int m = 3;
    const int lda = m;
    const int ldb = m;
    /*
     *      | 1     2     3 |
     * A0 = | 2     5     5 | = L0 * L0**T
     *      | 3     5    12 |
     *
     *            | 1.0000         0         0 |
     * where L0 = | 2.0000    1.0000         0 |
     *            | 3.0000   -1.0000    1.4142 |
     *
     *      | 1     2     3 |
     * A1 = | 2     4     5 | is not s.p.d., failed at row 2
     *      | 3     5    12 |
     *
     */

    const std::vector<double> A0 = {1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0};
    const std::vector<double> A1 = {1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 12.0};
    const std::vector<double> B0 = {1.0, 1.0, 1.0};
    std::vector<double> X0(m, 0);             /* X0 = A0\B0 */
    std::vector<int> infoArray(batchSize, 0); /* host copy of error info */

    std::vector<double> L0(lda * m); /* cholesky factor of A0 */

    std::vector<double *> Aarray(batchSize, nullptr);
    std::vector<double *> Barray(batchSize, nullptr);

    double **d_Aarray = nullptr;
    double **d_Barray = nullptr;
    int *d_infoArray = nullptr;

    std::printf("A0 = (matlab base-1)\n");
    print_matrix(m, m, A0.data(), lda);
    std::printf("=====\n");

    std::printf("A1 = (matlab base-1)\n");
    print_matrix(m, m, A1.data(), lda);
    std::printf("=====\n");

    std::printf("B0 = (matlab base-1)\n");
    print_matrix(m, 1, B0.data(), ldb);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    for (int j = 0; j < batchSize; j++) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Aarray[j]), sizeof(double) * lda * m));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&Barray[j]), sizeof(double) * ldb * nrhs));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_infoArray), sizeof(int) * infoArray.size()));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Aarray), sizeof(double *) * Aarray.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Barray), sizeof(double *) * Barray.size()));

    CUDA_CHECK(cudaMemcpyAsync(Aarray[0], A0.data(), sizeof(double) * A0.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(Aarray[1], A1.data(), sizeof(double) * A1.size(),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(
        cudaMemcpyAsync(Barray[0], B0.data(), sizeof(double) * B0.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(Barray[1], B0.data(), sizeof(double) * B0.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_Aarray, Aarray.data(), sizeof(double) * Aarray.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_Barray, Barray.data(), sizeof(double) * Barray.size(),
                               cudaMemcpyHostToDevice, stream));

    /* step 3: Cholesky factorization */
    CUSOLVER_CHECK(
        cusolverDnDpotrfBatched(cusolverH, uplo, m, d_Aarray, lda, d_infoArray, batchSize));

    CUDA_CHECK(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int) * infoArray.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(L0.data(), Aarray[0], sizeof(double) * lda * m,
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int j = 0; j < batchSize; j++) {
        std::printf("info[%d] = %d\n", j, infoArray[j]);
    }

    assert(0 == infoArray[0]);
    /* A1 is singular */
    assert(2 == infoArray[1]);

    std::printf("L = (matlab base-1), upper triangle is don't care \n");
    print_matrix(m, m, L0.data(), lda);
    std::printf("=====\n");

    /*
     * step 4: solve A0*X0 = B0
     *        | 1 |        | 10.5 |
     *   B0 = | 1 |,  X0 = | -2.5 |
     *        | 1 |        | -1.5 |
     */
    CUSOLVER_CHECK(cusolverDnDpotrsBatched(cusolverH, uplo, m, nrhs, /* only support rhs = 1*/
                                           d_Aarray, lda, d_Barray, ldb, d_infoArray, batchSize));

    CUDA_CHECK(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(
        cudaMemcpyAsync(X0.data(), Barray[0], sizeof(double) * X0.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after potrsBatched: infoArray[0] = %d\n", infoArray[0]);
    if (0 > infoArray[0]) {
        std::printf("%d-th parameter is wrong \n", -infoArray[0]);
        exit(1);
    }

    std::printf("X0 = (matlab base-1)\n");
    print_matrix(m, 1, X0.data(), ldb);
    std::printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_Aarray));
    CUDA_CHECK(cudaFree(d_Barray));
    CUDA_CHECK(cudaFree(d_infoArray));
    for (int j = 0; j < batchSize; j++) {
        CUDA_CHECK(cudaFree(Aarray[j]));
        CUDA_CHECK(cudaFree(Barray[j]));
    }

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
