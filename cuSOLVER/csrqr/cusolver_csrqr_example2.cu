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
#include <cusolverSp.h>
#include <cusparse.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
    cusolverSpHandle_t cusolverH = NULL;
    csrqrInfo_t info = NULL;
    cusparseMatDescr_t descrA = NULL;
    cudaStream_t stream = NULL;

    // GPU does batch QR
    // d_A is CSR format, d_csrValA is of size nnzA*batchSize
    // d_x is a matrix of size batchSize * m
    // d_b is a matrix of size batchSize * m
    int *d_csrRowPtrA = nullptr;
    int *d_csrColIndA = nullptr;
    double *d_csrValA = nullptr;
    double *d_b = nullptr; // batchSize * m
    double *d_x = nullptr; // batchSize * m

    size_t size_qr = 0;
    size_t size_internal = 0;
    void *buffer_qr = nullptr; // working space for numerical factorization

    /*
     *      | 1                |
     *  A = |       2          |
     *      |            3     |
     *      | 0.1  0.1  0.1  4 |
     *  CSR of A is based-1
     *
     *  b = [1 1 1 1]
     */

    const int m = 4;
    const int nnzA = 7;
    const std::vector<int> csrRowPtrA = {1, 2, 3, 4, 8};
    const std::vector<int> csrColIndA = {1, 2, 3, 1, 2, 3, 4};
    const std::vector<double> csrValA = {1.0, 2.0, 3.0, 0.1, 0.1, 0.1, 4.0};
    const std::vector<double> b = {1.0, 1.0, 1.0, 1.0};
    const int batchSize = 17;

    std::vector<double> csrValABatch(nnzA * batchSize);
    std::vector<double> bBatch(m * batchSize);
    std::vector<double> xBatch(m * batchSize);

    // step 1: prepare Aj and bj on host
    //  Aj is a small perturbation of A
    //  bj is a small perturbation of b
    //  csrValABatch = [A0, A1, A2, ...]
    //  bBatch = [b0, b1, b2, ...]
    for (int colidx = 0; colidx < nnzA; colidx++) {
        double Areg = csrValA[colidx];
        for (int batchId = 0; batchId < batchSize; batchId++) {
            double eps = (static_cast<double>((std::rand() % 100) + 1)) * 1.e-4;
            csrValABatch[batchId * nnzA + colidx] = Areg + eps;
        }
    }

    for (int j = 0; j < m; j++) {
        double breg = b[j];
        for (int batchId = 0; batchId < batchSize; batchId++) {
            double eps = (static_cast<double>((std::rand() % 100) + 1)) * 1.e-4;
            bBatch[batchId * m + j] = breg + eps;
        }
    }

    // step 2: create cusolver handle, qr info and matrix descriptor
    CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverSpSetStream(cusolverH, stream));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));

    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE)); // base-1

    CUSOLVER_CHECK(cusolverSpCreateCsrqrInfo(&info));

    // step 3: copy Aj and bj to device
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_csrValA), sizeof(double) * csrValABatch.size()));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_csrColIndA), sizeof(int) * csrColIndA.size()));
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_csrRowPtrA), sizeof(int) * csrRowPtrA.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * bBatch.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double) * xBatch.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_csrColIndA, csrColIndA.data(), sizeof(int) * csrColIndA.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_csrRowPtrA, csrRowPtrA.data(), sizeof(int) * csrRowPtrA.size(),
                               cudaMemcpyHostToDevice, stream));

    // step 4: symbolic analysis
    CUSOLVER_CHECK(cusolverSpXcsrqrAnalysisBatched(cusolverH, m, m, nnzA, descrA, d_csrRowPtrA,
                                                   d_csrColIndA, info));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // step 5: find "proper" batchSize
    // get available device memory
    size_t free_mem = 0;
    size_t total_mem = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    int batchSizeMax = 2;
    while (batchSizeMax < batchSize) {
        std::printf("batchSizeMax = %d\n", batchSizeMax);
        CUSOLVER_CHECK(cusolverSpDcsrqrBufferInfoBatched(cusolverH, m, m, nnzA,
                                                         // d_csrValA is don't care
                                                         descrA, d_csrValA, d_csrRowPtrA,
                                                         d_csrColIndA,
                                                         batchSizeMax, // WARNING: use batchSizeMax
                                                         info, &size_internal, &size_qr));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if ((size_internal + size_qr) > free_mem) {
            // current batchSizeMax exceeds hardware limit, so cut it by half.
            batchSizeMax /= 2;
            break;
        }
        batchSizeMax *= 2; // double batchSizMax and try it again.
    }
    // correct batchSizeMax such that it is not greater than batchSize.
    batchSizeMax = std::min(batchSizeMax, batchSize);
    std::printf("batchSizeMax = %d\n", batchSizeMax);

    // Assume device memory is not big enough, and batchSizeMax = 2
    batchSizeMax = 2;

    // step 6: prepare working space
    // [necessary]
    // Need to call cusolverDcsrqrBufferInfoBatched again with batchSizeMax
    // to fix batchSize used in numerical factorization.
    CUSOLVER_CHECK(cusolverSpDcsrqrBufferInfoBatched(cusolverH, m, m, nnzA,
                                                     // d_csrValA is don't care
                                                     descrA, d_csrValA, d_csrRowPtrA, d_csrColIndA,
                                                     batchSizeMax, // WARNING: use batchSizeMax
                                                     info, &size_internal, &size_qr));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("numerical factorization needs internal data %lld bytes\n",
           static_cast<long long>(size_internal));
    std::printf("numerical factorization needs working space %lld bytes\n",
           static_cast<long long>(size_qr));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&buffer_qr), size_qr));

    // step 7: solve Aj*xj = bj
    for (int idx = 0; idx < batchSize; idx += batchSizeMax) {
        // current batchSize 'cur_batchSize' is the batchSize used in numerical
        // factorization
        const int cur_batchSize = std::min(batchSizeMax, batchSize - idx);
        std::printf("current batchSize = %d\n", cur_batchSize);
        // copy part of Aj and bj to device
        CUDA_CHECK(cudaMemcpyAsync(d_csrValA, csrValABatch.data() + idx * nnzA,
                                   sizeof(double) * nnzA * cur_batchSize, cudaMemcpyHostToDevice,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b, bBatch.data() + idx * m, sizeof(double) * m * cur_batchSize,
                                   cudaMemcpyHostToDevice, stream));
        // solve part of Aj*xj = bj
        CUSOLVER_CHECK(cusolverSpDcsrqrsvBatched(cusolverH, m, m, nnzA, descrA, d_csrValA,
                                                 d_csrRowPtrA, d_csrColIndA, d_b, d_x,
                                                 cur_batchSize, // WARNING: use current batchSize
                                                 info, buffer_qr));
        // copy part of xj back to host
        CUDA_CHECK(cudaMemcpyAsync(xBatch.data() + idx * m, d_x, sizeof(double) * m * cur_batchSize,
                                   cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // step 8: check residual
    // xBatch = [x0, x1, x2, ...]
    const int baseA = (CUSPARSE_INDEX_BASE_ONE == cusparseGetMatIndexBase(descrA)) ? 1 : 0;

    for (int batchId = 0; batchId < batchSize; batchId++) {
        // measure |bj - Aj*xj|
        double *csrValAj = csrValABatch.data() + batchId * nnzA;
        double *xj = xBatch.data() + batchId * m;
        double *bj = bBatch.data() + batchId * m;
        // sup| bj - Aj*xj|
        double sup_res = 0;
        for (int row = 0; row < m; row++) {
            const int start = csrRowPtrA[row] - baseA;
            const int end = csrRowPtrA[row + 1] - baseA;
            double Ax = 0.0; // Aj(row,:)*xj
            for (int colidx = start; colidx < end; colidx++) {
                const int col = csrColIndA[colidx] - baseA;
                const double Areg = csrValAj[colidx];
                const double xreg = xj[col];
                Ax = Ax + Areg * xreg;
            }
            double r = bj[row] - Ax;
            sup_res = (sup_res > fabs(r)) ? sup_res : fabs(r);
        }
        std::printf("batchId %d: sup|bj - Aj*xj| = %E \n", batchId, sup_res);
    }

    for (int batchId = 0; batchId < batchSize; batchId++) {
        double *xj = xBatch.data() + batchId * m;
        for (int row = 0; row < m; row++) {
            std::printf("x%d[%d] = %E\n", batchId, row, xj[row]);
        }
        std::printf("\n");
    }

    /* free resources */
    CUDA_CHECK(cudaFree(d_csrRowPtrA));
    CUDA_CHECK(cudaFree(d_csrColIndA));
    CUDA_CHECK(cudaFree(d_csrValA));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(buffer_qr));

    CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
