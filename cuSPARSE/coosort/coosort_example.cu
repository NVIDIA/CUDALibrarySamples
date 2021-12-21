/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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
#include <cusparse.h>

#define CHECK_CUDA(func)                                                                           \
    {                                                                                              \
        cudaError_t status = (func);                                                               \
        if (status != cudaSuccess) {                                                               \
            std::printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,              \
                        cudaGetErrorString(status), status);                                       \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    }

#define CHECK_CUSPARSE(func)                                                                       \
    {                                                                                              \
        cusparseStatus_t status = (func);                                                          \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                   \
            std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__,          \
                        cusparseGetErrorString(status), status);                                   \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    }

int main(void) {
    cusparseHandle_t cusparseH = NULL;
    cudaStream_t stream = NULL;
    cusparseSpVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;

    const int m = 3;
    const int n = 3;
    const int nnz = 4;
    /*
     * A is a 3x3 sparse matrix
     *     | 1 2 0 |
     * A = | 0 5 0 |
     *     | 0 8 0 |
     */

#if 0
    /* index starts at 0 */
    std::vector<int> h_cooRows = {2, 1, 0, 0 };
    std::vector<int> h_cooCols = {1, 1, 0, 1 };
    const std::vector<int> h_cooRowsRef = {0, 0, 1, 2};
    const std::vector<int> h_cooColsRef = {0, 1, 1, 1};
#else
    /* index starts at -2 */
    std::vector<int> h_cooRows = {0, -1, -2, -2};
    std::vector<int> h_cooCols = {-1, -1, -2, -1};
    const std::vector<int> h_cooRowsRef = {-2, -2, -1, 0};
    const std::vector<int> h_cooColsRef = {-2, -1, -1, -1};
#endif
    std::vector<double> h_cooVals = {8.0, 5.0, 1.0, 2.0};
    std::vector<int> h_P(nnz, 0);

    const std::vector<double> h_cooValsRef = {1.0, 2.0, 5.0, 8.0};
    const std::vector<int> h_PRef = {2, 3, 1, 0};

    int *d_cooRows = nullptr;
    int *d_cooCols = nullptr;
    int *d_P = nullptr;
    double *d_cooVals = nullptr;
    double *d_cooVals_sorted = nullptr;
    size_t pBufferSizeInBytes = 0;
    void *pBuffer = nullptr;

    /* step 1: create cusolver handle, bind a stream */
    CHECK_CUSPARSE(cusparseCreate(&cusparseH));

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUSPARSE(cusparseSetStream(cusparseH, stream));

    /* step 2: copy data to device */
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_cooRows), sizeof(int) * h_cooRows.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_cooCols), sizeof(int) * h_cooCols.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_P), sizeof(int) * h_P.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_cooVals), sizeof(double) * h_cooVals.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_cooVals_sorted), sizeof(double) * nnz));

    CHECK_CUDA(cudaMemcpyAsync(d_cooRows, h_cooRows.data(), sizeof(int) * h_cooRows.size(),
                               cudaMemcpyHostToDevice, stream))
    CHECK_CUDA(cudaMemcpyAsync(d_cooCols, h_cooCols.data(), sizeof(int) * h_cooCols.size(),
                               cudaMemcpyHostToDevice, stream))
    CHECK_CUDA(cudaMemcpyAsync(d_cooVals, h_cooVals.data(), sizeof(double) * h_cooVals.size(),
                               cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of COO sort */
    CHECK_CUSPARSE(cusparseXcoosort_bufferSizeExt(cusparseH, m, n, nnz, d_cooRows, d_cooCols,
                                                  &pBufferSizeInBytes));

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&pBuffer), sizeof(int) * pBufferSizeInBytes));

    /* step 4: setup permutation vector P to identity */
    CHECK_CUSPARSE(cusparseCreateIdentityPermutation(cusparseH, nnz, d_P));

    /* step 4: sort COO format by Row */
    CHECK_CUSPARSE(cusparseXcoosortByRow(cusparseH, m, n, nnz, d_cooRows, d_cooCols, d_P, pBuffer));

    /* step 5: gather sorted cooVals */
    CHECK_CUSPARSE( cusparseCreateSpVec(&vecX, h_cooVals.size(), h_cooVals.size(), d_P, d_cooVals_sorted,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, h_cooVals.size(), d_cooVals, CUDA_R_64F) );
    CHECK_CUSPARSE( cusparseGather(cusparseH, vecY, vecX) );

    /* step 6: copy data to host */
    CHECK_CUDA(cudaMemcpyAsync(h_cooRows.data(), d_cooRows, sizeof(int) * h_cooRows.size(),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_cooCols.data(), d_cooCols, sizeof(int) * h_cooCols.size(),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(
        cudaMemcpyAsync(h_P.data(), d_P, sizeof(int) * h_P.size(), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(h_cooVals.data(), d_cooVals_sorted,
                               sizeof(double) * h_cooVals.size(), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    /* step 7: Check results */
    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if (h_cooRows[i] != h_cooRowsRef[i]) { // direct floating point comparison is not
            correct = 0;                       // reliable in standard code
            break;
        }
        if (h_cooCols[i] != h_cooColsRef[i]) { // direct floating point comparison is not
            correct = 0;                       // reliable in standard code
            break;
        }
        if (h_P[i] != h_PRef[i]) { // direct floating point comparison is not
            correct = 0;           // reliable in standard code
            break;
        }
        if (h_cooVals[i] != h_cooValsRef[i]) { // direct floating point comparison is not
            correct = 0;                       // reliable in standard code
            break;
        }
    }
    if (correct)
        std::printf("coosort_example test PASSED\n");
    else
        std::printf("coosort_example test FAILED: wrong result\n");

    /* free resources */
    CHECK_CUDA(cudaFree(d_cooRows));
    CHECK_CUDA(cudaFree(d_cooCols));
    CHECK_CUDA(cudaFree(d_P));
    CHECK_CUDA(cudaFree(d_cooVals));
    CHECK_CUDA(cudaFree(d_cooVals_sorted));
    CHECK_CUDA(cudaFree(pBuffer));

    CHECK_CUSPARSE(cusparseDestroy(cusparseH));

    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
