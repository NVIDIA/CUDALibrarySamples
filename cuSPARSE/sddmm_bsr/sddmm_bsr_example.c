/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
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
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSDDMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

int main(void) {
    // Host problem definition
    int   A_num_rows    = 8;
    int   A_num_cols    = 8;
    int   B_num_rows    = A_num_cols;
    int   B_num_cols    = 8;
    int   row_block_dim = 4;
    int   col_block_dim = 4;
    int   C_num_brows   = A_num_rows / row_block_dim;
    int   C_num_bcols   = B_num_cols / col_block_dim;
    int   C_bnnz        = 2;
    int   C_nnz         = C_bnnz * row_block_dim * col_block_dim;
    int   lda           = A_num_rows;
    int   ldb           = B_num_cols;
    int   A_size        = lda * A_num_cols;
    int   B_size        = ldb * B_num_rows;
    float hA[]          = { 1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f };
    float hB[]          = { 1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f,
                            1.0f,   2.0f,  3.0f,  4.0f, 5.0f,   6.0f,  7.0f,  8.0f };
    int   hC_boffsets[] = { 0, 1, 2 };
    int   hC_bcolumns[] = { 0, 1 };
    float hC_values[]   = { 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f};
    float hC_result[]   = { 36.0f, 72.0f, 108.0f, 144.0f,
                            36.0f, 72.0f, 108.0f, 144.0f,
                            36.0f, 72.0f, 108.0f, 144.0f,
                            36.0f, 72.0f, 108.0f, 144.0f,
                            180.0f, 216.0f, 252.0f, 288.0f,
                            180.0f, 216.0f, 252.0f, 288.0f,
                            180.0f, 216.0f, 252.0f, 288.0f,
                            180.0f, 216.0f, 252.0f, 288.0f};
    // C in CUSPARSE_ORDER_COL
    /* float hC_result[]  = { 36.0f, 36.0f, 36.0f, 36.0f, */
    /*                        72.0f, 72.0f, 72.0f, 72.0f, */
    /*                        108.0f, 108.0f, 108.0f, 108.0f, */
    /*                        144.0f,144.0f,144.0f,144.0f, */
    /*                        180.0f, 180.0f, 180.0f, 180.0f, */
    /*                        216.0f, 216.0f, 216.0f, 216.0f, */
    /*                        252.0f, 252.0f, 252.0f, 252.0f, */
    /*                        288.0f,288.0f,288.0f,288.0f}; */
    // A in CUSPARSE_ORDER_COL
    /* float hC_result[]  = {8.0f, 16.0f, 24.0f, 32.0f, */
    /*                       16.0f, 32.0f, 48.0f, 64.0f, */
    /*                       24.0f, 48.0f, 72.0f, 96.0f, */
    /*                       32.0f, 64.0f, 96.0f, 128.0f, */
    /*                       200.0f, 240.0f, 280.0f, 320.0f, */
    /*                       240.0f, 288.0f, 336.0f, 384.0f, */
    /*                       280.0f, 336.0f, 392.0f, 448.0f, */
    /*                       320.0f, 384.0f, 448.0f, 512.0f}; */
    // B in CUSPARSE_ORDER_COL
    /* float hC_result[]  = {204.0f, 204.0f, 204.0f, 204.0f, */
    /*                       204.0f, 204.0f, 204.0f, 204.0f, */
    /*                       204.0f, 204.0f, 204.0f, 204.0f, */
    /*                       204.0f, 204.0f, 204.0f, 204.0f, */
    /*                       204.0f, 204.0f, 204.0f, 204.0f, */
    /*                       204.0f, 204.0f, 204.0f, 204.0f, */
    /*                       204.0f, 204.0f, 204.0f, 204.0f, */
    /*                       204.0f, 204.0f, 204.0f, 204.0f }; */

    float alpha        = 1.0f;
    float beta         = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dC_boffsets, *dC_bcolumns;
    float *dC_values, *dB, *dA;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_boffsets,
                           (C_num_brows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_bcolumns, C_bnnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_boffsets, hC_boffsets,
                           (C_num_brows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_bcolumns, hC_bcolumns, C_bnnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix C in BSR format
    CHECK_CUSPARSE( cusparseCreateBsr(&matC, C_num_brows, C_num_bcols,
                                      C_bnnz, row_block_dim, col_block_dim,
                                      dC_boffsets, dC_bcolumns, dC_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F,
                                      CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSDDMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute preprocess (optional)
    CHECK_CUSPARSE( cusparseSDDMM_preprocess(
                                  handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    // execute SDDMM
    CHECK_CUSPARSE( cusparseSDDMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC_values, dC_values, C_nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < C_nnz; i++) {
        /* printf("%d %0.2f %0.2f\n", i, hC_values[i], hC_result[i]); */
        if (hC_values[i] != hC_result[i]) {
            correct = 0; // direct floating point comparison is not reliable
            break;
        }
    }
    if (correct)
        printf("sddmm_bsr_example test PASSED\n");
    else
        printf("sddmm_bsr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC_boffsets) )
    CHECK_CUDA( cudaFree(dC_bcolumns) )
    CHECK_CUDA( cudaFree(dC_values) )
    return EXIT_SUCCESS;
}
