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
#include <cusparse.h>         // cusparseSpMM
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
    int   A_num_rows   = 4;
    int   A_num_cols   = 4;
    int   A_nnz        = 9;
    int   B_num_rows   = A_num_cols;
    int   B_num_cols   = 3;
    int   ldb          = B_num_rows;
    int   ldc          = A_num_rows;
    int   B_size       = ldb * B_num_cols;
    int   C_size       = ldc * B_num_cols;
    int   num_batches  = 2;

    int   hA_rows[]     = { 0, 0, 0, 1, 2, 2, 2, 3, 3 };
    int   hA_columns1[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    int   hA_columns2[] = { 1, 2, 3, 0, 0, 1, 3, 1, 2 };
    float hA_values1[]  = { /*0*/ 1.0f, 2.0f, 3.0f,
                            4.0f, /*0*/ /*0*/ /*0*/
                            5.0f, /*0*/ 6.0f, 7.0f,
                            /*0*/ 8.0f, /*0*/ 9.0f };
    float hA_values2[]  = { /*0*/ 10.0f,  11.0f, 12.0f,
                            13.0f, /*0*/  /*0*/ /*0*/
                            14.0f, 15.0f, /*0*/ 16.0f,
                            /*0*/ 17.0f, 18.0f  /*0*/ };
    float hB1[]         = { 1.0f,  2.0f,  3.0f,  4.0f,
                            5.0f,  6.0f,  7.0f,  8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f };
    float hB2[]         = { 6.0f,  4.0f,  3.0f,  2.0f,
                            1.0f,  6.0f,  9.0f,  8.0f,
                            9.0f, 3.0f,   2.0f,  5.0f };

    float hC1[]         = { 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f };
    float hC2[]         = { 0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f };

    float hC1_result[]  = { 19.0f,  8.0f,  51.0f,  52.0f,
                            43.0f, 24.0f, 123.0f, 120.0f,
                            67.0f, 40.0f, 195.0f, 188.0f };
    float hC2_result[]  = { 97.0f,  78.0f,  176.0f, 122.0f,
                            255.0f, 13.0f,  232.0f, 264.0f,
                            112.0f, 117.0f, 251.0f, 87.0f };
    float  alpha        = 1.0f;
    float  beta         = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_rows, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_rows,
                           A_nnz * num_batches * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns,
                           A_nnz * num_batches * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,
                           A_nnz * num_batches * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB,
                           B_size * num_batches * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,
                           C_size * num_batches * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_rows, hA_rows, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_rows + A_nnz, hA_rows, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns1, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns + A_nnz, hA_columns2, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values1, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values + A_nnz, hA_values2, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB1, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB + B_size, hB2, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC1, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC + C_size, hC2, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in COO format
    CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_rows, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCooSetStridedBatch(matA, num_batches, A_nnz) )
    // Alternatively, the following code can be used for matA broadcast
    // CHECK_CUSPARSE( cusparseCooSetStridedBatch(matA, num_batches, 0) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matB, num_batches, B_size) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matC, num_batches, C_size) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_COO_ALG4, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_COO_ALG4, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC1, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC2, dC + C_size, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            if (hC1[i + j * ldc] != hC1_result[i + j * ldc]) {
                correct = 0; // direct floating point comparison is not reliable
                break;
            }
            if (hC2[i + j * ldc] != hC2_result[i + j * ldc]) {
                correct = 0;
                break;
            }
        }
    }
    if (correct)
        printf("spmm_coo_batched_example test PASSED\n");
    else
        printf("spmm_coo_batched_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_rows) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return EXIT_SUCCESS;
}
