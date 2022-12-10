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
#include <cusparse.h>         // cusparseSparseToDense
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
    int   num_rows   = 5;
    int   num_cols   = 4;
    int   ld         = num_cols;
    int   dense_size = ld * num_rows;
    float h_dense[]  = { 1.0f,  0.0f,  2.0f,  3.0f,
                         0.0f,  4.0f,  0.0f,  0.0f,
                         5.0f,  0.0f,  6.0f,  7.0f,
                         0.0f,  8.0f,  0.0f,  9.0f,
                         0.0f, 10.0f, 11.0f,  0.0f };
    int   h_csr_offsets[]         = { 0, 0, 0, 0, 0, 0  };
    int   h_csr_columns[]         = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float h_csr_values[]          = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int   h_csr_offsets_result[]  = { 0, 3, 4, 7, 9, 11 };
    int   h_csr_columns_result[]  = { 0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2 };
    float h_csr_values_result[]   = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                      7.0f, 8.0f, 9.0f, 10.0f, 11.0f };
    //--------------------------------------------------------------------------
    // Device memory management
    int   *d_csr_offsets, *d_csr_columns;
    float *d_csr_values,  *d_dense;
    CHECK_CUDA( cudaMalloc((void**) &d_dense, dense_size * sizeof(float)))
    CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets,
                           (num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMemcpy(d_dense, h_dense, dense_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix B in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseDenseToSparse_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_analysis(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp,
                                         &nnz) )

    // allocate CSR column indices and values
    CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float)) )
    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE( cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns,
                                           d_csr_values) )
    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseDenseToSparse_convert(handle, matA, matB,
                                        CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                        dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(h_csr_offsets, d_csr_offsets,
                           (num_rows + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_csr_columns, d_csr_columns, nnz * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_csr_values, d_csr_values, nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < num_rows + 1; i++) {
        if (h_csr_offsets[i] != h_csr_offsets_result[i]) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnz; i++) {
        if (h_csr_columns[i] != h_csr_columns_result[i]) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnz; i++) {
        if (h_csr_values[i] != h_csr_values_result[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("dense2sparse_csr_example test PASSED\n");
    else
        printf("dense2sparse_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(d_csr_offsets) )
    CHECK_CUDA( cudaFree(d_csr_columns) )
    CHECK_CUDA( cudaFree(d_csr_values) )
    CHECK_CUDA( cudaFree(d_dense) )
    return EXIT_SUCCESS;
}
