/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#ifndef CUSPARSE_ENABLE_EXPERIMENTAL_API
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#endif
#include <cusparse.h>         // cusparseSpMatConvert
#include <stdint.h>           // int64_t
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                                    \
{                                                                           \
    cudaError_t status = (func);                                            \
    if (status != cudaSuccess) {                                            \
        printf("CUDA API failed at line %d with error: %s (%d)\n",          \
               __LINE__, cudaGetErrorString(status), status);               \
        return EXIT_FAILURE;                                                \
    }                                                                       \
}

#define CHECK_CUSPARSE(func)                                                \
{                                                                           \
    cusparseStatus_t status = (func);                                       \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",      \
               __LINE__, cusparseGetErrorString(status), status);           \
        return EXIT_FAILURE;                                                \
    }                                                                       \
}

int main(void) {
    // Host problem definition: convert a 4x4 matrix from CSR to CSC.
    //         | 1  0  2  3 |
    //     A = | 0  4  0  0 |
    //         | 5  0  6  7 |
    //         | 0  8  0  9 |
    const int num_rows = 4;
    const int num_cols = 4;
    const int nnz      = 9;
    int   h_csr_offsets[] = { 0, 3, 4, 7, 9 };
    int   h_csr_columns[] = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float h_csr_values[]  = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f };
    // Expected result: the same matrix in CSC (column-major compression)
    int   h_csc_offsets_result[] = { 0, 2, 4, 6, 9 };
    int   h_csc_rows_result[]    = { 0, 2, 1, 3, 0, 2, 0, 2, 3 };
    float h_csc_values_result[]  = { 1.0f, 5.0f, 4.0f, 8.0f, 2.0f,
                                     6.0f, 3.0f, 7.0f, 9.0f };
    //--------------------------------------------------------------------------
    // Device memory management
    int   *d_csr_offsets, *d_csr_columns, *d_csc_offsets, *d_csc_rows;
    float *d_csr_values,  *d_csc_values;
    CHECK_CUDA( cudaMalloc((void**) &d_csr_offsets,
                           (num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_columns, nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &d_csr_values,  nnz * sizeof(float)) )
    // The CSC column-offset array has a known size (num_cols + 1); the
    // row-index and value arrays are allocated after the analysis step.
    CHECK_CUDA( cudaMalloc((void**) &d_csc_offsets,
                           (num_cols + 1) * sizeof(int)) )

    CHECK_CUDA( cudaMemcpy(d_csr_offsets, h_csr_offsets,
                           (num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_csr_columns, h_csr_columns, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_csr_values, h_csr_values, nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t            handle      = NULL;
    cusparseSpMatDescr_t        matA        = NULL; // source: CSR
    cusparseSpMatDescr_t        matC        = NULL; // target: CSC
    cusparseSpMatConvertDescr_t convertDesc = NULL;
    void*  dBuffer    = NULL;
    size_t bufferSize = 0;
    cusparseOperation_t       opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseAction_t          act = CUSPARSE_ACTION_NUMERIC;
    cusparseSpMatConvertAlg_t alg = CUSPARSE_SPMAT_CONVERT_ALG_DEFAULT;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create source matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, num_rows, num_cols, nnz,
                                      d_csr_offsets, d_csr_columns,
                                      d_csr_values, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create target matrix C in CSC format with the same dimensions and nnz.
    // The row-index and value arrays are attached after planning.
    CHECK_CUSPARSE( cusparseCreateCsc(&matC, num_rows, num_cols, nnz,
                                      d_csc_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create the conversion descriptor
    CHECK_CUSPARSE( cusparseSpMatConvertDescrCreate(&convertDesc) )

    // Query the external workspace size and allocate the buffer
    CHECK_CUSPARSE( cusparseSpMatConvert_bufferSize(handle, opA, matA, matC,
                                        act, alg, convertDesc, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // Prepare the conversion (must be called before cusparseSpMatConvert)
    CHECK_CUSPARSE( cusparseSpMatConvert_analysis(handle, opA, matA, matC,
                                        act, alg, convertDesc, dBuffer) )

    // Query the target sizes and allocate the remaining CSC arrays
    cusparseSpMatSizes_t sizes;
    CHECK_CUSPARSE( cusparseSpMatGetSizes(matC, &sizes) )
    int64_t csc_nnz = sizes.data.csc.nnz;
    CHECK_CUDA( cudaMalloc((void**) &d_csc_rows,   csc_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &d_csc_values, csc_nnz * sizeof(float)) )

    // Attach the allocated storage to the target descriptor
    cusparseSpMatPointers_t pointers;
    pointers.format              = CUSPARSE_FORMAT_CSC;
    pointers.data.csc.colOffsets = d_csc_offsets;
    pointers.data.csc.rowInd     = d_csc_rows;
    pointers.data.csc.values     = d_csc_values;
    CHECK_CUSPARSE( cusparseSpMatSetPointers(matC, &pointers) )

    // Execute the CSR -> CSC conversion
    CHECK_CUSPARSE( cusparseSpMatConvert(handle, opA, matA, matC,
                                         act, alg, convertDesc, dBuffer) )

    // Destroy descriptors
    CHECK_CUSPARSE( cusparseSpMatConvertDescrDestroy(convertDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    int   h_csc_offsets[5]; // num_cols + 1
    int   h_csc_rows[9];    // nnz
    float h_csc_values[9];  // nnz
    CHECK_CUDA( cudaMemcpy(h_csc_offsets, d_csc_offsets,
                           (num_cols + 1) * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_csc_rows, d_csc_rows, csc_nnz * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_csc_values, d_csc_values, csc_nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < num_cols + 1; i++) {
        if (h_csc_offsets[i] != h_csc_offsets_result[i]) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < csc_nnz; i++) {
        if (h_csc_rows[i] != h_csc_rows_result[i]) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < csc_nnz; i++) {
        if (h_csc_values[i] != h_csc_values_result[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("spmat_convert_example test PASSED\n");
    else
        printf("spmat_convert_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer)       )
    CHECK_CUDA( cudaFree(d_csr_offsets) )
    CHECK_CUDA( cudaFree(d_csr_columns) )
    CHECK_CUDA( cudaFree(d_csr_values)  )
    CHECK_CUDA( cudaFree(d_csc_offsets) )
    CHECK_CUDA( cudaFree(d_csc_rows)    )
    CHECK_CUDA( cudaFree(d_csc_values)  )
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
