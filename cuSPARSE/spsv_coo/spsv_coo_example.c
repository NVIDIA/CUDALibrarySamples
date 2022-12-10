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
#include <cusparse.h>         // cusparseSpSV
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
    const int A_num_rows      = 4;
    const int A_num_cols      = 4;
    const int A_nnz           = 9;
    int       hA_coo_rows[]   = { 0, 0, 0, 1, 2, 2, 2, 3, 3 };
    int       hA_coo_cols[]   = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float     hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f };
    float     hX[]            = { 1.0f, 8.0f, 23.0f, 52.0f };
    float     hY[]            = { 0.0f, 0.0f, 0.0f, 0.0f };
    float     hY_result[]     = { 1.0f, 2.0f, 3.0f, 4.0f };
    float     alpha           = 1.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_coo_rows, *dA_coo_cols;
    float *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_coo_rows, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_coo_cols, A_nnz * sizeof(int))        )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,   A_nnz * sizeof(float))      )
    CHECK_CUDA( cudaMalloc((void**) &dX,          A_num_cols * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dY,          A_num_rows * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_coo_rows, hA_coo_rows, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_coo_cols, hA_coo_cols, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_cols * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, hY, A_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseSpSVDescr_t  spsvDescr;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in COO format
    CHECK_CUSPARSE( cusparseCreateCoo(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_coo_rows, dA_coo_cols, dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F) )
    // Create opaque data structure, that holds analysis data between calls.
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescr) )
    // Specify Lower|Upper fill mode.
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                              &fillmode, sizeof(fillmode)) )
    // Specify Unit|Non-Unit diagonal type.
    cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diagtype, sizeof(diagtype)) )
    // allocate an external buffer for analysis
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, vecY, CUDA_R_32F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
                                &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    CHECK_CUSPARSE( cusparseSpSV_analysis(
                                handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, vecY, CUDA_R_32F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer) )
    // execute SpSV
    CHECK_CUSPARSE( cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA, vecX, vecY, CUDA_R_32F,
                                       CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (hY[i] != hY_result[i]) { // direct floating point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
        printf("spsv_coo_example test PASSED\n");
    else
        printf("spsv_coo_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_coo_rows) )
    CHECK_CUDA( cudaFree(dA_coo_cols) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return EXIT_SUCCESS;
}
