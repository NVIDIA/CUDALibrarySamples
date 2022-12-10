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
#include <cuda_fp16.h>        // data types
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

const int EXIT_UNSUPPORTED = 2;

int main() {
    // Host problem definition
    int   A_num_rows      = 4;
    int   A_num_cols      = 4;
    int   A_ell_blocksize = 2;
    int   A_ell_cols      = 2;
    int   A_num_blocks    = A_ell_cols * A_num_rows /
                           (A_ell_blocksize * A_ell_blocksize);
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = 3;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
    int   hA_columns[]    = { 1, 0};
    __half hA_values[]    = { 1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f};
    __half hB[]           = { 1.0f,  2.0f,  3.0f,  4.0f,
                              5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f };
    __half hC[]           = { 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f };
    __half hC_result[]    = { 11.0f, 25.0f,  17.0f,  23.0f,
                              23.0f, 53.0f,  61.0f,  83.0f,
                              35.0f, 81.0f, 105.0f, 143.0f };
    float alpha           = 1.0f;
    float beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Check compute capability
    cudaDeviceProp props;
    CHECK_CUDA( cudaGetDeviceProperties(&props, 0) )
    if (props.major < 7) {
      std::printf("cusparseSpMM with blocked ELL format is supported only "
                  "with compute capability at least 7.0\n");
      return EXIT_UNSUPPORTED;
    }
    //--------------------------------------------------------------------------
    // Device memory management
    int    *dA_columns;
    __half *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,
                                    A_ell_cols * A_num_rows * sizeof(__half)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(__half)) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(__half)) )

    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns,
                           A_num_blocks * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                           A_ell_cols * A_num_rows * sizeof(__half),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(__half),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(__half),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(
                                      &matA,
                                      A_num_rows, A_num_cols, A_ell_blocksize,
                                      A_ell_cols, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_16F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(__half),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            float c_value  = static_cast<float>(hC[i + j * ldc]);
            float c_result = static_cast<float>(hC_result[i + j * ldc]);
            if (c_value != c_result) {
                correct = 0; // direct floating point comparison is not reliable
                break;
            }
        }
    }
    if (correct)
        std::printf("spmm_blockedell_example test PASSED\n");
    else
        std::printf("spmm_blockedell_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return EXIT_SUCCESS;
}
