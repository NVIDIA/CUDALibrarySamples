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
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusp/csr_matrix.h>
#include <utils/generate_random_data.h>

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

int main(void)
{
    // Host problem definition
    int A_num_rows = 4;
    int A_num_cols = 4;
    int B_num_rows = A_num_cols;
    int B_num_cols = 3;
    // int   C_nnz        = 9;
    float sparsity = 0.1f;
    int C_nnz = A_num_rows * B_num_cols * sparsity;
    int lda = A_num_cols;
    int ldb = B_num_cols;
    int A_size = lda * A_num_rows;
    int B_size = ldb * B_num_rows;
    // initializing data
    // float hA[]         = { 1.0f,   2.0f,  3.0f,  4.0f,
    //                        5.0f,   6.0f,  7.0f,  8.0f,
    //                        9.0f,  10.0f, 11.0f, 12.0f,
    //                        13.0f, 14.0f, 15.0f, 16.0f };
    // float hB[]         = {  1.0f,  2.0f,  3.0f,
    //                         4.0f,  5.0f,  6.0f,
    //                         7.0f,  8.0f,  9.0f,
    //                        10.0f, 11.0f, 12.0f };
    // int   hC_offsets[] = { 0, 3, 4, 7, 9 };
    // int   hC_columns[] = { 0, 1, 2, 1, 0, 1, 2, 0, 2 };
    // float hC_values[]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    //                        0.0f, 0.0f, 0.0f, 0.0f };
    float *hA = (float *)malloc(A_size * sizeof(float));
    float *hB = (float *)malloc(B_size * sizeof(float));
    generate_random_matrix(hA, A_size);
    generate_random_matrix(hB, B_size);
    cusp::csr_matrix<int, float, cusp::host_memory> hC = generate_random_sparse_matrix<cusp::csr_matrix<int, float, cusp::host_memory>>(A_num_rows, B_num_cols, C_nnz);

    cusp::csr_matrix<int, float, cusp::device_memory> dC(hC);

    float alpha = 1.0f;
    float beta = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    // int   *dC_offsets, *dC_columns;
    // float *dC_values,
    float *dB, *dA;
    CHECK_CUDA(cudaMalloc((void **)&dA, A_size * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(float)))
    // CHECK_CUDA( cudaMalloc((void**) &dC_offsets,
    //                        (A_num_rows + 1) * sizeof(int)) )
    // CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz * sizeof(int))   )
    // CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz * sizeof(float)) )

    CHECK_CUDA(cudaMemcpy(dA, hA, A_size * sizeof(float),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(float),
                          cudaMemcpyHostToDevice))
    // CHECK_CUDA( cudaMemcpy(dC_offsets, hC_offsets,
    //                        (A_num_rows + 1) * sizeof(int),
    //                        cudaMemcpyHostToDevice) )
    // CHECK_CUDA( cudaMemcpy(dC_columns, hC_columns, C_nnz * sizeof(int),
    //                        cudaMemcpyHostToDevice) )
    // CHECK_CUDA( cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(float),
    //                        cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create dense matrix A
    CHECK_CUSPARSE(cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW))
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW))
    // Create sparse matrix C in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, A_num_rows, B_num_cols, C_nnz,
                                     // dC_offsets, dC_columns, dC_values,
                                     (void *)thrust::raw_pointer_cast(dC.row_offsets.data()),
                                     (void *)thrust::raw_pointer_cast(dC.column_indices.data()),
                                     (void *)thrust::raw_pointer_cast(dC.values.data()),
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSDDMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSDDMM_preprocess(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer))
    // execute SpMM
    CHECK_CUSPARSE(cusparseSDDMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer))
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroySpMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    // CHECK_CUDA( cudaMemcpy(hC_values, (void*)thrust::raw_pointer_cast(dC.values.data()), C_nnz * sizeof(float),
    //                        cudaMemcpyDeviceToHost) )
    // int correct = 1;
    // for (int i = 0; i < C_nnz; i++) {
    //     if (hC_values[i] != hC_result[i]) {
    //         correct = 0; // direct floating point comparison is not reliable
    //         printf("%d: %f != %f\n", i, hC_values[i], hC_result[i]);
    //         break;
    //     }
    //     else{
    //         printf("%d: %f == %f\n", i, hC_values[i], hC_result[i]);
    //     }
    // }
    // if (correct)
    //     printf("sddmm_csr_example test PASSED\n");
    // else
    //     printf("sddmm_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA))
    CHECK_CUDA(cudaFree(dB))
    // CHECK_CUDA( cudaFree(dC_offsets) )
    // CHECK_CUDA( cudaFree(dC_columns) )
    // CHECK_CUDA( cudaFree(dC_values) )
    free(hA);
    free(hB);
    return EXIT_SUCCESS;
}
