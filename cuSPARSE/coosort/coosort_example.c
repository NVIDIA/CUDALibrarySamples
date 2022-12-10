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
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

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

//               | 1  0  2  3 |
//               | 0  4  0  0 |
// coo matrix:   | 5  0  6  7 |
//               | 0  8  0  9 |
//               | 0 10 11  0 |

int main(void) {
    int    num_rows     = 4;
    int    num_columns  = 4;
    int    nnz          = 11;
    int    h_rows[]    = {3, 2, 0, 3, 0, 4, 1, 0, 4, 2, 2};     // unsorted
    int    h_columns[] = {1, 0, 0, 3, 2, 2, 1, 3, 1, 2, 3};     // unsorted
    double h_values[]  = {8.0, 5.0, 1.0, 9.0, 2.0, 11.0, 4.0, 3.0, 10.0, 6.0,
                          7.0};                                 // unsorted
    double h_values_sorted[11]; // nnz
    int    h_permutation[11];   // nnz
    int    h_rows_ref[]    = {0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4}; // sorted
    int    h_columns_ref[] = {0, 2, 3, 1, 0, 2, 3, 1, 3, 1, 2}; // sorted
    double h_values_ref[]  = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                              10.0, 11.0};                      // sorted
    int    h_permutation_ref[] = {2, 4, 7, 6, 1, 9, 10, 0, 3, 8, 5};
    // sort(h_coo_values)[i] = h_coo_values[h_permutation_ref[i]]
    //--------------------------------------------------------------------------
    // Device memory management
    int    *d_rows, *d_columns, *d_permutation;
    double *d_values, *d_values_sorted;
    void   *d_buffer;
    size_t bufferSize;
    CHECK_CUDA( cudaMalloc((void**) &d_rows,          nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_columns,       nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &d_values,        nnz * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_values_sorted, nnz * sizeof(double)) )
    CHECK_CUDA( cudaMalloc((void**) &d_permutation,   nnz * sizeof(int)) )

    CHECK_CUDA( cudaMemcpy(d_rows, h_rows, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_columns, h_columns, nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(d_values, h_values, nnz * sizeof(double),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpVecDescr_t vec_permutation;
    cusparseDnVecDescr_t vec_values;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse vector for the permutation
    CHECK_CUSPARSE( cusparseCreateSpVec(&vec_permutation, nnz, nnz,
                                        d_permutation, d_values_sorted,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) )
    // Create dense vector for wrapping the original coo values
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_values, nnz, d_values,
                                        CUDA_R_64F) )

    // Query working space of COO sort
    CHECK_CUSPARSE( cusparseXcoosort_bufferSizeExt(handle, num_rows,
                                                   num_columns, nnz, d_rows,
                                                   d_columns, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&d_buffer, bufferSize) )
    // Setup permutation vector to identity
    CHECK_CUSPARSE( cusparseCreateIdentityPermutation(handle, nnz,
                                                      d_permutation) )
    CHECK_CUSPARSE( cusparseXcoosortByRow(handle, num_rows, num_columns, nnz,
                                          d_rows, d_columns, d_permutation,
                                          d_buffer) )
    CHECK_CUSPARSE( cusparseGather(handle, vec_values, vec_permutation) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpVec(vec_permutation) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vec_values) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(h_rows, d_rows, nnz * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_columns, d_columns, nnz * sizeof(int),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_values_sorted, d_values_sorted,
                           nnz * sizeof(double), cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(h_permutation, d_permutation,
                           nnz * sizeof(int), cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if (h_rows[i]          != h_rows_ref[i]    ||
            h_columns[i]       != h_columns_ref[i] ||
            h_values_sorted[i] != h_values_ref[i]  ||
            h_permutation[i]   != h_permutation_ref[i]) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("coosort_example test PASSED\n");
    else
        printf("coosort_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(d_rows) )
    CHECK_CUDA( cudaFree(d_columns) )
    CHECK_CUDA( cudaFree(d_permutation) )
    CHECK_CUDA( cudaFree(d_values) )
    CHECK_CUDA( cudaFree(d_values_sorted) )
    CHECK_CUDA( cudaFree(d_buffer) )
    return EXIT_SUCCESS;
}
