/*
 * Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"

/*
    This example demonstrates usage of cuDSS helper APIs for
    matrix objects.
*/

#define CUDSS_EXAMPLE_FREE \
    do { \
        for (int i = 0; i < batchCount; i++) { \
            free(csr_offsets_h[i]); \
            free(csr_columns_h[i]); \
            free(csr_values_h[i]); \
            free(csr_offsets_h2[i]); \
            free(csr_columns_h2[i]); \
            free(csr_values_h2[i]); \
            free(b_values_h[i]); \
            cudaFree(batch_csr_offsets_h[i]); \
            cudaFree(batch_csr_columns_h[i]); \
            cudaFree(batch_csr_values_h[i]); \
            cudaFree(batch_b_values_h[i]); \
            cudaFree(batch_csr_offsets_h2[i]); \
            cudaFree(batch_csr_columns_h2[i]); \
            cudaFree(batch_csr_values_h2[i]); \
            cudaFree(batch_b_values_h2[i]); \
        } \
        cudaFree(batch_csr_offsets_d); \
        cudaFree(batch_csr_columns_d); \
        cudaFree(batch_csr_values_d); \
        cudaFree(batch_b_values_d); \
        cudaFree(batch_csr_offsets_d2); \
        cudaFree(batch_csr_columns_d2); \
        cudaFree(batch_csr_values_d2); \
        cudaFree(batch_b_values_d2); \
    } while(0);

#define CUDA_CALL_AND_CHECK(call, msg) \
    do { \
        cuda_error = call; \
        if (cuda_error != cudaSuccess) { \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n", cuda_error); \
            CUDSS_EXAMPLE_FREE; \
            return -1; \
        } \
    } while(0);


#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE; \
            return -2; \
        } \
    } while(0);

static void print_sparse_matrix(int nrows, int *csr_offsets, int *csr_columns, double *csr_values) {
    for (int i = 0; i < nrows; i++) {
        for (int j = csr_offsets[i]; j < csr_offsets[i+1]; j++) {
            printf("(%d, %3.3e) ", csr_columns[j], csr_values[j]);
        }
        printf("\n");
    }
}

int main (int argc, char *argv[]) {
    printf("----------------------------------------------------------\n");
    printf("cuDSS example: creating, modifying, and destroying cuDSS\n"
           "sparse and dense matrix object batches\n");
    printf("----------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int batchCount = 2;
    int n[2]    = {5, 6};
    int nnz[2]  = {8, 11};
    int nnz2[2] = {7, 10};
    int nrhs[2] = {1, 1};

    int *csr_offsets_h[2] = { NULL };
    int *csr_columns_h[2] = { NULL };
    double *csr_values_h[2] = { NULL };
    int *csr_offsets_h2[2] = { NULL };
    int *csr_columns_h2[2] = { NULL };
    double *csr_values_h2[2] = { NULL };
    double *b_values_h[2] = { NULL };

    // (intermediate) host arrays with device pointers for the batch
    int *batch_csr_offsets_h[2] = { NULL }, *batch_csr_offsets_h2[2] = { NULL };
    int *batch_csr_columns_h[2] = { NULL }, *batch_csr_columns_h2[2] = { NULL };
    double *batch_csr_values_h[2] = { NULL }, *batch_csr_values_h2[2] = { NULL };
    double *batch_b_values_h[2] = { NULL }, *batch_b_values_h2[2] = { NULL };

    void **batch_csr_offsets_d = NULL, **batch_csr_offsets_d2 = NULL;
    void **batch_csr_columns_d = NULL, **batch_csr_columns_d2 = NULL;
    void **batch_csr_values_d  = NULL, **batch_csr_values_d2 = NULL;
    void **batch_b_values_d = NULL, **batch_b_values_d2 = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side b*/
    for (int i = 0; i < batchCount; i++) {
        csr_offsets_h[i] = (int*)malloc((n[i] + 1) * sizeof(int));
        csr_columns_h[i] = (int*)malloc(nnz[i] * sizeof(int));
        csr_values_h[i] = (double*)malloc(nnz[i] * sizeof(double));
        csr_offsets_h2[i] = (int*)malloc((n[i] + 1) * sizeof(int));
        csr_columns_h2[i] = (int*)malloc(nnz[i] * sizeof(int));
        csr_values_h2[i] = (double*)malloc(nnz[i] * sizeof(double));
        b_values_h[i] = (double*)malloc(nrhs[i] * n[i] * sizeof(double));

        if (!csr_offsets_h[i] || ! csr_columns_h[i] || !csr_values_h[i] ||
            !b_values_h[i]) {
            printf("Error: host memory allocation failed\n");
            return -1;
        }
    }

    /* Initialize host memory for the first A and b in the batch*/
    int i = 0;
    csr_offsets_h[0][i++] = 0;
    csr_offsets_h[0][i++] = 2;
    csr_offsets_h[0][i++] = 4;
    csr_offsets_h[0][i++] = 6;
    csr_offsets_h[0][i++] = 7;
    csr_offsets_h[0][i++] = 8;

    i = 0;
    csr_columns_h[0][i++] = 0; csr_columns_h[0][i++] = 2;
    csr_columns_h[0][i++] = 1; csr_columns_h[0][i++] = 2;
    csr_columns_h[0][i++] = 2; csr_columns_h[0][i++] = 4;
    csr_columns_h[0][i++] = 3;
    csr_columns_h[0][i++] = 4;

    i = 0;
    csr_values_h[0][i++] = 4.0; csr_values_h[0][i++] = 1.0;
    csr_values_h[0][i++] = 3.0; csr_values_h[0][i++] = 2.0;
    csr_values_h[0][i++] = 5.0; csr_values_h[0][i++] = 1.0;
    csr_values_h[0][i++] = 1.0;
    csr_values_h[0][i++] = 2.0;

    /* Note: Right-hand side b is initialized with values which correspond
       to the exact solution vector {1, 2, 3, 4, 5} */
    i = 0;
    b_values_h[0][i++] = 7.0;
    b_values_h[0][i++] = 12.0;
    b_values_h[0][i++] = 25.0;
    b_values_h[0][i++] = 4.0;
    b_values_h[0][i++] = 13.0;

    /* Initialize host memory for the second A and b in the batch*/
    i = 0;
    csr_offsets_h[1][i++] = 0;
    csr_offsets_h[1][i++] = 2;
    csr_offsets_h[1][i++] = 4;
    csr_offsets_h[1][i++] = 7;
    csr_offsets_h[1][i++] = 8;
    csr_offsets_h[1][i++] = 10;
    csr_offsets_h[1][i++] = 11;

    i = 0;
    csr_columns_h[1][i++] = 0; csr_columns_h[1][i++] = 5;
    csr_columns_h[1][i++] = 1; csr_columns_h[1][i++] = 4;
    csr_columns_h[1][i++] = 2; csr_columns_h[1][i++] = 4; csr_columns_h[1][i++] = 5;
    csr_columns_h[1][i++] = 3;
    csr_columns_h[1][i++] = 4; csr_columns_h[1][i++] = 5;
    csr_columns_h[1][i++] = 5;

    i = 0;
    csr_values_h[1][i++] = 3.0; csr_values_h[1][i++] = 1.0;
    csr_values_h[1][i++] = 2.0; csr_values_h[1][i++] = 1.0;
    csr_values_h[1][i++] = 6.0; csr_values_h[1][i++] = 2.0; csr_values_h[1][i++] = 2.0;
    csr_values_h[1][i++] = 5.0;
    csr_values_h[1][i++] = 7.0; csr_values_h[1][i++] = 3.0;
    csr_values_h[1][i++] = 8.0;

    /* Print the initialized matrices A */
    printf("The first matrix A in the batch:\n");
    print_sparse_matrix(n[0], csr_offsets_h[0], csr_columns_h[0], csr_values_h[0]);

    printf("The second matrix A in the batch:\n");
    print_sparse_matrix(n[1], csr_offsets_h[1], csr_columns_h[1], csr_values_h[1]);

    /* Note: Right-hand side b is initialized with values which correspond
       to the exact solution vector {1, 2, 3, 4, 5} */
    i = 0;
    b_values_h[1][i++] = 9.0;
    b_values_h[1][i++] = 9.0;
    b_values_h[1][i++] = 40.0;
    b_values_h[1][i++] = 20.0;
    b_values_h[1][i++] = 61.0;
    b_values_h[1][i++] = 70.0;

    for (int i = 0; i < batchCount; i++) {
        /* Allocate device memory for A, x and b */
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_offsets_h[i], (n[i] + 1) * sizeof(int)),
            "cudaMalloc for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_columns_h[i], nnz[i] * sizeof(int)),
            "cudaMalloc for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_values_h[i], nnz[i] * sizeof(double)),
            "cudaMalloc for csr_values");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_b_values_h[i], nrhs[i] * n[i] * sizeof(double)),
            "cudaMalloc for b_values");

        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_offsets_h2[i], (n[i] + 1) * sizeof(int)),
            "cudaMalloc for csr_offsets2");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_columns_h2[i], nnz[i] * sizeof(int)),
            "cudaMalloc for csr_columns2");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_values_h2[i], nnz[i] * sizeof(double)),
            "cudaMalloc for csr_values2");
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_b_values_h2[i], nrhs[i] * n[i] * sizeof(double)),
            "cudaMalloc for b2_values");

        /* Copy host memory to device for A and b */
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_h[i], csr_offsets_h[i], (n[i] + 1) * sizeof(int),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_h[i], csr_columns_h[i], nnz[i] * sizeof(int),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_h[i], csr_values_h[i], nnz[i] * sizeof(double),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_h[i], b_values_h[i], nrhs[i] * n[i] * sizeof(double),
            cudaMemcpyHostToDevice), "cudaMemcpy for b_values");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_h2[i], csr_values_h[i], nnz[i] * sizeof(double),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values2");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_h2[i], b_values_h[i], nrhs[i] * n[i] * sizeof(double),
            cudaMemcpyHostToDevice), "cudaMemcpy for batch_b_values_h2");
    }

    /* Allocate device memory for batch pointers of A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_offsets_d, batchCount * sizeof(int*)),
        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_columns_d, batchCount * sizeof(int*)),
        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_values_d, batchCount * sizeof(double*)),
        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_values_d2, batchCount * sizeof(double*)),
        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_b_values_d, batchCount * sizeof(double*)),
        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_b_values_d2, batchCount * sizeof(double*)),
        "cudaMalloc for csr_values");

    /* Copy host batch pointers to device */
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_d, batch_csr_offsets_h, batchCount * sizeof(int*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_d, batch_csr_columns_h, batchCount * sizeof(int*),
        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_d, batch_csr_values_h, batchCount * sizeof(double*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_d, batch_b_values_h, batchCount * sizeof(double*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_b_values_d");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_d2, batch_csr_values_h2, batchCount * sizeof(double*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_values2");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_d2, batch_b_values_h2, batchCount * sizeof(double*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_b_values_d2");

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as batches of dense matrices). */
    cudssMatrix_t b;

    int *nrows = n, *ncols = n;
    int *ldb = ncols;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateBatchDn(&b, batchCount, ncols, nrhs, ldb,
        batch_b_values_d, CUDA_R_32I, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateBatchDn for b");

    /* Create a matrix object for the batch of sparse input matrices. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateBatchCsr(&A, batchCount, nrows, ncols, nnz,
        batch_csr_offsets_d, NULL, batch_csr_columns_d, batch_csr_values_d,
        CUDA_R_32I, CUDA_R_64F, mtype, mview, base), status, "cudssMatrixCreateBatchCsr");

    /* Note: cuDSS matrix objects are lightweight wrappers around the user data.*/
    int64_t batchCount_out;
    void *nrows_out, *ncols_out, *nnz_out, *nrhs_out, *ldb_out;
    void **batch_csr_offsets_out = NULL;
    void **batch_csr_columns_out = NULL;
    void **batch_csr_values_out = NULL;
    cudaDataType_t index_type_out, value_type_out;
    cudssMatrixType_t mtype_out;
    cudssMatrixViewType_t mview_out;
    cudssIndexBase_t index_base_out;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetBatchCsr(A,
                         &batchCount_out, &nrows_out, &ncols_out, &nnz_out,
                         &batch_csr_offsets_out, NULL, &batch_csr_columns_out,
                         &batch_csr_values_out, &index_type_out, &value_type_out,
                         &mtype_out, &mview_out, &index_base_out),
                         status, "cudssMatrixGetBatchCsr for A");

    if (batch_csr_offsets_out != batch_csr_offsets_d ||
        batch_csr_columns_out != batch_csr_columns_d ||
        batch_csr_values_out != batch_csr_values_d ||
        batchCount_out != batchCount ||
        nrows_out != nrows || ncols_out != ncols || nnz_out != nnz ||
        index_type_out != CUDA_R_32I || value_type_out != CUDA_R_64F ||
        mtype_out != mtype || mview_out != mview || index_base_out != base) {
        printf("Error: Example FAILED: wrong data returned for matrix A\n");
        CUDSS_EXAMPLE_FREE;
        return -4;
    } else {
        printf("Success: Check for matrix A from cudssMatrixGetBatchCsr passed\n");
    }

    void **batch_b_values_out = NULL;
    cudssLayout_t layout_out;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetBatchDn(b,
                         &batchCount_out, &nrows_out, &nrhs_out, &ldb_out,
                         &batch_b_values_out, &index_type_out, &value_type_out, &layout_out),
                         status, "cudssMatrixGetBatchDn for B");

    if (batch_b_values_out != batch_b_values_d ||
        batchCount_out != batchCount ||
        nrows_out != nrows || nrhs_out != nrhs || ldb_out != ldb ||
        index_type_out != CUDA_R_32I || value_type_out != CUDA_R_64F ||
        layout_out != CUDSS_LAYOUT_COL_MAJOR) {
        printf("Error: Example FAILED: wrong data returned for matrix B\n");
        CUDSS_EXAMPLE_FREE;
        return -4;
    } else {
        printf("Success: Check for matrix B from cudssMatrixGetBatchDn passed\n");
    }

    /* Replacing values of matrix A and B with new data */
    double new_csr_value[2] = {-4.0, -3.0};
    double new_b_value[2] = {-2.0, -5.0};

    CUDA_CALL_AND_CHECK(cudaMemcpy(&batch_csr_values_h2[0][0], &new_csr_value[0],
        1 * sizeof(double), cudaMemcpyHostToDevice), "cudaMemset with new values");

    CUDA_CALL_AND_CHECK(cudaMemcpy(&batch_csr_values_h2[1][0], &new_csr_value[1],
        1 * sizeof(double), cudaMemcpyHostToDevice), "cudaMemset with new values");

    CUDA_CALL_AND_CHECK(cudaMemcpy(&batch_b_values_h2[0][0], &new_b_value[0],
        1 * sizeof(double), cudaMemcpyHostToDevice), "cudaMemset with new values");

    CUDA_CALL_AND_CHECK(cudaMemcpy(&batch_b_values_h2[1][0], &new_b_value[1],
        1 * sizeof(double), cudaMemcpyHostToDevice), "cudaMemset with new values");

    CUDSS_CALL_AND_CHECK(cudssMatrixSetBatchValues(A, batch_csr_values_d2),
                         status, "cudssMatrixSetBatchValues for A");

    CUDSS_CALL_AND_CHECK(cudssMatrixSetBatchValues(b, batch_b_values_d2),
                         status, "cudssMatrixSetBatchValues for B");

    CUDSS_CALL_AND_CHECK(cudssMatrixGetBatchCsr(A, NULL, NULL, NULL, NULL,
                         NULL, NULL, NULL, &batch_csr_values_out,
                         NULL, NULL, NULL, NULL, NULL),
                         status, "cudssMatrixGetBatchCsr for A");

    CUDSS_CALL_AND_CHECK(cudssMatrixGetBatchDn(b, NULL, NULL, NULL,
                         NULL, &batch_b_values_out, NULL, NULL, NULL),
                         status, "cudssMatrixGetBatchDn for B");

    double *batch_csr_values_out_h[2] = {NULL}, *batch_b_values_out_h[2] = {NULL};
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_out_h, batch_csr_values_out,
        batchCount * sizeof(double*), cudaMemcpyDeviceToHost),
        "cudaMemcpy for batch_csr_values_out");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_out_h, batch_b_values_out,
        batchCount * sizeof(double*), cudaMemcpyDeviceToHost),
        "cudaMemcpy for batch_b_values_out");

    double out_csr_value[2];
    double out_b_value[2];

    CUDA_CALL_AND_CHECK(cudaMemcpy(&out_csr_value[0], &batch_csr_values_out_h[0][0],
        1 * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy with out values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(&out_csr_value[1], &batch_csr_values_out_h[1][0],
        1 * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy with out values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(&out_b_value[0], &batch_b_values_out_h[0][0],
        1 * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy with out values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(&out_b_value[1], &batch_b_values_out_h[1][0],
        1 * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy with out values");

    if (out_csr_value[0] != new_csr_value[0] || out_csr_value[1] != new_csr_value[1]) {
        printf("Error: Example FAILED: modification of A values out-of-place failed\n"
               "value of matrix A in each batch equals %3.3e and %3.3e (expected %3.3e %3.3e)\n",
                out_csr_value[0], out_csr_value[1], new_csr_value[0], new_csr_value[1]);
        CUDSS_EXAMPLE_FREE;
        return -6;
    } else {
        printf("Success: After modifying out-of-place value of A in each batch\n"
               "equals %3.3e and %3.3e (expected %3.3e, %3.3e)\n",
                out_csr_value[0], out_csr_value[1], new_csr_value[0], new_csr_value[1]);
    }

    if (out_b_value[0] != new_b_value[0] || out_b_value[1] != new_b_value[1]) {
        printf("Error: Example FAILED: modification of B values out-of-place failed\n"
               "value of matrix B in each batch equals %3.3e and %3.3e (expected %3.3e %3.3e)\n",
                out_b_value[0], out_b_value[1], new_b_value[0], new_b_value[1]);
        CUDSS_EXAMPLE_FREE;
        return -6;
    } else {
        printf("Success: After modifying out-of-place value of B in each batch\n"
               "equals %3.3e and %3.3e (expected %3.3e, %3.3e)\n",
                out_b_value[0], out_b_value[1], new_b_value[0], new_b_value[1]);
    }

    /* Changing the matrix structure (here: removing one of the entries) */
    /* Initialize host memory for the first A and b in the batch*/
    i = 0;
    csr_offsets_h2[0][i++] = 0;
    csr_offsets_h2[0][i++] = 2;
    csr_offsets_h2[0][i++] = 4;
    csr_offsets_h2[0][i++] = 5;
    csr_offsets_h2[0][i++] = 6;
    csr_offsets_h2[0][i++] = 7;

    i = 0;
    csr_columns_h2[0][i++] = 0; csr_columns_h2[0][i++] = 2;
    csr_columns_h2[0][i++] = 1; csr_columns_h2[0][i++] = 2;
    csr_columns_h2[0][i++] = 2;
    csr_columns_h2[0][i++] = 3;
    csr_columns_h2[0][i++] = 4;

    i = 0;
    csr_values_h2[0][i++] = 4.0; csr_values_h2[0][i++] = 1.0;
    csr_values_h2[0][i++] = 3.0; csr_values_h2[0][i++] = 2.0;
    csr_values_h2[0][i++] = 5.0;
    csr_values_h2[0][i++] = 1.0;
    csr_values_h2[0][i++] = 2.0;

    /* Initialize host memory for the second A and b in the batch*/
    i = 0;
    csr_offsets_h2[1][i++] = 0;
    csr_offsets_h2[1][i++] = 2;
    csr_offsets_h2[1][i++] = 4;
    csr_offsets_h2[1][i++] = 6;
    csr_offsets_h2[1][i++] = 7;
    csr_offsets_h2[1][i++] = 9;
    csr_offsets_h2[1][i++] = 10;

    i = 0;
    csr_columns_h2[1][i++] = 0; csr_columns_h2[1][i++] = 5;
    csr_columns_h2[1][i++] = 1; csr_columns_h2[1][i++] = 4;
    csr_columns_h2[1][i++] = 2; csr_columns_h2[1][i++] = 4;
    csr_columns_h2[1][i++] = 3;
    csr_columns_h2[1][i++] = 4; csr_columns_h2[1][i++] = 5;
    csr_columns_h2[1][i++] = 5;

    i = 0;
    csr_values_h2[1][i++] = 3.0; csr_values_h2[1][i++] = 1.0;
    csr_values_h2[1][i++] = 2.0; csr_values_h2[1][i++] = 1.0;
    csr_values_h2[1][i++] = 6.0; csr_values_h2[1][i++] = 2.0;
    csr_values_h2[1][i++] = 5.0;
    csr_values_h2[1][i++] = 7.0; csr_values_h2[1][i++] = 3.0;
    csr_values_h2[1][i++] = 8.0;

    /* Copy host memory for new matrix arrays to device */
    for (int i = 0; i < batchCount; i++) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_h2[i], csr_offsets_h2[i],
            (n[i] + 1) * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy for csr_offsets2");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_h2[i], csr_columns_h2[i],
            nnz2[i] * sizeof(int), cudaMemcpyHostToDevice),
            "cudaMemcpy for csr_columns2");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_h2[i], csr_values_h2[i],
            nnz2[i] * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy for csr_values2");
    }
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_offsets_d2, batchCount * sizeof(int*)),
        "cudaMalloc for batch_csr_offsets2");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_columns_d2, batchCount * sizeof(int*)),
        "cudaMalloc for batch_csr_columns2");

    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_d2, batch_csr_offsets_h2, batchCount * sizeof(int*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_offsets2");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_d2, batch_csr_columns_h2, batchCount * sizeof(int*),
        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns2");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_d2, batch_csr_values_h2, batchCount * sizeof(double*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_values2");
 
    /* Resetting the CSR pointers for the matrix A */
    CUDSS_CALL_AND_CHECK(cudssMatrixSetBatchCsrPointers(A, batch_csr_offsets_d2, NULL,
                         batch_csr_columns_d2, batch_csr_values_d2),
                         status, "cudssMatrixSetBatchCsrPointers for A");
    int *new_nnz_out;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetBatchCsr(A, NULL, NULL, NULL, (void**)&new_nnz_out,
                         &batch_csr_offsets_out, NULL, &batch_csr_columns_out,
                         &batch_csr_values_out, NULL, NULL, NULL, NULL, NULL),
                         status, "cudssMatrixGetBatchCsr for A");

    int *batch_csr_offsets_out_h[2] = {NULL};
    int *batch_csr_columns_out_h[2] = {NULL};
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_out_h, batch_csr_offsets_out,
        batchCount * sizeof(int*), cudaMemcpyDeviceToHost),
        "cudaMemcpy for batch_csr_offsets_out");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_out_h, batch_csr_columns_out,
        batchCount * sizeof(int*), cudaMemcpyDeviceToHost),
        "cudaMemcpy for batch_csr_columns_out");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_out_h, batch_csr_values_out,
        batchCount * sizeof(double*), cudaMemcpyDeviceToHost),
        "cudaMemcpy for batch_csr_values_out");

    /* Copy device memory for new matrices back to host */
    for (int i = 0; i < batchCount; i++) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_h2[i], batch_csr_offsets_h2[i],
            (n[i] + 1) * sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy for csr_offsets2");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_h2[i], batch_csr_columns_h2[i],
            nnz2[i] * sizeof(int), cudaMemcpyDeviceToHost),
            "cudaMemcpy for csr_columns2");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_h2[i], batch_csr_values_h2[i],
            nnz2[i] * sizeof(double), cudaMemcpyDeviceToHost),
            "cudaMemcpy for csr_values2");
    }

    if (new_nnz_out[0] == nnz[0] && new_nnz_out[1] == nnz[1]) {
        printf("Note: Changing the matrix structure does not change the nnz returned by\n"
               "cudssMatrixGetBatchCsr. The original value = %d and %d passed to cudssMatrixCreateBatchCsr are\n"
               "returned\n", new_nnz_out[0], new_nnz_out[1]);
        printf("First matrix A (after changing the matrix structure):\n");
        print_sparse_matrix(n[0], csr_offsets_h2[0], csr_columns_h2[0], csr_values_h2[0]);
        printf("Second matrix A (after changing the matrix structure):\n");
        print_sparse_matrix(n[1], csr_offsets_h2[1], csr_columns_h2[1], csr_values_h2[1]);
    } else {
        printf("Error: Example FAILED: unexpected nnz = %d and %d are returned after changing the\n"
               " matrix structure (expected %d and %d)\n",
                new_nnz_out[0], new_nnz_out[1], nnz[0], nnz[1]);
        CUDSS_EXAMPLE_FREE;
        return -7;
    }


    /* Demonstrating cudssMatrixGetFormat */
    int format;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetFormat(A, &format),
                         status, "cudssMatrixGetFormat for A");
    if (! (format & (CUDSS_MFORMAT_CSR | CUDSS_MFORMAT_BATCH))) {
        printf("Error: Example FAILED: wrong format = %d was returned for A\n",
                format);
        CUDSS_EXAMPLE_FREE;
        return -8;
    } else
        printf("Success: Matrix A format returned as %d (CSR + batch)\n", format);

    CUDSS_CALL_AND_CHECK(cudssMatrixGetFormat(b, &format),
                         status, "cudssMatrixGetFormat for b");
    if (! (format & (CUDSS_MFORMAT_DENSE | CUDSS_MFORMAT_BATCH))) {
        printf("Error: Example FAILED: wrong format = %d was returned for b\n",
                format);
        CUDSS_EXAMPLE_FREE;
        return -9;
    } else
        printf("Success: Matrix b format returned as %d (dense + batch)\n", format);

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Release the data allocated on the user side */
    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS) {
        printf("Example PASSED\n");
        return 0;
    } else {
        printf("Example FAILED\n");
        return -1;
    }
}
