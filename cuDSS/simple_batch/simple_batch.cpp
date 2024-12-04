/*
 * Copyright 2024 NVIDIA Corporation.  All rights reserved.
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
    This example demonstrates basic usage of batched cuDSS APIs for solving
    two systems of linear algebraic equations with a sparse matrices.
    Each of them can be written in the form:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
*/

#define CUDSS_EXAMPLE_FREE \
    do { \
        for (int i = 0; i < batchCount; i++) { \
            free(csr_offsets_h[i]); \
            free(csr_columns_h[i]); \
            free(csr_values_h[i]); \
            free(x_values_h[i]); \
            free(b_values_h[i]); \
            cudaFree(batch_csr_offsets_h[i]); \
            cudaFree(batch_csr_columns_h[i]); \
            cudaFree(batch_csr_values_h[i]); \
            cudaFree(batch_x_values_h[i]); \
            cudaFree(batch_b_values_h[i]); \
        } \
        cudaFree(batch_csr_offsets_d); \
        cudaFree(batch_csr_columns_d); \
        cudaFree(batch_csr_values_d); \
        cudaFree(batch_b_values_d); \
        cudaFree(batch_x_values_d); \
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


int main (int argc, char *argv[]) {
    printf("----------------------------------------------------------\n");
    printf("cuDSS example: solving two real linear systems of size 5x5 and 6x6\n"
           "with symmetric positive-definite matrices \n");
    printf("----------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int batchCount = 2;
    int n[2]    = {5, 6};
    int nnz[2]  = {8, 11};
    int nrhs[2] = {1, 1};

    int *csr_offsets_h[2] = { NULL };
    int *csr_columns_h[2] = { NULL };
    double *csr_values_h[2] = { NULL };
    double *x_values_h[2] = { NULL }, *b_values_h[2] = { NULL };

    // (intermediate) host arrays with device pointers for the batch
    int *batch_csr_offsets_h[2] = { NULL };
    int *batch_csr_columns_h[2] = { NULL };
    double *batch_csr_values_h[2] = { NULL };
    double *batch_x_values_h[2] = { NULL }, *batch_b_values_h[2] = { NULL };

    void **batch_csr_offsets_d = NULL;
    void **batch_csr_columns_d = NULL;
    void **batch_csr_values_d = NULL;
    void **batch_x_values_d = NULL, **batch_b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/
    for (int i = 0; i < batchCount; i++) {
        csr_offsets_h[i] = (int*)malloc((n[i] + 1) * sizeof(int));
        csr_columns_h[i] = (int*)malloc(nnz[i] * sizeof(int));
        csr_values_h[i] = (double*)malloc(nnz[i] * sizeof(double));
        x_values_h[i] = (double*)malloc(nrhs[i] * n[i] * sizeof(double));
        b_values_h[i] = (double*)malloc(nrhs[i] * n[i] * sizeof(double));

        if (!csr_offsets_h[i] || ! csr_columns_h[i] || !csr_values_h[i] ||
            !x_values_h[i] || !b_values_h[i]) {
            printf("Error: host memory allocation failed\n");
            return -1;
        }
    }

    /* Initialize host memory for the first A and b */
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

    /* Initialize host memory for the second A and b */
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
        CUDA_CALL_AND_CHECK(cudaMalloc(&batch_x_values_h[i], nrhs[i] * n[i] * sizeof(double)),
            "cudaMalloc for x_values");

        /* Copy host memory to device for A and b */
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_h[i], csr_offsets_h[i], (n[i] + 1) * sizeof(int),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_h[i], csr_columns_h[i], nnz[i] * sizeof(int),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_h[i], csr_values_h[i], nnz[i] * sizeof(double),
            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
        CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_h[i], b_values_h[i], nrhs[i] * n[i] * sizeof(double),
            cudaMemcpyHostToDevice), "cudaMemcpy for b_values");
    }

    /* Allocate device memory for batch pointers of A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_offsets_d, batchCount * sizeof(int*)),
        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_columns_d, batchCount * sizeof(int*)),
        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_csr_values_d, batchCount * sizeof(double*)),
        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_b_values_d, batchCount * sizeof(double*)),
        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&batch_x_values_d, batchCount * sizeof(double*)),
        "cudaMalloc for x_values");

    /* Copy host batch pointers to device */
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_offsets_d, batch_csr_offsets_h, batchCount * sizeof(int*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_columns_d, batch_csr_columns_h, batchCount * sizeof(int*),
        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_csr_values_d, batch_csr_values_h, batchCount * sizeof(double*),
        cudaMemcpyHostToDevice), "cudaMemcpy for batch_csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_b_values_d, batch_b_values_h, batchCount * sizeof(double*),
        cudaMemcpyHostToDevice), "cudaMemcpy for b_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(batch_x_values_d, batch_x_values_h, batchCount * sizeof(double*),
        cudaMemcpyHostToDevice), "cudaMemcpy for x_values");

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
    cudssMatrix_t x, b;

    int *nrows = n, *ncols = n;
    int *ldb = ncols, *ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateBatchDn(&b, batchCount, ncols, nrhs, ldb,
        batch_b_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateBatchDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateBatchDn(&x, batchCount, nrows, nrhs, ldx,
        batch_x_values_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
        status, "cudssMatrixCreateBatchDn for x");

    /* Create a matrix object for the batch of sparse input matrices. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateBatchCsr(&A, batchCount, nrows, ncols, nnz,
        batch_csr_offsets_d, NULL, batch_csr_columns_d, batch_csr_values_d,
        CUDA_R_32I, CUDA_R_64F, mtype, mview, base), status, "cudssMatrixCreateBatchCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for solve");

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    int passed = 1;
    for (int j = 0; j < batchCount; j++) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h[j], batch_x_values_h[j],
            nrhs[j] * n[j] * sizeof(double), cudaMemcpyDeviceToHost),
            "cudaMemcpy for x_values");

        for (int i = 0; i < n[j]; i++) {
            printf("batch idx = %d x[%d] = %1.4f expected %1.4f\n",
                j, i, x_values_h[j][i], double(i+1));
            if (fabs(x_values_h[j][i] - (i + 1)) > 2.e-15)
                passed = 0;
        }
        printf("\n");
    }

    /* Release the data allocated on the user side */

    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS && passed) {
        printf("Example PASSED\n");
        return 0;
    } else {
        printf("Example FAILED\n");
        return -1;
    }
}
