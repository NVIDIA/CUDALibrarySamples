/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"

/*
    This example demonstrates basic usage of cuDSS APIs for solving
    two systems of linear algebraic equations with the same sparsity pattern:
                                A1x1 = b1,
                                A2x2 = b2
    where:
        A1, A2 are the sparse input matrices (with the same non-zero pattern),
        b1,b2 are dense right-hand side vectors (or a matrices),
        x1,x2 are dense solution vectors (or a matrices).
*/

#define CUDSS_EXAMPLE_FREE \
    do { \
        free(csr_offsets_h); \
        free(csr_columns_h); \
        free(csr_values_h); \
        free(x_values_h); \
        free(b_values_h); \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d); \
        cudaFree(x_values_d); \
        cudaFree(b_values_d); \
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
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving 2 real linear 5x5 systems\n"
           "with symmetric positive-definite matrices (same pattern) \n");
    printf("---------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int n = 5;
    int nnz = 8;
    int nrhs = 1;
    int ubatch_size = 2;

    int *csr_offsets_h = NULL;
    int *csr_columns_h = NULL;
    double *csr_values_h = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    double *csr_values_d = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    csr_columns_h = (int*)malloc(nnz * sizeof(int));
    csr_values_h = (double*)malloc(ubatch_size * nnz * sizeof(double));
    x_values_h = (double*)malloc(ubatch_size * nrhs * n * sizeof(double));
    b_values_h = (double*)malloc(ubatch_size * nrhs * n * sizeof(double));

    if (!csr_offsets_h || ! csr_columns_h || !csr_values_h ||
        !x_values_h || !b_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    /* Initialize host memory for A and b */
    int i = 0;
    csr_offsets_h[i++] = 0;
    csr_offsets_h[i++] = 2;
    csr_offsets_h[i++] = 4;
    csr_offsets_h[i++] = 6;
    csr_offsets_h[i++] = 7;
    csr_offsets_h[i++] = 8;

    i = 0;
    csr_columns_h[i++] = 0; csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 1; csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 2; csr_columns_h[i++] = 4;
    csr_columns_h[i++] = 3;
    csr_columns_h[i++] = 4;

    //Fill the first matrix
    i = 0;
    csr_values_h[i++] = 4.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 3.0; csr_values_h[i++] = 2.0;
    csr_values_h[i++] = 5.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 2.0;
    //Fill the second matrix
    csr_values_h[i++] = 2.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 3.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 6.0; csr_values_h[i++] = 2.0;
    csr_values_h[i++] = 4.0;
    csr_values_h[i++] = 8.0;

    /* Note: Right-hand side b is initialized with values which correspond
       to the exact solution vectors {1, 2, 3, 4, 5}
                                 and {5, 4, 3, 2, 1} */
    //Fill the first right-hand-side
    i = 0;
    b_values_h[i++] = 7.0;
    b_values_h[i++] = 12.0;
    b_values_h[i++] = 25.0;
    b_values_h[i++] = 4.0;
    b_values_h[i++] = 13.0;
    //Fill the second right-hand-side
    b_values_h[i++] = 13.0;
    b_values_h[i++] = 15.0;
    b_values_h[i++] = 29.0;
    b_values_h[i++] = 8.0;
    b_values_h[i++] = 14.0;

    /* Allocate device memory for A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, ubatch_size * nnz * sizeof(double)),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, ubatch_size * nrhs * n * sizeof(double)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, ubatch_size * nrhs * n * sizeof(double)),
                        "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, ubatch_size * nnz * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, ubatch_size * nrhs * n * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy for b_values");

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

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                         csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
                         base), status, "cudssMatrixCreateCsr");

    /* cuDSS will factorize and solve all matrices in the batch simultaneously*/
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_UBATCH_SIZE,
        &ubatch_size, sizeof(ubatch_size)),
        status, "cudssConfigSet for CUDSS_CONFIG_UBATCH_SIZE");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, ubatch_size * nrhs * n * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    int passed = 1;
    printf("First solution\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n", i, x_values_h[i], double(i+1));
        if (fabs(x_values_h[i] - (i + 1)) > 2.e-15)
          passed = 0;
    }
    printf("Second solution\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n", i, x_values_h[1 * n * nrhs + i], double(n-i));
        if (fabs(x_values_h[1 * n * nrhs + i] - (n - i)) > 2.e-15)
          passed = 0;
    }

    /* Optionally ubatch_index can be set to factorize one specific matrix of the
     * batch or use a specific factorized matrix on solve. That allows to keep
     * in the memory multiple factorized matrices*/

    /* cuDSS will factorize the matrix and keep it inside on ubatch_index position*/
    int ubatch_index = 1;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_UBATCH_INDEX,
        &ubatch_index, sizeof(ubatch_index)),
        status, "cudssConfigSet for CUDSS_CONFIG_UBATCH_INDEX");

    /* Note that since ubatch_index is set, cudss will take just one matrix
     * as an input, so we shift pointer outside (here)*/
    CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(A, csr_values_d + ubatch_index * nnz),
        status, "cudssMatrixSetValues for matrix A");
    CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(b, b_values_d + ubatch_index * n * nrhs),
        status, "cudssMatrixSetValues for matrix A");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, ubatch_size * nrhs * n * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    printf("Second solution (solved separately by specifying CUDSS_CONFIG_UBATCH_INDEX)\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n",i, x_values_h[i], double(n-i));
        if (fabs(x_values_h[i] - (n - i)) > 2.e-15)
          passed = 0;
    }

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

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
