/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cusolverDn.h>

#include "cudss.h"

/*
    This example demonstrates usage of cuDSS APIs for computing
    a (non-factorized) Schur complement matrix.
    The subset of rows and columns which define the Schur complement
    is provided as an integer vector with binary values.
    As in the main example, a system of linear algebraic equations
    with a sparse matrix is considered:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or matrix),
        x is the (dense) solution vector (or matrix).
    To solve the full system while also extracting the Schur complement,
    the Schur complement system must be solved outside of cuDSS (e.g.,
    using cuSOLVER in case when the Schur complement is dense).
*/

#define CUDSS_EXAMPLE_FREE                                                               \
    do {                                                                                 \
        free(csr_offsets_h);                                                             \
        free(csr_columns_h);                                                             \
        free(csr_values_h);                                                              \
        free(x_values_h);                                                                \
        free(b_values_h);                                                                \
        free(schur_indices_h);                                                           \
        cudaFree(csr_offsets_d);                                                         \
        cudaFree(csr_columns_d);                                                         \
        cudaFree(csr_values_d);                                                          \
        cudaFree(x_values_d);                                                            \
        cudaFree(b_values_d);                                                            \
    } while (0);

#define CUDA_CALL_AND_CHECK(call, msg)                                                   \
    do {                                                                                 \
        cuda_error = call;                                                               \
        if (cuda_error != cudaSuccess) {                                                 \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n",  \
                   cuda_error);                                                          \
            CUDSS_EXAMPLE_FREE;                                                          \
            return -1;                                                                   \
        }                                                                                \
    } while (0);


#define CUDSS_CALL_AND_CHECK(call, status, msg)                                          \
    do {                                                                                 \
        status = call;                                                                   \
        if (status != CUDSS_STATUS_SUCCESS) {                                            \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, "  \
                   "details: " #msg "\n",                                                \
                   status);                                                              \
            CUDSS_EXAMPLE_FREE;                                                          \
            return -2;                                                                   \
        }                                                                                \
    } while (0);

#define CUSOLVER_CALL_AND_CHECK(call, status, error, msg)                                \
    do {                                                                                 \
        status = call;                                                                   \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                         \
            printf(                                                                      \
                "Example FAILED: cuSOLVER call ended unsuccessfully with status = %d, "  \
                "details: " #msg "\n",                                                   \
                status);                                                                 \
            error = -3;                                                                  \
        }                                                                                \
    } while (0);

/* This function is called for solving the Schur complement system with a dense matrix.
   In this example, we use cuSOLVER.
   Note: this function solves Ax = b while factoring the matrix A in-place and
   replacing the right-hand side b with the solution x.
   Note: just for simplicity (and because cuDSS returns full dense matrix for
   the Schur complement even if it is symmetric), we use getrf/getrs from cuSOLVER.
   In a real application, one should use a different solver.
 */
int dense_matrix_solve(int64_t m, int64_t n, int64_t lda, double *A, double *b,
                       int64_t ldb, cudaStream_t stream) {
    int              error           = 0;
    cudaError_t      cuda_error      = cudaSuccess;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CALL_AND_CHECK(cusolverDnCreate(&cusolver_handle), cusolver_status, error,
                            "cusolverDnCreate for cusolver_handle");

    CUSOLVER_CALL_AND_CHECK(cusolverDnSetStream(cusolver_handle, stream), cusolver_status,
                            error, "cusolverDnSetStream for cusolver_handle");

    /* Create advanced params */
    cusolverDnParams_t cusolver_params;
    CUSOLVER_CALL_AND_CHECK(cusolverDnCreateParams(&cusolver_params), cusolver_status,
                            error, "cusolverDnCreateParams for cusolver_params");

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    size_t workspaceInBytesOnHost   = 0; /* size of workspace */

    const cudaDataType_t compute_type = CUDA_R_64F;

    CUSOLVER_CALL_AND_CHECK(
        cusolverDnXgetrf_bufferSize(cusolver_handle, cusolver_params, m, m, compute_type,
                                    A, lda, compute_type, &workspaceInBytesOnDevice,
                                    &workspaceInBytesOnHost),
        cusolver_status, error, "cusolverDnXgetrf_bufferSize for schur_matrix");

    char *work_h = NULL;
    work_h       = (char *)malloc(workspaceInBytesOnHost);
    if (!work_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    /* Allocate device memory for the workspaces, ipiv and info */
    char *work_d = NULL;
    cuda_error   = cudaMalloc(&work_d, workspaceInBytesOnDevice);
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaMalloc for work_d failed\n");
        return -1;
    }

    int *info  = NULL;
    cuda_error = cudaMalloc(&info, sizeof(int));
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaMalloc for info failed\n");
        return -1;
    }

    /* Note: ipiv is used for pivoting in the factorization (optionally) */
    const int pivot_on = 1;
    int64_t  *ipiv_d   = NULL;
    cuda_error         = cudaMalloc(&ipiv_d, m * sizeof(int64_t));
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaMalloc for ipiv_d failed\n");
        return -1;
    }

    // CUSOLVER_CALL_AND_CHECK(cusolverDnLoggerSetLevel(5), cusolver_status, error,
    // "cusolverDnLoggerSetLevel");

    CUSOLVER_CALL_AND_CHECK(
        cusolverDnXgetrf(cusolver_handle, cusolver_params, m, m, compute_type, A, lda,
                         (pivot_on ? ipiv_d : nullptr), compute_type, work_d,
                         workspaceInBytesOnDevice, work_h, workspaceInBytesOnHost, info),
        cusolver_status, error, "cusolverDnXgetrf for schur_matrix");

    CUSOLVER_CALL_AND_CHECK(cusolverDnXgetrs(cusolver_handle, cusolver_params,
                                             CUBLAS_OP_N, m, n, compute_type, A, lda,
                                             (pivot_on ? ipiv_d : nullptr), compute_type,
                                             b, ldb, info),
                            cusolver_status, error, "cusolverDnXgetrs for schur_matrix");

    CUSOLVER_CALL_AND_CHECK(cusolverDnDestroyParams(cusolver_params), cusolver_status,
                            error, "cusolverDnDestroyParams for cusolver_params");

    CUSOLVER_CALL_AND_CHECK(cusolverDnDestroy(cusolver_handle), cusolver_status, error,
                            "cusolverDnDestroy for cusolver_handle");

    free(work_h);
    cuda_error = cudaFree(work_d);
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaFree for work_d failed\n");
        return -1;
    }
    cuda_error = cudaFree(info);
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaFree for info failed\n");
        return -1;
    }
    cuda_error = cudaFree(ipiv_d);
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaFree for ipiv_d failed\n");
        return -1;
    }

    return error;
}

int main(int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: computing a 2x2 Schur complement for a\n"
           "real linear 5x5 system with a symmetric matrix\n");
    printf("---------------------------------------------------------\n");

    int major, minor, patch;
    cudssGetProperty(MAJOR_VERSION, &major);
    cudssGetProperty(MINOR_VERSION, &minor);
    cudssGetProperty(PATCH_LEVEL, &patch);
    printf("CUDSS Version (Major,Minor,PatchLevel): %d.%d.%d\n", major, minor, patch);

    cudaError_t   cuda_error = cudaSuccess;
    cudssStatus_t status     = CUDSS_STATUS_SUCCESS;

    int n    = 5;
    int nnz  = 8;
    int nrhs = 1;

    int    *csr_offsets_h = NULL;
    int    *csr_columns_h = NULL;
    double *csr_values_h  = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int    *schur_indices_h       = NULL;
    double *schur_matrix_values_h = NULL;

    int    *csr_offsets_d = NULL;
    int    *csr_columns_d = NULL;
    double *csr_values_d  = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    double *schur_matrix_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b, and the Schur
      complement index vector */

    csr_offsets_h   = (int *)malloc((n + 1) * sizeof(int));
    csr_columns_h   = (int *)malloc(nnz * sizeof(int));
    csr_values_h    = (double *)malloc(nnz * sizeof(double));
    x_values_h      = (double *)malloc(nrhs * n * sizeof(double));
    b_values_h      = (double *)malloc(nrhs * n * sizeof(double));
    schur_indices_h = (int *)malloc(n * sizeof(int));

    if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h || !b_values_h ||
        !schur_indices_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    /* Initialize host memory for A and b */
    int i              = 0;
    csr_offsets_h[i++] = 0;
    csr_offsets_h[i++] = 2;
    csr_offsets_h[i++] = 4;
    csr_offsets_h[i++] = 6;
    csr_offsets_h[i++] = 7;
    csr_offsets_h[i++] = 8;

    i                  = 0;
    csr_columns_h[i++] = 0;
    csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 1;
    csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 2;
    csr_columns_h[i++] = 4;
    csr_columns_h[i++] = 3;
    csr_columns_h[i++] = 4;

    i                 = 0;
    csr_values_h[i++] = 4.0;
    csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 3.0;
    csr_values_h[i++] = 2.0;
    csr_values_h[i++] = 5.0;
    csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 2.0;

    /* Note: Right-hand side b is initialized with values which correspond
       to the exact solution vector {1, 2, 3, 4, 5} */
    i               = 0;
    // for symmetric-case
    b_values_h[i++] = 7.0;
    b_values_h[i++] = 12.0;
    b_values_h[i++] = 25.0;
    b_values_h[i++] = 4.0;
    b_values_h[i++] = 13.0;

    /* Note: Schur complement indices should be a vector of size n with 1s
       for rows/columns which belong to the Schur complement and 0s for the rest */
    i                    = 0;
    schur_indices_h[i++] = 0;
    schur_indices_h[i++] = 1;
    schur_indices_h[i++] = 0;
    schur_indices_h[i++] = 0;
    schur_indices_h[i++] = 1;

    /* Allocate device memory for A, x, b and diag (for future use) */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(double)),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * n * sizeof(double)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(double)),
                        "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (n + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(double),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrhs * n * sizeof(double),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy for b_values");

    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t   solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Enavling the Schur complement computation */
    int compute_schur = 1;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_SCHUR_MODE,
                                        &compute_schur, sizeof(int)),
                         status, "cudssConfigSet for the Schur complement mode");

    CUDSS_CALL_AND_CHECK(cudssDataSet(handle, solverData, CUDSS_DATA_USER_SCHUR_INDICES,
                                      schur_indices_h, n * sizeof(int)),
                         status, "cudssDataSet for Schur complement indices");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices).
     */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int64_t ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t         A;
    cudssMatrixType_t     mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t      base  = CUDSS_BASE_ZERO;

    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDA_R_32I,
                                              CUDA_R_64F, mtype, mview, base),
                         status, "cudssMatrixCreateCsr");

    /* Analysis (reordering and symbolic factorization) */
    /* Note: Schur complement mode must be enabled and the corresponding indices must be
       set before the analysis phase */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b),
        status, "cudssExecute for analysis");

    size_t sizeWritten;

    /* Getting back the shape of the Schur complement matrix */
    /* Note: when the Schur complement is going to be extracted as a dense matrix,
       the dimensions are trivial (number of rows/columns is the same as the number
       of nonzero values in the Schur complement index vector). But for the sparse
       case, one additionally needs to know the number of nonzero entries.
       cuDSS returns the number of nonzero values for a general (non-symmetric)
       sparse Schur complement matrix */
    int64_t schur_shape[3] = {0};
    if (compute_schur) {
        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_SCHUR_SHAPE,
                                          &schur_shape, sizeof(schur_shape),
                                          &sizeWritten),
                             status, "cudssDataGet for schur shape");
        printf("schur shape: nrows = %ld, ncols = %ld, sparse nnz (for a general matrix) "
               " = %ld\n",
               schur_shape[0], schur_shape[1], schur_shape[2]);
    }

    /* Factorization */
    /* Note: when the Schur complement is computed, the numerical factorization and
       solve phases do not complete the full process, so one cannot do the full
       system solve without using an external solver for the Schur complement system */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b),
                         status, "cudssExecute for numerical factorization");

    /* Allocating memory and creating a matrix object for the dense Schur complement
     * matrix */

    cudssMatrix_t S; /*Schur complement matrix */

    int64_t schur_nrows = schur_shape[0];
    int64_t schur_ncols = schur_shape[1];
    int64_t schur_ld    = schur_nrows;

    CUDA_CALL_AND_CHECK(
        cudaMalloc(&schur_matrix_values_d, schur_ld * schur_shape[1] * sizeof(double)),
        "cudaMalloc for schur_matrix_values");

    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&S, schur_nrows, schur_ncols, schur_ld,
                                             schur_matrix_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for the Schur complement matrix");

    /* Retrieving the Schur complement matrix from cuDSS data object */
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_SCHUR_MATRIX, &S,
                                      sizeof(cudssMatrix_t), &sizeWritten),
                         status, "cudssDataGet for the Schur complement matrix");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    schur_matrix_values_h =
        (double *)malloc(schur_shape[0] * schur_shape[1] * sizeof(double));
    if (!schur_matrix_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    CUDA_CALL_AND_CHECK(cudaMemcpy(schur_matrix_values_h, schur_matrix_values_d,
                                   schur_ld * schur_shape[1] * sizeof(double),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy for schur_matrix_values");

    for (int i = 0; i < schur_shape[0]; i++) {
        for (int j = 0; j < schur_shape[1]; j++) {
            printf("schur_matrix_values[%d][%d] = %1.4f\n", i, j,
                   schur_matrix_values_h[j * schur_ld + i]);
        }
    }
    free(schur_matrix_values_h);

    /* As mentioned above, in case when the Schur complement is computed,
       numerical factorization is done only partially. Therefore, one cannot
       solve the full system with just one call to the solve phase.
       Instead, one needs to perform the following steps:
       1. Do the forward solve up to the Schur complement
       2. Do the diagonal solve (necessary only if matrix type is symmetric or Hermitian,
          since cuDSS for these matrix types performs LDL^T(H) factorization)
       3. Solve the Schur complement system with an external solver
       4. Do the backward solve up to the Schur complement
     */
    /* Step 1: Forward solve up to the Schur complement */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD,
                     solverConfig, solverData, A, x, b),
        status, "cudssExecute for a partial forward solve up to the Schur complement");

    /* Step 2: Diagonal solve (necessary only if matrix type is symmetric or Hermitian,
       since cuDSS for these matrix types performs LDL^T(H) factorization) */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_SOLVE_DIAG, solverConfig, solverData, A, b, x),
        status,
        "cudssExecute for a partial diagonal solve (excluding the Schur complement)");

    /* Step 3: Solve the Schur complement system with an external solver */
    int dense_solve_error =
        dense_matrix_solve(schur_shape[0], nrhs, schur_ld, schur_matrix_values_d,
                           b_values_d + (n - schur_shape[0]), n, stream);
    if (dense_solve_error != 0) {
        printf("Error: dense matrix solve failed\n");
        CUDSS_EXAMPLE_FREE;
        return dense_solve_error;
    }

    /* Since in this example the Schur complement matrix is extracted as a dense matrix,
       one can use a dense solver from LAPACK (e.g. using cuSOLVER)*/

    /* Step 4: Do the backward solve up to the Schur complement */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle,
                                      CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM,
                                      solverConfig, solverData, A, x, b),
                         status, "cudssExecute for a partial backward solve");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x_values");

    int passed = 1;
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n", i, x_values_h[i], double(i + 1));
        if (fabs(x_values_h[i] - (i + 1)) > 2.e-15)
            passed = 0;
    }

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(S), status, "cudssMatrixDestroy for S");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status,
                         "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    /* Release the data allocated on the user side */

    CUDA_CALL_AND_CHECK(cudaFree(schur_matrix_values_d),
                        "cudaFree for schur_matrix_values");

    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS && passed) {
        printf("Example PASSED\n");
        return 0;
    } else {
        printf("Example FAILED\n");
        return -1;
    }
}