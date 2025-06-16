/*
 * Copyright 2023-2025 NVIDIA Corporation.  All rights reserved.
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
    This example demonstrates usage of cuDSS APIs with a focus on
    using non-default settings and retrieving extra data from the
    solver.
    As in the main example, a system of linear algebraic equations
    with a sparse matrix is solved:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or matrix),
        x is the (dense) solution vector (or matrix).
*/

#define CUDSS_EXAMPLE_FREE \
    do { \
        free(csr_offsets_h); \
        free(csr_columns_h); \
        free(csr_values_h); \
        free(x_values_h); \
        free(b_values_h); \
        free(diag_h);       \
        free(row_scale_h); \
        free(col_scale_h); \
        cudaFree(csr_offsets_d); \
        cudaFree(csr_columns_d); \
        cudaFree(csr_values_d); \
        cudaFree(x_values_d); \
        cudaFree(b_values_d); \
        cudaFree(diag_d); \
        cudaFree(row_scale_d); \
        cudaFree(col_scale_d); \
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
    printf("cuDSS example: solving a real linear 5x5 system\n"
           "with a symmetric positive-definite matrix\n"
           "with extra settings and extra information retrieved\n");
    printf("---------------------------------------------------------\n");

    int major, minor, patch;
    cudssGetProperty(MAJOR_VERSION, &major);
    cudssGetProperty(MINOR_VERSION, &minor);
    cudssGetProperty(PATCH_LEVEL,   &patch);
    printf("CUDSS Version (Major,Minor,PatchLevel): %d.%d.%d\n", major, minor, patch);

    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int n = 5;
    int nnz = 8;
    int nrhs = 1;

    int *csr_offsets_h = NULL;
    int *csr_columns_h = NULL;
    double *csr_values_h = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;
    double *diag_h = NULL;
    double *row_scale_h = NULL, *col_scale_h = NULL;

    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    double *csr_values_d = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;
    double *diag_d = NULL;
    double *row_scale_d = NULL, *col_scale_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    csr_columns_h = (int*)malloc(nnz * sizeof(int));
    csr_values_h = (double*)malloc(nnz * sizeof(double));
    x_values_h = (double*)malloc(nrhs * n * sizeof(double));
    b_values_h = (double*)malloc(nrhs * n * sizeof(double));

    diag_h = (double*)malloc(n * sizeof(double));
    row_scale_h = (double*)malloc(n * sizeof(double));
    col_scale_h = (double*)malloc(n * sizeof(double));

    if (!csr_offsets_h || ! csr_columns_h || !csr_values_h ||
        !x_values_h || !b_values_h || !diag_h || !row_scale_h || !col_scale_h) {
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

    i = 0;
    csr_values_h[i++] = 4.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 3.0; csr_values_h[i++] = 2.0;
    csr_values_h[i++] = 5.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 2.0;

    /* Note: Right-hand side b is initialized with values which correspond
       to the exact solution vector {1, 2, 3, 4, 5} */
    i = 0;
    // for symmetric-case
    b_values_h[i++] = 7.0;
    b_values_h[i++] = 12.0;
    b_values_h[i++] = 25.0;
    b_values_h[i++] = 4.0;
    b_values_h[i++] = 13.0;

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

    CUDA_CALL_AND_CHECK(cudaMalloc(&diag_d, n * sizeof(double)),
                        "cudaMalloc for diag");
    CUDA_CALL_AND_CHECK(cudaMalloc(&row_scale_d, n * sizeof(double)),
                        "cudaMalloc for row_scale");
    CUDA_CALL_AND_CHECK(cudaMalloc(&col_scale_d, n * sizeof(double)),
                        "cudaMalloc for col_scale");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (n + 1) * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrhs * n * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy for b_values");

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

    /* (optional) Setting algorithmic knobs */
    cudssAlgType_t reorder_alg = CUDSS_ALG_DEFAULT;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_REORDERING_ALG,
                         &reorder_alg, sizeof(cudssAlgType_t)), status, "cudssConfigSet for reordering alg");

    int ione = 1;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_USE_MATCHING,
                         &ione, sizeof(int)), status, "cudssConfigSet for matching");

    cudssAlgType_t matching_alg = CUDSS_ALG_DEFAULT; // matching with scaling, same as CUDSS_ALG_5
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_MATCHING_ALG,
                         &matching_alg, sizeof(cudssAlgType_t)), status, "cudssConfigSet for matching alg");

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
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;

    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                         csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
                         base), status, "cudssMatrixCreateCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    size_t sizeWritten;
    int perm[5];
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_REORDER_ROW, &perm,
                         sizeof(perm), &sizeWritten), status, "cudssDataGet for reorder row perm");
    for (int i = 0; i < 5; i++)
        printf("reorder row perm[%d] = %d\n", i, perm[i]);

    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_REORDER_COL, &perm,
                         sizeof(perm), &sizeWritten), status, "cudssDataGet for reorder col perm");
    for (int i = 0; i < 5; i++)
        printf("reorder col perm[%d] = %d\n", i, perm[i]);

    int used_matching = 0;
    CUDSS_CALL_AND_CHECK(cudssConfigGet(solverConfig, CUDSS_CONFIG_USE_MATCHING,
                            &used_matching, sizeof(int), &sizeWritten),
                        status, "cudssConfigGet for use_matching");

    if (used_matching) {
        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_MATCHING, &perm,
                            sizeof(perm), &sizeWritten), status, "cudssDataGet for matching perm");
        for (int i = 0; i < 5; i++)
            printf("matching (col) perm[%d] = %d\n", i, perm[i]);
    }

    /*  Note: currently these features are only supported for CUDSS_ALG_1 and CUDSS_ALG_2
        reordering algorithms. */
    if (reorder_alg == CUDSS_ALG_1 || reorder_alg == CUDSS_ALG_2) {
        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_ROW, &perm,
                             sizeof(perm), &sizeWritten), status, "cudssDataGet for row perm");
        for (int i = 0; i < 5; i++)
            printf("final row perm[%d] = %d\n", i, perm[i]);

        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_COL, &perm,
                             sizeof(perm), &sizeWritten), status, "cudssDataGet for col perm");
        for (int i = 0; i < 5; i++)
            printf("final col perm[%d] = %d\n", i, perm[i]);
    }

    int64_t memory_estimates[16] = {0};
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_MEMORY_ESTIMATES,
                                      &memory_estimates, sizeof(memory_estimates), &sizeWritten),
                              status, "cudssDataGet for the memory estimates");
    printf("memory estimates: device: %ld (stable) %ld (peak)\n",
            memory_estimates[0], memory_estimates[1]);
    printf("memory estimates: host: %ld (stable) %ld (peak)\n",
            memory_estimates[2], memory_estimates[3]);
    printf("memory estimates: hybrid peak: %ld = (GPU) %ld (CPU)\n",
            memory_estimates[4], memory_estimates[5]);
    fflush(0);

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    /* (optional) Recommended: getting runtime errors from device side
        Note: cudssDataGet is a synchronous API.
        Note: per cuDSS documentation, CUDSS_DATA_INFO is always returned
        as a pointer to int.
    */
    int info;
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_INFO, &info,
                         sizeof(info), &sizeWritten), status, "cudssDataGet for info");
    printf("cuDSS info = %d\n", info);

    int npivots;
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_NPIVOTS, &npivots,
                         sizeof(npivots), &sizeWritten), status, "cudssDataGet for npivots");
    printf("cuDSS npivots = %d\n", npivots);

    int inertia[2];
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_INERTIA, &inertia,
                         sizeof(inertia), &sizeWritten), status, "cudssDataGet for inertia");
    printf("cuDSS inertia = %d %d\n", inertia[0], inertia[1]);

    /* (optional) Recommended: getting data back when the user does not know the size
        Note: cudssDataGet is a synchronous API.
    */
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ, NULL, 0,
                         &sizeWritten), status, "cudssDataGet size request for lu_nnz");
    printf("cuDSS requests %zu bytes for returning the #nnz in the factors\n", sizeWritten);
    /* Note: per cuDSS documentation, CUDSS_DATA_LU_NNZ is always returned as a
       pointer to int64_t.
       In the general case, the user can allocate the returned #bytes and pass a
       pointer to the allocated piece of memory into the cudssDataGet API and
       reinterpret the results with correct types (defined in the documentation).
    */
    int64_t lu_nnz;
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ, &lu_nnz,
                         sizeof(int64_t), NULL), status, "cudssDataGet for lu_nnz");
    printf("cuDSS #nnz in LU = %ld\n", lu_nnz);

    /* (optional) Getting data back when the user knows the size
    */
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_DIAG, diag_d,
                         n*sizeof(double), &sizeWritten), status, "cudssDataGet for diag");
    CUDA_CALL_AND_CHECK(cudaMemcpy(diag_h, diag_d, n * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy for diag");
    for (int i = 0; i < 5; i++)
        printf("diag[%d] = %f\n", i, diag_h[i]);

    /* Note: scales are always real (never complex) and positive */
    if (matching_alg == CUDSS_ALG_DEFAULT || matching_alg == CUDSS_ALG_5) {
        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_SCALE_ROW, row_scale_d,
                             n*sizeof(double), &sizeWritten), status, "cudssDataGet for row scale");
        CUDA_CALL_AND_CHECK(cudaMemcpy(row_scale_h, row_scale_d, n * sizeof(double),
                            cudaMemcpyDeviceToHost), "cudaMemcpy for row scale");
        for (int i = 0; i < 5; i++)
            printf("row scale[%d] = %f\n", i, row_scale_h[i]);

        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_SCALE_COL, col_scale_d,
            n*sizeof(double), &sizeWritten), status, "cudssDataGet for col scale");
        CUDA_CALL_AND_CHECK(cudaMemcpy(col_scale_h, col_scale_d, n * sizeof(double),
                            cudaMemcpyDeviceToHost), "cudaMemcpy for col scale");
        for (int i = 0; i < 5; i++)
            printf("col scale[%d] = %f\n", i, col_scale_h[i]);
    }

    int iter_ref_nsteps = 2;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_IR_N_STEPS,
                         &iter_ref_nsteps, sizeof(iter_ref_nsteps)), status,
                         "cudssSolverSet for IR nsteps");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status, "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    int passed = 1;
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n", i, x_values_h[i], double(i+1));
        if (fabs(x_values_h[i] - (i + 1)) > 2.e-15)
          passed = 0;
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
