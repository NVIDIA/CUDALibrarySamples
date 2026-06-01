/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <type_traits>

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

#define CUDSS_EXAMPLE_FREE                                                               \
    do {                                                                                 \
        free(csr_offsets_h);                                                             \
        free(csr_columns_h);                                                             \
        free(csr_values_h);                                                              \
        free(x_values_h);                                                                \
        free(b_values_h);                                                                \
        free(diag_h);                                                                    \
        free(row_scale_h);                                                               \
        free(col_scale_h);                                                               \
        cudaFree(csr_offsets_d);                                                         \
        cudaFree(csr_columns_d);                                                         \
        cudaFree(csr_values_d);                                                          \
        cudaFree(x_values_d);                                                            \
        cudaFree(b_values_d);                                                            \
        cudaFree(diag_d);                                                                \
        cudaFree(row_scale_d);                                                           \
        cudaFree(col_scale_d);                                                           \
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


int main(int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear 5x5 system\n"
           "with a symmetric positive-definite matrix\n"
           "with extra settings and extra information retrieved\n");
    printf("---------------------------------------------------------\n");

    // This example supports index_type = {int, int64_t} and
    // value_type = {float, double}
    using index_type = int;
    using value_type = double;

    constexpr cudssDataType_t cuda_error_type = CUDSS_DATA_TYPE_UNSET;

    // These values are adapted to index_type and value_type
    const cudssDataType_t cuda_index_type =
        std::is_same<index_type, int>::value
            ? CUDSS_R_32I
            : (std::is_same<index_type, int64_t>::value ? CUDSS_R_64I : cuda_error_type);
    const cudssDataType_t cuda_value_type =
        std::is_same<value_type, double>::value
            ? CUDSS_R_64F
            : (std::is_same<value_type, float>::value ? CUDSS_R_32F : cuda_error_type);

    int major, minor, patch;
    cudssGetProperty(MAJOR_VERSION, &major);
    cudssGetProperty(MINOR_VERSION, &minor);
    cudssGetProperty(PATCH_LEVEL, &patch);
    printf("CUDSS Version (Major,Minor,PatchLevel): %d.%d.%d\n", major, minor, patch);

    cudaError_t   cuda_error = cudaSuccess;
    cudssStatus_t status     = CUDSS_STATUS_SUCCESS;

    int     n    = 5;
    int64_t nnz  = 8;
    int     nrhs = 1;

    index_type *csr_offsets_h = NULL;
    index_type *csr_columns_h = NULL;
    value_type *csr_values_h  = NULL;
    value_type *x_values_h = NULL, *b_values_h = NULL;
    value_type *diag_h      = NULL;
    value_type *row_scale_h = NULL, *col_scale_h = NULL;

    index_type *csr_offsets_d = NULL;
    index_type *csr_columns_d = NULL;
    value_type *csr_values_d  = NULL;
    value_type *x_values_d = NULL, *b_values_d = NULL;
    value_type *diag_d      = NULL;
    value_type *row_scale_d = NULL, *col_scale_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csr_offsets_h = (index_type *)malloc((n + 1) * sizeof(index_type));
    csr_columns_h = (index_type *)malloc(nnz * sizeof(index_type));
    csr_values_h  = (value_type *)malloc(nnz * sizeof(value_type));
    x_values_h    = (value_type *)malloc(nrhs * n * sizeof(value_type));
    b_values_h    = (value_type *)malloc(nrhs * n * sizeof(value_type));

    diag_h      = (value_type *)malloc(n * sizeof(value_type));
    row_scale_h = (value_type *)malloc(n * sizeof(value_type));
    col_scale_h = (value_type *)malloc(n * sizeof(value_type));

    if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h || !b_values_h ||
        !diag_h || !row_scale_h || !col_scale_h) {
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

    /* Allocate device memory for A, x, b and diag (for future use) */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(index_type)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(index_type)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(value_type)),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * n * sizeof(value_type)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(value_type)),
                        "cudaMalloc for x_values");

    CUDA_CALL_AND_CHECK(cudaMalloc(&diag_d, n * sizeof(value_type)),
                        "cudaMalloc for diag");
    CUDA_CALL_AND_CHECK(cudaMalloc(&row_scale_d, n * sizeof(value_type)),
                        "cudaMalloc for row_scale");
    CUDA_CALL_AND_CHECK(cudaMalloc(&col_scale_d, n * sizeof(value_type)),
                        "cudaMalloc for col_scale");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h,
                                   (n + 1) * sizeof(index_type), cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(index_type),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(value_type),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrhs * n * sizeof(value_type),
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

    /* (optional) Setting algorithmic knobs */
    cudssReorderingAlg_t reorder_alg = CUDSS_REORDERING_ALG_DEFAULT;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_REORDERING_ALG,
                                        &reorder_alg, sizeof(cudssReorderingAlg_t)),
                         status, "cudssConfigSet for reordering alg");

    cudssMatchingAlg_t matching_alg =
        CUDSS_MATCHING_ALG_AUTO; // matching with scaling, same as CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_MATCHING_ALG,
                                        &matching_alg, sizeof(cudssMatchingAlg_t)),
                         status, "cudssConfigSet for matching alg");

    int use_superpanel = 0; // Default is 1, setting it to 0 here as an example
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_USE_SUPERPANELS,
                                        &use_superpanel, sizeof(int)),
                         status, "cudssConfigSet for use_superpanel");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices).
     */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int     ldb = nrows, ldx = ncols;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, nrows, nrhs, ldb, b_values_d,
                                             cuda_value_type, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, ncols, nrhs, ldx, x_values_d,
                                             cuda_value_type, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t         A;
    cudssMatrixType_t     mtype = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t      base  = CUDSS_BASE_ZERO;

    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d,
                                              cuda_index_type, cuda_index_type,
                                              cuda_value_type, mtype, mview, base),
                         status, "cudssMatrixCreateCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b),
        status, "cudssExecute for analysis");

    size_t     sizeWritten;
    constexpr int  nperm = 5;
    index_type perm[nperm];
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_REORDER_ROW,
                                      &perm, sizeof(perm), &sizeWritten),
                         status, "cudssDataGet for reorder row perm");
    for (int i = 0; i < nperm; i++)
        printf("reorder row perm[%d] = %d\n", i, int(perm[i]));

    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_REORDER_COL,
                                      &perm, sizeof(perm), &sizeWritten),
                         status, "cudssDataGet for reorder col perm");
    for (int i = 0; i < nperm; i++)
        printf("reorder col perm[%d] = %d\n", i, int(perm[i]));

    CUDSS_CALL_AND_CHECK(cudssConfigGet(solverConfig, CUDSS_CONFIG_MATCHING_ALG,
                                        &matching_alg, sizeof(cudssMatchingAlg_t), &sizeWritten),
                            status, "cudssConfigGet for matching_alg");
    int used_matching = matching_alg != CUDSS_MATCHING_ALG_NONE;

    if (used_matching) {
        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_MATCHING,
                                          &perm, sizeof(perm), &sizeWritten),
                             status, "cudssDataGet for matching perm");
        for (int i = 0; i < nperm; i++)
            printf("matching (col) perm[%d] = %d\n", i, int(perm[i]));
    }

    /*  Note: currently these features are only supported for CUDSS_REORDERING_ALG_BTF_COLAMD and CUDSS_REORDERING_ALG_COLAMD
        reordering algorithms. */
    if (reorder_alg == CUDSS_REORDERING_ALG_BTF_COLAMD || reorder_alg == CUDSS_REORDERING_ALG_COLAMD) {
        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_ROW, &perm,
                                          sizeof(perm), &sizeWritten),
                             status, "cudssDataGet for row perm");
        for (int i = 0; i < nperm; i++)
            printf("final row perm[%d] = %d\n", i, int(perm[i]));

        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_PERM_COL, &perm,
                                          sizeof(perm), &sizeWritten),
                             status, "cudssDataGet for col perm");
        for (int i = 0; i < nperm; i++)
            printf("final col perm[%d] = %d\n", i, int(perm[i]));
    }

    int64_t memory_estimates[16] = {0};
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_MEMORY_ESTIMATES,
                                      &memory_estimates, sizeof(memory_estimates),
                                      &sizeWritten),
                         status, "cudssDataGet for the memory estimates");
    printf("memory estimates: device: %ld (stable) %ld (peak)\n", memory_estimates[0],
           memory_estimates[1]);
    printf("memory estimates: host: %ld (stable) %ld (peak)\n", memory_estimates[2],
           memory_estimates[3]);
    printf("memory estimates: hybrid peak: %ld = (GPU) %ld (CPU)\n", memory_estimates[4],
           memory_estimates[5]);
    fflush(0);

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b),
                         status, "cudssExecute for factor");

    /* (optional) Recommended: getting runtime errors from device side
        Note: cudssDataGet is a synchronous API.
        Note: per cuDSS documentation, CUDSS_DATA_INFO is always returned
        as a pointer to int.
    */
    int info;
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_INFO, &info,
                                      sizeof(info), &sizeWritten),
                         status, "cudssDataGet for info");
    printf("cuDSS info = %d\n", info);

    index_type npivots;
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_NPIVOTS, &npivots,
                                      sizeof(npivots), &sizeWritten),
                         status, "cudssDataGet for npivots");
    printf("cuDSS npivots = %d\n", int(npivots));

    index_type inertia[2];
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_INERTIA, &inertia,
                                      sizeof(inertia), &sizeWritten),
                         status, "cudssDataGet for inertia");
    printf("cuDSS inertia = %d %d\n", int(inertia[0]), int(inertia[1]));

    /* (optional) Recommended: getting data back when the user does not know the size
        Note: cudssDataGet is a synchronous API.
    */
    CUDSS_CALL_AND_CHECK(
        cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ, NULL, 0, &sizeWritten),
        status, "cudssDataGet size request for lu_nnz");
    printf("cuDSS requests %zu bytes for returning the #nnz in the factors\n",
           sizeWritten);
    /* Note: per cuDSS documentation, CUDSS_DATA_LU_NNZ is always returned as a
       pointer to int64_t.
       In the general case, the user can allocate the returned #bytes and pass a
       pointer to the allocated piece of memory into the cudssDataGet API and
       reinterpret the results with correct types (defined in the documentation).
    */
    int64_t lu_nnz;
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_LU_NNZ, &lu_nnz,
                                      sizeof(int64_t), NULL),
                         status, "cudssDataGet for lu_nnz");
    printf("cuDSS #nnz in LU = %ld\n", lu_nnz);

    /* (optional) Getting data back when the user knows the size
     */
    CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_DIAG, diag_d,
                                      n * sizeof(value_type), &sizeWritten),
                         status, "cudssDataGet for diag");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(diag_h, diag_d, n * sizeof(value_type), cudaMemcpyDeviceToHost),
        "cudaMemcpy for diag");
    for (int i = 0; i < n; i++)
        printf("diag[%d] = %f\n", i, double(diag_h[i]));

    /* Note: scales are always real (never complex) and positive */
    if (matching_alg == CUDSS_MATCHING_ALG_AUTO || matching_alg == CUDSS_MATCHING_ALG_MAX_DIAG_PRODUCT) {
        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_SCALE_ROW,
                                          row_scale_d, n * sizeof(value_type),
                                          &sizeWritten),
                             status, "cudssDataGet for row scale");
        CUDA_CALL_AND_CHECK(cudaMemcpy(row_scale_h, row_scale_d, n * sizeof(value_type),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy for row scale");
        for (int i = 0; i < n; i++)
            printf("row scale[%d] = %f\n", i, double(row_scale_h[i]));

        CUDSS_CALL_AND_CHECK(cudssDataGet(handle, solverData, CUDSS_DATA_SCALE_COL,
                                          col_scale_d, n * sizeof(value_type),
                                          &sizeWritten),
                             status, "cudssDataGet for col scale");
        CUDA_CALL_AND_CHECK(cudaMemcpy(col_scale_h, col_scale_d, n * sizeof(value_type),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy for col scale");
        for (int i = 0; i < n; i++)
            printf("col scale[%d] = %f\n", i, double(col_scale_h[i]));
    }

    int iter_ref_nsteps = 2;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_IR_N_STEPS,
                                        &iter_ref_nsteps, sizeof(iter_ref_nsteps)),
                         status, "cudssSolverSet for IR nsteps");

    /* Solving */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b),
        status, "cudssExecute for solve");

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status,
                         "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(value_type),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x_values");

    int passed = 1;
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n", i, double(x_values_h[i]), double(i + 1));
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
