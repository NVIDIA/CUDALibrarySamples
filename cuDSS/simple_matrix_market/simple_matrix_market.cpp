/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Rémi Bourgeois. All rights reserved.
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
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cudss.h"
#include "matrix_market_reader.h"

/*
    This example demonstrates basic usage of cuDSS APIs for solving
    a system of linear algebraic equations with a sparse matrix
    coming from a matrix market file:
           Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).

    Example usage from build directory
    ./simple_matrix_market_example 0 general full ../matrix.mtx ../rhs.mtx
*/

double compute_residual_error(int n, const int *csr_offsets_h, const int *csr_columns_h,
                              const double *csr_values_h, const double *x_values_h,
                              const double *b_values_h, cudssMatrixViewType_t mview) {
    std::vector<double> Ax(n, 0.0);

    for (int row = 0; row < n; ++row) {
        for (int idx = csr_offsets_h[row]; idx < csr_offsets_h[row + 1]; ++idx) {
            int    col = csr_columns_h[idx];
            double val = csr_values_h[idx];

            switch (mview) {
            case CUDSS_MVIEW_FULL:
                Ax[row] += val * x_values_h[col];
                break;

            case CUDSS_MVIEW_UPPER:
                if (col >= row) {
                    Ax[row] += val * x_values_h[col];
                    if (col != row)
                        Ax[col] += val * x_values_h[row];
                }
                break;

            case CUDSS_MVIEW_LOWER:
                if (col <= row) {
                    Ax[row] += val * x_values_h[col];
                    if (col != row)
                        Ax[col] += val * x_values_h[row];
                }
                break;
            }
        }
    }

    // Compute L2 norm of residual
    double error = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff  = Ax[i] - b_values_h[i];
        error       += diff * diff;
    }

    return std::sqrt(error);
}

#define CUDSS_EXAMPLE_FREE                                                               \
    do {                                                                                 \
        if (csr_offsets_h)                                                               \
            free(csr_offsets_h);                                                         \
        if (csr_columns_h)                                                               \
            free(csr_columns_h);                                                         \
        if (csr_values_h)                                                                \
            free(csr_values_h);                                                          \
        free(x_values_h);                                                                \
        free(b_values_h);                                                                \
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

int main(int argc, char *argv[]) {

    cudaError_t   cuda_error = cudaSuccess;
    cudssStatus_t status     = CUDSS_STATUS_SUCCESS;

    /*timers*/
    cudaEvent_t start, stop;
    float       time_ms = 0.0f, total_ms = 0.0f;
    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n;
    int nnz;
    int nrhs = 1;

    if (argc < 5) {
        fprintf(
            stderr,
            "Usage: %s <alg: 0|1|2|3> <mtype: general|symmetric|hermitian|spd|hpd> "
            "<mview: full|lower|upper> <matrix_filename> [vector_filename (optional)]\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    // Parse algorithm
    int            alg_input = std::atoi(argv[1]);
    cudssAlgType_t reorder_alg;
    switch (alg_input) {
    case 0:
        reorder_alg = CUDSS_ALG_DEFAULT;
        break;
    case 1:
        reorder_alg = CUDSS_ALG_1;
        break;
    case 2:
        reorder_alg = CUDSS_ALG_2;
        break;
    case 3:
        reorder_alg = CUDSS_ALG_3;
        break;
    default:
        fprintf(stderr, "Error: Invalid algorithm '%s' (must be 0-3)\n", argv[1]);
        return EXIT_FAILURE;
    }

    // Parse matrix type
    cudssMatrixType_t mtype;
    if (strcmp(argv[2], "general") == 0)
        mtype = CUDSS_MTYPE_GENERAL;
    else if (strcmp(argv[2], "symmetric") == 0)
        mtype = CUDSS_MTYPE_SYMMETRIC;
    else if (strcmp(argv[2], "hermitian") == 0)
        mtype = CUDSS_MTYPE_HERMITIAN;
    else if (strcmp(argv[2], "spd") == 0)
        mtype = CUDSS_MTYPE_SPD;
    else if (strcmp(argv[2], "hpd") == 0)
        mtype = CUDSS_MTYPE_HPD;
    else {
        fprintf(stderr,
                "Error: Invalid matrix type '%s'. Supported types are general, "
                "symmetric, hermitian, spd, hpd\n",
                argv[2]);
        return EXIT_FAILURE;
    }

    if (((reorder_alg == 1) || (reorder_alg == 2)) && (mtype != CUDSS_MTYPE_GENERAL)) {
        fprintf(stderr,
                "Warining: Invalid algorithm. CUDSS_ALG_1 and CUDSS_ALG_2 are only "
                "supported for general (non-symmetric or non-hermitian) matrices.\n"
                "See https://docs.nvidia.com/cuda/cudss/types.html#cudssconfigparam-t\n"
                "Expect a large error.\n");
    }

    // Parse matrix view
    cudssMatrixViewType_t mview;
    if (strcmp(argv[3], "full") == 0)
        mview = CUDSS_MVIEW_FULL;
    else if (strcmp(argv[3], "lower") == 0)
        mview = CUDSS_MVIEW_LOWER;
    else if (strcmp(argv[3], "upper") == 0)
        mview = CUDSS_MVIEW_UPPER;
    else {
        fprintf(stderr,
                "Error: Invalid matrix view type '%s'. Supported types are: full, lower, "
                "upper.\n",
                argv[3]);
        return EXIT_FAILURE;
    }

    if ((mview != CUDSS_MVIEW_FULL) && (mtype == CUDSS_MTYPE_GENERAL)) {
        fprintf(stderr, "Error: you chose a lower/upper view of the matrix but you also "
                        "specified that it is general (not symmetric).\n"
                        "If your matrix file is truly non-symmetric, half of the "
                        "elements would not be used, as the lower/upper part of the\n"
                        "matrix will be mirrored and you will be solving for the wrong "
                        "matrix. Or the reader will throw an error.\n");
        return EXIT_FAILURE;
    }

    // Matrix and optional vector filenames
    const char *matrix_filename = argv[4];
    const char *vector_filename = (argc > 5) ? argv[5] : nullptr;

    int    *csr_offsets_h = NULL;
    int    *csr_columns_h = NULL;
    double *csr_values_h  = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int    *csr_offsets_d = NULL;
    int    *csr_columns_d = NULL;
    double *csr_values_d  = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    /* Read input matrix from file and allocate host memory accordingly */
    int failed = matrix_reader(matrix_filename, n, nnz, &csr_offsets_h, &csr_columns_h,
                               &csr_values_h, mview);
    if (failed) {
        fprintf(stderr, "Reader failed.\n");
        return EXIT_FAILURE;
    }

    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear %dx%d system from file \"%s\"\n", n, n,
           matrix_filename);
    printf("---------------------------------------------------------\n");

    /* Allocate host memory for solution x*/
    x_values_h = (double *)malloc(nrhs * n * sizeof(double));

    /* Allocate host memory for right hand side b and fill it*/
    /* Read from file if rhs file is provided */
    if (vector_filename != nullptr) {
        int failed = rhs_reader(vector_filename, n, &b_values_h);
        if (failed) {
            fprintf(stderr, "Reader failed.\n");
            return EXIT_FAILURE;
        }
    } else {
        printf("No rhs file provided, filling b with 1.0");
        b_values_h = (double *)malloc(nrhs * n * sizeof(double));
        for (int i = 0; i < n; i++) {
            b_values_h[i] = 1.0;
        }
    }
    if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h || !b_values_h) {
        fprintf(stderr, "Error: host memory allocation failed\n");
        return -1;
    }

    /* Allocate device memory for A, x and b */
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
    cudssData_t   solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");

    /* (optional) Setting algorithmic knobs */
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_REORDERING_ALG,
                                        &reorder_alg, sizeof(cudssAlgType_t)),
                         status, "cudssSolverSet");

    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices).
     */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int     ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;

    printf("--- Start solving--- \n");
    cudaEventRecord(start);
    cudssIndexBase_t base = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDA_R_32I,
                                              CUDA_R_64F, mtype, mview, base),
                         status, "cudssMatrixCreateCsr");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("cuDSS MatrixCreateCsr time: %.4f ms\n", time_ms);
    total_ms += time_ms;

    /* Symbolic factorization */
    cudaEventRecord(start);
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b),
        status, "cudssExecute for analysis");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("cuDSS Analysis time: %.4f ms\n", time_ms);
    total_ms += time_ms;

    /* Factorization */
    cudaEventRecord(start);
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b),
                         status, "cudssExecute for factor");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("cuDSS Factorization time: %.4f ms\n", time_ms);
    total_ms += time_ms;

    /* Solving */
    cudaEventRecord(start);
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b),
        status, "cudssExecute for solve");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("cuDSS Solve time: %.4f ms\n", time_ms);
    total_ms += time_ms;


    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x_values");

    printf("--- Solving complete ! --- \n");
    printf("cuDSS Total time: %.4f ms\n", total_ms);

    double residual = compute_residual_error(n, csr_offsets_h, csr_columns_h,
                                             csr_values_h, x_values_h, b_values_h, mview);

    printf("Residual L2 error ||Ax - b|| = %e\n", residual);
    bool passed = (residual < 1e-5);

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status,
                         "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS && passed) {
        printf("Example PASSED\n");
        return 0;
    } else {
        printf("Example FAILED\n");
        return -1;
    }
}
