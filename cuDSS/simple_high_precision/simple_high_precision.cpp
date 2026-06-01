/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "cudss.h"

/*
    This example demonstrates basic usage of cuDSS APIs for solving
    a high-precision linear system with a sparse matrix:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix),
    and A, b, and x use cudss_fp64mp2_t values.
    Note: This path requires device compute capability 9.0 or higher.
*/

#define CUDSS_EXAMPLE_FREE                                                               \
    do {                                                                                 \
        free(csr_offsets_h);                                                             \
        free(csr_columns_h);                                                             \
        free(csr_values_h);                                                              \
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


cudss_fp64mp2_t to_cudss_fp64mp2_t(double x) {
    return cudss_fp64mp2_t{x, 0.0};
}

double dabs(cudss_fp64mp2_t x) {
    return std::abs(x.hi + x.lo);
}

// Computes a + b with high precision
cudss_fp64mp2_t high_precision_add(cudss_fp64mp2_t a, cudss_fp64mp2_t b) {
    // First, add a.hi and b.hi without knowing which component is larger
    const double res_hi       = a.hi + b.hi;
    const double ah_prime     = res_hi - b.hi;
    const double bh_prime     = res_hi - ah_prime;
    const double delta_a      = a.hi - ah_prime;
    const double delta_b      = b.hi - bh_prime;
    const double res_hi_error = delta_a + delta_b;

    // Now, add the low parts
    const double res_lo = a.lo + b.lo + res_hi_error;

    // Finally, we need to normalize the result
    cudss_fp64mp2_t normalized_result{0, 0};

    normalized_result.hi = res_hi + res_lo;
    const double diff    = normalized_result.hi - res_hi;
    normalized_result.lo = res_lo - diff;

    return normalized_result;
}

// Computes a - b with high precision
cudss_fp64mp2_t high_precision_sub(cudss_fp64mp2_t a, cudss_fp64mp2_t b) {
    return high_precision_add(a, cudss_fp64mp2_t{-b.hi, -b.lo});
}


int main(int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear 5x5 system\n"
           "with a symmetric positive-definite matrix in high precision\n");
    printf("---------------------------------------------------------\n");
    cudaError_t   cuda_error = cudaSuccess;
    cudssStatus_t status     = CUDSS_STATUS_SUCCESS;

    int device = -1;
    cuda_error = cudaGetDevice(&device);
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaGetDevice failed with error %d\n", cuda_error);
        return -1;
    }
    cudaDeviceProp device_prop;
    cuda_error = cudaGetDeviceProperties(&device_prop, 0);
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaGetDeviceProperties failed with error %d\n", cuda_error);
        return -1;
    }
    int device_cc = device_prop.major * 10 + device_prop.minor;

    if (device_cc < 90) {
        printf("Example SKIPPED: The device compute capability is less than 9.0 "
               "(device_cc = %d < 90)\n",
               device_cc);
        return 0;
    }

    const int n    = 5;
    const int nnz  = 8;
    const int nrhs = 1;

    int             *csr_offsets_h = NULL;
    int             *csr_columns_h = NULL;
    cudss_fp64mp2_t *csr_values_h  = NULL;
    cudss_fp64mp2_t *x_values_h = NULL, *b_values_h = NULL;

    int             *csr_offsets_d = NULL;
    int             *csr_columns_d = NULL;
    cudss_fp64mp2_t *csr_values_d  = NULL;
    cudss_fp64mp2_t *x_values_d = NULL, *b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csr_offsets_h = (int *)malloc((n + 1) * sizeof(int));
    csr_columns_h = (int *)malloc(nnz * sizeof(int));
    csr_values_h  = (cudss_fp64mp2_t *)malloc(nnz * sizeof(cudss_fp64mp2_t));
    x_values_h    = (cudss_fp64mp2_t *)malloc(nrhs * n * sizeof(cudss_fp64mp2_t));
    b_values_h    = (cudss_fp64mp2_t *)malloc(nrhs * n * sizeof(cudss_fp64mp2_t));

    if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h || !b_values_h) {
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
    csr_values_h[i++] = to_cudss_fp64mp2_t(4.0);
    csr_values_h[i++] = to_cudss_fp64mp2_t(1.0);
    csr_values_h[i++] = to_cudss_fp64mp2_t(3.0);
    csr_values_h[i++] = to_cudss_fp64mp2_t(2.0);
    csr_values_h[i++] = to_cudss_fp64mp2_t(5.0);
    csr_values_h[i++] = to_cudss_fp64mp2_t(1.0);
    csr_values_h[i++] = to_cudss_fp64mp2_t(1.0);
    csr_values_h[i++] = to_cudss_fp64mp2_t(2.0);


    const cudss_fp64mp2_t expected_solution[5] = {
        cudss_fp64mp2_t{1.0, 1.0e-20}, cudss_fp64mp2_t{2.0, -1.0e-23},
        cudss_fp64mp2_t{3.0, 0.0}, cudss_fp64mp2_t{4.0, 0.0}, cudss_fp64mp2_t{5.0, 0.0}};

    /* Note: b is set to the solution of A * expected_solution */
    i               = 0;
    b_values_h[i++] = cudss_fp64mp2_t{7.0, 4.0e-20};
    b_values_h[i++] = cudss_fp64mp2_t{12.0, -3.0e-23};
    b_values_h[i++] = cudss_fp64mp2_t{25.0, 1.0e-20 - 2.0e-23};
    b_values_h[i++] = cudss_fp64mp2_t{4.0, 0.0};
    b_values_h[i++] = cudss_fp64mp2_t{13.0, 0.0};

    /* Allocate device memory for A, x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                        "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(cudss_fp64mp2_t)),
                        "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * n * sizeof(cudss_fp64mp2_t)),
                        "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(cudss_fp64mp2_t)),
                        "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (n + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h,
                                   nnz * sizeof(cudss_fp64mp2_t), cudaMemcpyHostToDevice),
                        "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h,
                                   nrhs * n * sizeof(cudss_fp64mp2_t),
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
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");
    int deterministicMode = 0;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_DETERMINISTIC_MODE,
                                        &deterministicMode, sizeof(deterministicMode)),
                         status, "cudssConfigSet for deterministic mode");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices).
     */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int     ldb = nrows, ldx = ncols;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, nrows, nrhs, ldb, b_values_d,
                                             CUDSS_R_64F_64F, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, ncols, nrhs, ldx, x_values_d,
                                             CUDSS_R_64F_64F, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t         A;
    cudssMatrixType_t     mtype = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t      base  = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDSS_R_32I,
                                              CUDSS_R_32I, CUDSS_R_64F_64F, mtype, mview,
                                              base),
                         status, "cudssMatrixCreateCsr");

    /* Reordering and symbolic factorization */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b),
        status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b),
                         status, "cudssExecute for factor");

    /* Solve */
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

    /* Copy solution x to host and validate against the expected solution */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d,
                                   nrhs * n * sizeof(cudss_fp64mp2_t),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x_values");

    constexpr double PASS_THRESHOLD = 1.e-30;
    int              passed         = 1;
    for (int i = 0; i < n; i++) {
        const double delta =
            dabs(high_precision_sub(x_values_h[i], expected_solution[i]));
        printf("x[%d] = (%1.14e, %+1.14e) expected (%1.14e, %+1.14e); delta = %1.14e\n",
               i, x_values_h[i].hi, x_values_h[i].lo, expected_solution[i].hi,
               expected_solution[i].lo, delta);
        if (delta > PASS_THRESHOLD) {
            printf("FAILED validation: solution error at x[%d] exceeds example pass "
                   "threshold (delta = %1.14e > %1.14e)\n",
                   i, delta, PASS_THRESHOLD);
            passed = 0;
        }
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
