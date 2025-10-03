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

#include "cudss.h"

/*
    This example demonstrates basic usage of cuDSS APIs for solving
    a system of linear algebraic equations with a sparse matrix:
                                Ax = b,
    using multiple devices, where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
*/

#define CUDSS_EXAMPLE_FREE                                                               \
    do {                                                                                 \
        free(csr_offsets_h);                                                             \
        free(csr_columns_h);                                                             \
        free(csr_values_h);                                                              \
        free(x_values_h);                                                                \
        free(b_values_h);                                                                \
        free(device_indices);                                                            \
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

    /* Query the actual number of available devices */
    int device_count = 0;
    cuda_error       = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess || device_count <= 0) {
        printf("ERROR: no GPU devices found\n");
        fflush(0);
        return -1;
    }

    /* device_indices can be set to NULL. In that cases cuDSS will take devices
     *     from 0 to (device_count - 1)
     */
    int *device_indices = NULL;
    device_indices      = (int *)malloc(device_count * sizeof(int));
    if (device_indices == NULL) {
        printf("ERROR: failed to allocate host memory\n");
        fflush(0);
        return -1;
    }
    for (int i = 0; i < device_count; i++)
        device_indices[i] = i;

    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear 5x5 system\n"
           "with a symmetric positive-definite matrix with %d devices\n",
           device_count);
    printf("---------------------------------------------------------\n");

    int n    = 5;
    int nnz  = 8;
    int nrhs = 1;

    int    *csr_offsets_h = NULL;
    int    *csr_columns_h = NULL;
    double *csr_values_h  = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int    *csr_offsets_d = NULL;
    int    *csr_columns_d = NULL;
    double *csr_values_d  = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csr_offsets_h = (int *)malloc((n + 1) * sizeof(int));
    csr_columns_h = (int *)malloc(nnz * sizeof(int));
    csr_values_h  = (double *)malloc(nnz * sizeof(double));
    x_values_h    = (double *)malloc(nrhs * n * sizeof(double));
    b_values_h    = (double *)malloc(nrhs * n * sizeof(double));

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
    b_values_h[i++] = 7.0;
    b_values_h[i++] = 12.0;
    b_values_h[i++] = 25.0;
    b_values_h[i++] = 4.0;
    b_values_h[i++] = 13.0;

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

    /* Initialize cudss handle for multiple devices */
    CUDSS_CALL_AND_CHECK(cudssCreateMg(&handle, device_count, device_indices), status,
                         "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t   solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");

    /* Pass same device_count and device_indices to solverConfig */
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_DEVICE_COUNT,
                                        &device_count, sizeof(device_count)),
                         status, "cudssConfigSet for device_count");

    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_DEVICE_INDICES,
                                        device_indices, device_count * sizeof(int)),
                         status, "cudssConfigSet for device_count");


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
    cudssMatrix_t         A;
    cudssMatrixType_t     mtype = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t      base  = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDA_R_32I,
                                              CUDA_R_64F, mtype, mview, base),
                         status, "cudssMatrixCreateCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b),
        status, "cudssExecute for analysis");

    /* Query CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN should be done for each device
     * separately by calling cudaSetDevice() prior to cudssDataGet.
     * Same for getting CUDSS_DATA_MEMORY_ESTIMATES.
     * Same for setting CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT with cudssConfigSet()
     */
    int default_device = 0;
    cudaGetDevice(&default_device);
    for (int dev_id = 0; dev_id < device_count; dev_id++) {
        cudaSetDevice(device_indices[dev_id]);

        int64_t hybrid_device_memory_limit = 0;
        size_t  sizeWritten;
        CUDSS_CALL_AND_CHECK(
            cudssDataGet(handle, solverData, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
                         &hybrid_device_memory_limit, sizeof(hybrid_device_memory_limit),
                         &sizeWritten),
            status, "cudssDataGet for the memory estimates");

        printf("dev_id = %d CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN %ld bytes\n",
               device_indices[dev_id], hybrid_device_memory_limit);
    }
    /* cuDSS requires all API calls to be made on the default device, so
     * resseting device context.
     */
    cudaSetDevice(default_device);

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b),
                         status, "cudssExecute for factor");

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
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x_values");

    int passed = 1;
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %1.4f expected %1.4f\n", i, x_values_h[i], double(i + 1));
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