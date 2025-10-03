/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"

/*
    This example demonstrates usage of cuDSS helper APIs for
    matrix objects.
*/

#define CUDSS_EXAMPLE_FREE \
    do { \
        free(denseB_values_h); \
        cudaFree(denseB_values_d); \
        cudaFree(denseB_values_d2); \
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

/* Note: assumes column-major layout */
void print_dense_matrix(int nrows, int ncols, int ld, double *dense_values) {
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            printf("%3.3e ", dense_values[j*ld + i]);
        }
        printf("\n");
    }
}


int main (int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: creating, modifying and destroying cuDSS \n"
           "dense matrix objects\n");
    printf("---------------------------------------------------------\n");

    int major, minor, patch;
    cudssGetProperty(MAJOR_VERSION, &major);
    cudssGetProperty(MINOR_VERSION, &minor);
    cudssGetProperty(PATCH_LEVEL,   &patch);
    printf("CUDSS Version (Major,Minor,PatchLevel): %d.%d.%d\n", major, minor, patch);

    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int nrows = 5;
    int ncols = nrows;
    int ld    = nrows;

    double *denseB_values_h = NULL;

    double *denseB_values_d  = NULL;
    double *denseB_values_d2 = NULL;

    /* Allocate host memory for the dense matrix B*/

    denseB_values_h = (double*)malloc(ncols * ld * sizeof(double));

    if (!denseB_values_h) {
        printf("Error: host memory allocation failed\n");
        return -3;
    }

    /* Initialize host memory for B */
    for (int i = 0; i < nrows * ld; i++)
        denseB_values_h[i] = (i + 1) * 1.0f;

    /* Print initialized matricx data */
    printf("matrix B:\n");
    print_dense_matrix(nrows, ncols, ld, denseB_values_h);

    cudssMatrix_t B = NULL;

    /* Allocate device memory for B */
    CUDA_CALL_AND_CHECK(cudaMalloc(&denseB_values_d, ncols * ld * sizeof(double)),
                        "cudaMalloc for denseB values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&denseB_values_d2, ncols * ld * sizeof(double)),
                        "cudaMalloc for future denseB values");

    /* Copy host memory to device for B */
    CUDA_CALL_AND_CHECK(cudaMemcpy(denseB_values_d, denseB_values_h, ncols * ld * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy H2D for denseB_values");

    /* (optional for this example) Creating a library handle is strictly not necessary
       for using cuDSS matrix helper APIs, but it is done here. */
    cudssHandle_t handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* Create a matrix object for the dense matrix B. */

    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&B, nrows, ncols, ld, denseB_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");

    /* Note: cuDSS matrix objects are lightweight wrappers around the user data.*/

    int64_t nrows_out, ncols_out, ldb_out;
    void *denseB_values_out = NULL;
    cudaDataType_t denseB_type_out;
    cudssLayout_t denseB_layout_out;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetDn(B, &nrows_out, &ncols_out, &ldb_out,
                         &denseB_values_out, &denseB_type_out, &denseB_layout_out),
                         status, "cudssMatrixGetDn for B");
    if (denseB_values_out != denseB_values_d || nrows_out != nrows || ncols_out != ncols ||
        ldb_out != ld || denseB_type_out != CUDA_R_64F ||
        denseB_layout_out != CUDSS_LAYOUT_COL_MAJOR) {
        printf("Error: Example FAILED: wrong data returned for matrix B\n");
        CUDSS_EXAMPLE_FREE;
        return -4;
    } else {
        printf("Success: Check for matrix B from cudssMatrixGetDn passed\n");
    }

    /* Modifying values of the matrix B in-place */
    double new_value = -1.0;
    CUDA_CALL_AND_CHECK(cudaMemcpy(denseB_values_d, &new_value, 1 * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemset");
    CUDSS_CALL_AND_CHECK(cudssMatrixGetDn(B, NULL, NULL, NULL,
                         &denseB_values_out, NULL, NULL),
                         status, "cudssMatrixGetDn for B");
    CUDA_CALL_AND_CHECK(cudaMemcpy(denseB_values_h, denseB_values_out, ncols * ld * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy D2H for denseB_values");

    if (denseB_values_h[0] != new_value) {
        printf("Error: Example FAILED: modification of B values in-place failed\n"
               "first value of matrix B equals %3.3e (expected %3.3e)\n",
                denseB_values_h[0], new_value);
        CUDSS_EXAMPLE_FREE;
        return -5;
    } else {
        printf("Success:After modifying in-place, first value of\n"
               "matrix B equals %3.3e (expected %3.3e)\n",
                denseB_values_h[0], new_value);
        printf("matrix B (after modifying values in-place):\n");
        print_dense_matrix(nrows, ncols, ld, denseB_values_h);
    }

    /* Replacing values of the matrix B with new data */
    new_value = -4.0;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&denseB_values_d2[1], &new_value, 1 * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemset");

    CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(B, denseB_values_d2),
                         status, "cudssMatrixSetValues for B");
    CUDSS_CALL_AND_CHECK(cudssMatrixGetDn(B, NULL, NULL, NULL,
                         &denseB_values_out, NULL, NULL),
                         status, "cudssMatrixGetDn for B");
    CUDA_CALL_AND_CHECK(cudaMemcpy(denseB_values_h, denseB_values_out, ncols * ld * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy D2H for denseB_values");
    if (denseB_values_h[1] != new_value) {
        printf("Error: Example FAILED: modification of B values out-of-place failed\n"
               "second value of matrix B equals %3.3e (expected %3.3e)\n",
                denseB_values_h[1], new_value);
        CUDSS_EXAMPLE_FREE;
        return -6;
    } else {
        printf("Success: After modifying in-place, second value of\n"
               "matrix B equals %3.3e (expected %3.3e)\n",
                denseB_values_h[1], new_value);
        printf("matrix B (after replacing the values array):\n");
        print_dense_matrix(nrows, ncols, ld, denseB_values_h);
    }

    /* Demonstrating cudssMatrixGetFormat */
    int format;

    CUDSS_CALL_AND_CHECK(cudssMatrixGetFormat(B, &format),
                         status, "cudssMatrixGetFormat for B");
    if (!(format & CUDSS_MFORMAT_DENSE)) {
        printf("Error: Example FAILED: wrong format = %d was returned for B\n",
                format);
        CUDSS_EXAMPLE_FREE;
        return -7;
    } else
        printf("Success: Matrix B format returned as %d (CSR)\n", format);

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(B), status, "cudssMatrixDestroy for B");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

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