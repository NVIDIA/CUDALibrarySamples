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
