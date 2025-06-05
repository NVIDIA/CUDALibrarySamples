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
        free(csrA_offsets_h); \
        free(csrA_columns_h); \
        free(csrA_values_h); \
        free(csrA_offsets_h2); \
        free(csrA_columns_h2); \
        free(csrA_values_h2); \
        cudaFree(csrA_offsets_d); \
        cudaFree(csrA_columns_d); \
        cudaFree(csrA_values_d); \
        cudaFree(csrA_offsets_d2); \
        cudaFree(csrA_columns_d2); \
        cudaFree(csrA_values_d2); \
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

void print_sparse_matrix(int nrows, int *csr_offsets, int *csr_columns, double *csr_values) {
    for (int i = 0; i < nrows; i++) {
        for (int j = csr_offsets[i]; j < csr_offsets[i+1]; j++) {
            printf("(%d, %3.3e) ", csr_columns[j], csr_values[j]);
        }
        printf("\n");
    }
}


int main (int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: creating, modifying and destroying cuDSS \n"
           "sparse matrix objects\n");
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
    int nnz   = 9;
    int nnz2  = 5;

    int *csrA_offsets_h    = NULL;
    int *csrA_columns_h    = NULL;
    double *csrA_values_h  = NULL;

    int *csrA_offsets_h2   = NULL;
    int *csrA_columns_h2   = NULL;
    double *csrA_values_h2 = NULL;

    int *csrA_offsets_d    = NULL;
    int *csrA_columns_d    = NULL;
    double *csrA_values_d  = NULL;

    int *csrA_offsets_d2   = NULL;
    int *csrA_columns_d2   = NULL;
    double *csrA_values_d2 = NULL;

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csrA_offsets_h = (int*)malloc((nrows + 1) * sizeof(int));
    csrA_columns_h = (int*)malloc(nnz * sizeof(int));
    csrA_values_h = (double*)malloc(nnz * sizeof(double));

    csrA_offsets_h2 = (int*)malloc((nrows + 1) * sizeof(int));
    csrA_columns_h2 = (int*)malloc(nnz2 * sizeof(int));
    csrA_values_h2  = (double*)malloc(nnz2 * sizeof(double));

    if (!csrA_offsets_h  || ! csrA_columns_h  || !csrA_values_h ||
        !csrA_offsets_h2 || ! csrA_columns_h2 || !csrA_values_h2) {
        printf("Error: host memory allocation failed\n");
        return -3;
    }

    /* Initialize host memory for A */
    csrA_offsets_h[0] = 0;
    for (int i = 0; i < nrows; i++)
        csrA_offsets_h[i + 1] = csrA_offsets_h[i] + (i == nrows - 1 ? 1 : 2);

    for (int i = 0; i < nrows; i++) {
        for (int j = csrA_offsets_h[i]; j < csrA_offsets_h[i + 1]; j++) {
            if (i != nrows - 1) {
                csrA_columns_h[j] = i;
                csrA_columns_h[j+1] = i + 1;
            } else
                csrA_columns_h[j] = i;
        }
    }

    for (int i = 0; i < nnz; i++)
        csrA_values_h[i] = i * 2.0f;

    csrA_offsets_h2[0] = 0;
    for (int i = 0; i < nrows; i++)
        csrA_offsets_h2[i + 1] = csrA_offsets_h2[i] + 1;

    for (int i = 0; i < nrows; i++) {
        csrA_columns_h2[i] = i;
    }

    for (int i = 0; i < nnz2; i++)
        csrA_values_h2[i] = i * 4.0f;

    /* Print the initialized matrix A */
    printf("matrix A:\n");
    print_sparse_matrix(nrows, csrA_offsets_h, csrA_columns_h, csrA_values_h);

    cudssMatrix_t A = NULL;

    /* Allocate device memory for A */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csrA_offsets_d, (nrows + 1) * sizeof(int)),
                        "cudaMalloc for csrA_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csrA_columns_d, nnz * sizeof(int)),
                        "cudaMalloc for csrA_columns");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csrA_values_d, nnz * sizeof(double)),
                        "cudaMalloc for csrA_values");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csrA_offsets_d2, (nrows + 1) * sizeof(int)),
                        "cudaMalloc for csrA_offsets");
    CUDA_CALL_AND_CHECK(cudaMalloc(&csrA_columns_d2, nnz2 * sizeof(int)),
                        "cudaMalloc for csrA_columns");
    /* Note: in this example we use csrA_values_d2 to demonstrate two things:
       First, how cudssMatrixSetValues can be used to replace the values while
       keeping the same sparsity structure.
       Second, how cudssMatrixSetCsrPointers can change values and the structure.
       Thus we allocate csrA_values_d2 so that it has space max(nnz, nnz2) = nnz values */
    CUDA_CALL_AND_CHECK(cudaMalloc(&csrA_values_d2, nnz * sizeof(double)),
                        "cudaMalloc for future csrA_values");

    /* Copy host memory to device for A */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_offsets_d, csrA_offsets_h, (nrows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy H2D for csrA_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_columns_d, csrA_columns_h, nnz * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy H2D for csrA_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_values_d, csrA_values_h, nnz * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy H2D for csrA_values");

    /* (optional for this example) Creating a library handle is strictly not necessary
       for using cuDSS matrix helper APIs, but it is done here. */
    cudssHandle_t handle;
    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* Create a matrix object for the sparse matrix A. */
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SYMMETRIC;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csrA_offsets_d, NULL,
                         csrA_columns_d, csrA_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
                         base), status, "cudssMatrixCreateCsr");

    /* Note: cuDSS matrix objects are lightweight wrappers around the user data.*/
    int64_t nrows_out, ncols_out;
    int64_t nnz_out;
    void *csrA_offsets_out = NULL;
    void *csrA_columns_out = NULL;
    void *csrA_values_out = NULL;
    cudaDataType_t index_type_out, value_type_out;
    cudssMatrixType_t mtype_out;
    cudssMatrixViewType_t mview_out;
    cudssIndexBase_t index_base_out;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetCsr(A, &nrows_out, &ncols_out, &nnz_out,
                         &csrA_offsets_out, NULL, &csrA_columns_out, &csrA_values_out,
                         &index_type_out, &value_type_out, &mtype_out, &mview_out, &index_base_out),
                         status, "cudssMatrixGetCsr for A");

    if (csrA_offsets_out != csrA_offsets_d || csrA_columns_out != csrA_columns_d ||
        csrA_values_out != csrA_values_d ||
        nrows_out != nrows || ncols_out != ncols || nnz_out != nnz ||
        index_type_out != CUDA_R_32I || value_type_out != CUDA_R_64F ||
        mtype_out != mtype || mview_out != mview || index_base_out != base) {
        printf("Error: Example FAILED: wrong data returned for matrix A\n");
        CUDSS_EXAMPLE_FREE;
        return -4;
    } else {
        printf("Success: Check for matrix A from cudssMatrixGetCsr passed\n");
    }

    /* Modifying values of the matrix A in-place */
    double new_value = -1.0;
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_values_d, &new_value, 1 * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemset");
    CUDSS_CALL_AND_CHECK(cudssMatrixGetCsr(A, NULL, NULL, NULL,
                         NULL, NULL, NULL, &csrA_values_out,
                         NULL, NULL, NULL, NULL, NULL),
                         status, "cudssMatrixGetCsr for A");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_offsets_h, csrA_offsets_out, (nrows + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost), "cudaMemcpy DTH for csrA_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_columns_h, csrA_columns_out, nnz * sizeof(int),
                        cudaMemcpyDeviceToHost), "cudaMemcpy DTH for csrA_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_values_h, csrA_values_out, nnz * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy DTH for csrA_values");

    if (csrA_values_h[0] != new_value) {
        printf("Error: Example FAILED: modification of A values in-place failed\n"
               "first value of matrix A equals %3.3e (expected %3.3e)\n",
                csrA_values_h[0], new_value);
        CUDSS_EXAMPLE_FREE;
        return -5;
    } else {
        printf("matrix A (after modifying values in place):\n");
        print_sparse_matrix(nrows, csrA_offsets_h, csrA_columns_h, csrA_values_h);
    }

    /* Replacing values of matrix A with new data */
    new_value = -4.0;
    CUDA_CALL_AND_CHECK(cudaMemcpy(&csrA_values_d2[1], &new_value, 1 * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemset with new values");
    CUDSS_CALL_AND_CHECK(cudssMatrixSetValues(A, csrA_values_d2),
                         status, "cudssMatrixSetValues for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixGetCsr(A, NULL, NULL, NULL,
                         NULL, NULL, NULL, &csrA_values_out,
                         NULL, NULL, NULL, NULL, NULL),
                         status, "cudssMatrixGetCsr for A");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_values_h, csrA_values_out, nnz * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy for csrA_values");
    if (csrA_values_h[1] != new_value) {
        printf("Error: Example FAILED: modification of B values out-of-place failed\n"
               "second value of matrix A equals %3.3e (expected %3.3e)\n",
                csrA_values_h[1], new_value);
        CUDSS_EXAMPLE_FREE;
        return -6;
    } else {
        printf("Success: After modifying in-place, second value of\n"
               "matrix A equals %3.3e (expected %3.3e)\n",
                csrA_values_h[1], new_value);
    }

    /* Changing the matrix structure (here: removing the last entry) */
    /* Copy host memory for new matrix arrays to device */
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_offsets_d2, csrA_offsets_h2, (nrows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy H2D for csrA_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_columns_d2, csrA_columns_h2, nnz2 * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy H2D for csrA_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_values_d2, csrA_values_h2, nnz2 * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy H2D for csrA_values");

    /* Resetting the CSR pointers for the matrix A */
    CUDSS_CALL_AND_CHECK(cudssMatrixSetCsrPointers(A, csrA_offsets_d2, NULL,
                         csrA_columns_d2, csrA_values_d2),
                         status, "cudssMatrixSetCsrPointers for A");
    int64_t new_nnz_out;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetCsr(A, NULL, NULL, &new_nnz_out,
                         &csrA_offsets_out, NULL, &csrA_columns_out, &csrA_values_out,
                         NULL, NULL, NULL, NULL, NULL),
                         status, "cudssMatrixGetCsr for A");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_offsets_h, csrA_offsets_out, (nrows + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost), "cudaMemcpy DTH for csrA_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_columns_h, csrA_columns_out, nnz2 * sizeof(int),
                        cudaMemcpyDeviceToHost), "cudaMemcpy DTH for csrA_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csrA_values_h, csrA_values_out, nnz2 * sizeof(double),
                        cudaMemcpyDeviceToHost), "cudaMemcpy DTH for csrA_values");

    if (new_nnz_out == nnz_out) {
        printf("Note: Changing the matrix structure does not change the nnz returned by\n"
               "cudssMatrixGetCsr. The original value = %ld passed to cudssMatrixCreateCsr is\n"
               "returned\n", new_nnz_out);
        printf("matrix A (after changing the matrix structure):\n");
        print_sparse_matrix(nrows, csrA_offsets_h, csrA_columns_h, csrA_values_h);
    } else {
        printf("Error: Example FAILED: unexpected nnz = %ld is returned after changing the\n"
               " matrix structure (expected %ld)\n",
                new_nnz_out, nnz_out);
        CUDSS_EXAMPLE_FREE;
        return -7;
    }

    /* Demonstrating cudssMatrixGetFormat */
    int format;
    CUDSS_CALL_AND_CHECK(cudssMatrixGetFormat(A, &format),
                         status, "cudssMatrixGetFormat for A");
    if (format != CUDSS_MFORMAT_CSR) {
        printf("Error: Example FAILED: wrong format = %d was returned for A\n",
                format);
        CUDSS_EXAMPLE_FREE;
        return -8;
    } else
        printf("Success: Matrix A format returned as %d (CSR)\n", format);

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
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
