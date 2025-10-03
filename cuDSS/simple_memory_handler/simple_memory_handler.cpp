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
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"

/*
    This example demonstrates usage of cuDSS APIs for user-defined memory allocators
    The APIs are showed for an example of solving a system of linear algebraic
    equations with a sparse matrix:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
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


/*
 * This is just a working example of a simple C++ memory pool solely for illustration purposes only.
 */
class ExampleDeviceMemPool {
  public:
    int alloc(void** ptr, size_t size, cudaStream_t stream) {
      int status = cudaMalloc(ptr, size);
      printf("alloc() from the ExampleDeviceMemPool allocates memory at %p (allocation size %zu)\n", (void*)*ptr, size);
      return status;
    }

    int dealloc(void* ptr, size_t size, cudaStream_t stream) {
      printf("dealloc() from the ExampleDeviceMemPool is called with ptr %p\n", (void*)ptr);
      int status = cudaFree(ptr);
      return status;
    }
};

/*
 * The device memory handler APIs of cuDSS are in C and these wrappers below
 * demonstrate how a C++ memory pool could be used to define an object of
 * type cudssDeviceMemHandler_t
 */
int example_device_alloc(void* ctx, void** ptr, size_t size, cudaStream_t stream) {
  return reinterpret_cast<ExampleDeviceMemPool*>(ctx)->alloc(ptr, size, stream);
}

int example_device_dealloc(void* ctx, void* ptr, size_t size, cudaStream_t stream) {
  return reinterpret_cast<ExampleDeviceMemPool*>(ctx)->dealloc(ptr, size, stream);
}

int main (int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear 5x5 system\n"
           "with a symmetric positive-definite matrix \n");
    printf("---------------------------------------------------------\n");
    cudaError_t cuda_error = cudaSuccess;
    cudssStatus_t status = CUDSS_STATUS_SUCCESS;

    int n = 5;
    int nnz = 8;
    int nrhs = 1;

    int *csr_offsets_h = NULL;
    int *csr_columns_h = NULL;
    double *csr_values_h = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int *csr_offsets_d = NULL;
    int *csr_columns_d = NULL;
    double *csr_values_d = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    /* Creating a memory pool object for device memory */
    printf("Creating a simple device memory pool\n");
    ExampleDeviceMemPool pool = ExampleDeviceMemPool(); // kept alive for the entire process in real apps

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating a device memory handler based on the memory pool defined above */
    printf("Creating a memory pool wrapper object of type cudssDeviceMemHandler_t\n");
    cudssDeviceMemHandler_t handler;
    handler.ctx = reinterpret_cast<void*>(&pool);
    handler.device_alloc = example_device_alloc;
    handler.device_free = example_device_dealloc;
    strncpy(handler.name, "simple verbose device memory pool", CUDSS_ALLOCATOR_NAME_LEN);

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
    csr_columns_h = (int*)malloc(nnz * sizeof(int));
    csr_values_h = (double*)malloc(nnz * sizeof(double));
    x_values_h = (double*)malloc(nrhs * n * sizeof(double));
    b_values_h = (double*)malloc(nrhs * n * sizeof(double));

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

    i = 0;
    csr_values_h[i++] = 4.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 3.0; csr_values_h[i++] = 2.0;
    csr_values_h[i++] = 5.0; csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 1.0;
    csr_values_h[i++] = 2.0;

    /* Note: Right-hand side b is initialized with values which correspond
       to the exact solution vector {1, 2, 3, 4, 5} */
    i = 0;
    b_values_h[i++] = 7.0;
    b_values_h[i++] = 12.0;
    b_values_h[i++] = 25.0;
    b_values_h[i++] = 4.0;
    b_values_h[i++] = 13.0;

    /* Allocate device memory for A, x and b
       Note: These device memory allocations are "external" to cuDSS and hence can be regular cudaMalloc()
       calls, or use the memory pool defined above, it does not matter.
       In real applications one should use the memory pool alloc/dealloc routines but here,
       to keep example code short, we rely on regular cudaMalloc/cudaFree(). */
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
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrhs * n * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy for b_values");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* (optional) Setting a user-defined device memory pool for the library handle */
    printf("Setting the device memory handler in the cudSS library handle so that now "
           "device memory allocations inside cuDSS will use the user's memory pool \n");
    CUDSS_CALL_AND_CHECK(cudssSetDeviceMemHandler(handle, &handler), status,
                         "cudssSetDeviceMemHandler");

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

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for solve");

    /* (optional) Getting a user-defined memory pool from the library handle */
    cudssDeviceMemHandler_t returned_handler;
    CUDSS_CALL_AND_CHECK(cudssGetDeviceMemHandler(handle, &returned_handler), status,
                         "cudssGetDeviceMemHandler");
    printf("Device memory handler name: %s\n", returned_handler.name);

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

    /* Release host and device data allocated on the user side */

    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS && cuda_error == cudaSuccess && passed) {
        printf("Example PASSED\n");
        return 0;
    } else {
        printf("Example FAILED\n");
        return -1;
    }
}