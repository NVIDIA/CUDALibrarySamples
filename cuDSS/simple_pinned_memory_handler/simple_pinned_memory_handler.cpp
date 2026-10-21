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


#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(USE_OPENMP)
#include <omp.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "cudss.h"
#include "laplace_generator.hxx"

/*
    This example mirrors cudss_memory_handler.cpp, but demonstrates the pinned
    (page-locked) host memory handler APIs (cudssSetPinnedMemHandler /
    cudssGetPinnedMemHandler) instead of the device ones. It solves a system of
    linear algebraic equations with a sparse matrix:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).

    This example focuses on how to define, attach (cudssSetPinnedMemHandler),
    query (cudssGetPinnedMemHandler), and exercise a pinned host memory handler.
    It generates a 3D Laplacian large enough for cuDSS hybrid mode to route
    selected internal host allocations through the handler.
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


/*
 * Illustrative C++ pinned host memory handler. Large requests use mmap()
 * plus cudaHostRegister(); when OpenMP is enabled, pages are pre-faulted in
 * parallel before registration. Non-Linux builds and small requests use
 * cudaMallocHost().
 */
class ExamplePinnedMemPool {
  public:
    explicit ExamplePinnedMemPool(size_t large_threshold) : large_threshold_(large_threshold) {}

    int alloc(void **ptr, size_t size, cudaStream_t) {
        if (size == 0) {
            *ptr = NULL;
            return 0;
        }

#ifdef __linux__
        if (use_large_path(size)) {
            void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) {
                *ptr = NULL;
                return -1;
            }
            int madv_ret = madvise(p, size, MADV_HUGEPAGE);
            bool huge_pages = (madv_ret == 0);
            int nthreads = max_threads();
            size_t stride = page_stride(huge_pages);
            if (nthreads > 1) {
                prefault_pages(p, size, stride, nthreads);
            }

            cudaError_t cuda_status = cudaHostRegister(p, size, 0);
            if (cuda_status == cudaErrorHostMemoryAlreadyRegistered) {
                cudaGetLastError();
            } else if (cuda_status != cudaSuccess) {
                munmap(p, size);
                *ptr = NULL;
                return static_cast<int>(cuda_status);
            }

            *ptr = p;
            printf("alloc() from the ExamplePinnedMemPool: registered large path "
                   "(mmap + madvise(MADV_HUGEPAGE)=%d + %s cudaHostRegister) "
                   "at %p (size %zu)\n",
                   madv_ret, nthreads > 1 ? "parallel pre-fault + " : "", p, size);
            return 0;
        }
#endif

        /* Typical path: a plain pinned host allocation. */
        int status = cudaMallocHost(ptr, size);
        printf("alloc() from the ExamplePinnedMemPool: typical path (cudaMallocHost) "
               "at %p (size %zu)\n",
               *ptr, size);
        return status;
    }

    int dealloc(void *ptr, size_t size, cudaStream_t) {
        if (ptr == NULL) {
            return 0;
        }

#ifdef __linux__
        if (use_large_path(size)) {
            printf("dealloc() from the ExamplePinnedMemPool: registered large path "
                   "(cudaHostUnregister + munmap) at %p (size %zu)\n",
                   ptr, size);
            cudaError_t cuda_status = cudaHostUnregister(ptr);
            int unmap_status = munmap(ptr, size);
            if (cuda_status != cudaSuccess) {
                return static_cast<int>(cuda_status);
            }
            return unmap_status;
        }
#endif

        printf("dealloc() from the ExamplePinnedMemPool: typical path (cudaFreeHost) "
               "at %p (size %zu)\n",
               ptr, size);
        return static_cast<int>(cudaFreeHost(ptr));
    }

  private:
    /* The size dispatch is deterministic, so pinned_free can re-derive which path a
       buffer took purely from its size (no per-pointer book-keeping needed). */
    bool use_large_path(size_t size) const {
        return size > large_threshold_;
    }

    static int max_threads() {
#if defined(USE_OPENMP)
        int nthreads = omp_get_max_threads();
        return nthreads > 0 ? nthreads : 1;
#else
        return 1;
#endif
    }

    static size_t page_stride(bool huge_pages) {
        if (huge_pages) {
            return size_t(2) * 1024 * 1024;
        }
#ifdef __linux__
        long ps = sysconf(_SC_PAGESIZE);
        return ps > 0 ? static_cast<size_t>(ps) : size_t(4096);
#else
        return size_t(4096);
#endif
    }

    static void prefault_pages(void *ptr, size_t size, size_t stride, int nthreads) {
        size_t n_pages = (size + stride - 1) / stride;
#if defined(USE_OPENMP)
#pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
        for (int64_t page = 0; page < static_cast<int64_t>(n_pages); page++) {
            volatile char *addr = static_cast<volatile char *>(ptr) + size_t(page) * stride;
            *addr = 0;
        }
    }

    size_t large_threshold_;
};

/*
 * The pinned host memory handler APIs of cuDSS are in C and these wrappers below
 * demonstrate how a C++ memory pool could be used to define an object of
 * type cudssPinnedMemHandler_t.
 */
int example_pinned_alloc(void *ctx, void **ptr, size_t size, cudaStream_t stream) {
    return reinterpret_cast<ExamplePinnedMemPool *>(ctx)->alloc(ptr, size, stream);
}

int example_pinned_free(void *ctx, void *ptr, size_t size, cudaStream_t stream) {
    return reinterpret_cast<ExamplePinnedMemPool *>(ctx)->dealloc(ptr, size, stream);
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("---------------------------------------------------------\n");
    printf("cuDSS example: pinned host memory allocator API\n"
           "with a generated 3D Laplacian in hybrid memory mode\n");
    printf("---------------------------------------------------------\n");
    cudaError_t   cuda_error = cudaSuccess;
    cudssStatus_t status     = CUDSS_STATUS_SUCCESS;

    int nx   = 32;
    int n    = nx;
    int nnz  = 0;
    int nrhs = 1;

    int    *csr_offsets_h = NULL;
    int    *csr_columns_h = NULL;
    double *csr_values_h  = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int    *csr_offsets_d = NULL;
    int    *csr_columns_d = NULL;
    double *csr_values_d  = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    /* Creating a pinned host memory pool object. The policy lives in the ctx. */
    printf("Creating an illustrative pinned host memory handler\n");
#ifdef __linux__
#if defined(USE_OPENMP)
    printf("  Large Linux requests use mmap + madvise(MADV_HUGEPAGE) + "
           "parallel pre-fault + cudaHostRegister\n");
#else
    printf("  Large Linux requests use mmap + madvise(MADV_HUGEPAGE) + "
           "cudaHostRegister\n");
#endif
#else
    printf("  Non-Linux build: using the portable cudaMallocHost/cudaFreeHost path\n");
#endif
    // Demo value for the pool's own size dispatch if cuDSS routes a request to it.
    const size_t large_threshold = 4096;
    // Keep the pool alive for at least as long as the cuDSS handle that uses it.
    ExamplePinnedMemPool pool(large_threshold);

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating a pinned host memory handler based on the memory pool defined above */
    printf("Creating a memory pool wrapper object of type cudssPinnedMemHandler_t\n");
    cudssPinnedMemHandler_t handler;
    handler.ctx          = reinterpret_cast<void *>(&pool);
    handler.pinned_alloc = example_pinned_alloc;
    handler.pinned_free  = example_pinned_free;
    strncpy(handler.name, "illustrative pinned host memory handler",
            CUDSS_ALLOCATOR_NAME_LEN);
    handler.name[CUDSS_ALLOCATOR_NAME_LEN - 1] = '\0';

    /* Generate a full 7-point 3D Laplacian. nx=32 keeps the example small while
       still large enough for cuDSS hybrid mode to call the pinned handler. */
    laplace_3d_7p<double>(&n, &nnz, &csr_offsets_h, &csr_columns_h, &csr_values_h,
                          nrhs);
    x_values_h    = static_cast<double *>(malloc(nrhs * n * sizeof(double)));
    b_values_h    = static_cast<double *>(malloc(nrhs * n * sizeof(double)));

    if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h || !b_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }
    printf("Generated 3D Laplacian: nx = %d, n = %d, nnz = %d\n", nx, n, nnz);

    /* Manufactured right-hand side: exact solution is x = 1. */
    for (int row = 0; row < n; row++) {
        double row_sum = 0.0;
        for (int p = csr_offsets_h[row]; p < csr_offsets_h[row + 1]; p++) {
            row_sum += csr_values_h[p];
        }
        b_values_h[row] = row_sum;
    }

    const size_t user_device_bytes =
        size_t(n + 1) * sizeof(int) + size_t(nnz) * sizeof(int) +
        size_t(nnz) * sizeof(double) + 2 * size_t(nrhs) * size_t(n) * sizeof(double);
    printf("User-side device allocations for A, b, and x require %zu bytes "
           "(%.2f MiB)\n",
           user_device_bytes, double(user_device_bytes) / (1024.0 * 1024.0));

    size_t free_device_bytes  = 0;
    size_t total_device_bytes = 0;
    CUDA_CALL_AND_CHECK(cudaMemGetInfo(&free_device_bytes, &total_device_bytes),
                        "cudaMemGetInfo before device allocations");
    printf("CUDA device memory before user allocations: free %zu bytes (%.2f MiB), "
           "total %zu bytes (%.2f MiB)\n",
           free_device_bytes, double(free_device_bytes) / (1024.0 * 1024.0),
           total_device_bytes, double(total_device_bytes) / (1024.0 * 1024.0));

    /* Allocate device memory for A, x and b
       Note: These device memory allocations are "external" to cuDSS and are
       unrelated to the pinned host memory handler demonstrated here, so we rely on
       regular cudaMalloc/cudaFree(). */
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

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* (optional) Setting a user-defined pinned host memory pool for the library
       handle so that selected internal host allocations inside cuDSS use the user's
       memory pool. If this is not set, cuDSS keeps using its previous plain host
       allocations. */
    printf("Setting the pinned host memory handler in the cuDSS library handle so that "
           "internal host memory allocations inside cuDSS would use the user's "
           "memory pool\n");
    CUDSS_CALL_AND_CHECK(cudssSetPinnedMemHandler(handle, &handler), status,
                         "cudssSetPinnedMemHandler");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t   solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");

    /* Enable hybrid mode, where selected cuDSS-owned host allocations are routed
       through the pinned host memory handler. */
    int hybrid_mode = 1;
    CUDSS_CALL_AND_CHECK(cudssConfigSet(solverConfig, CUDSS_CONFIG_HYBRID_MEMORY_MODE,
                                        &hybrid_mode, sizeof(hybrid_mode)),
                         status, "cudssConfigSet for hybrid memory mode");

    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices).
     */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int     ldb = nrows, ldx = ncols;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, nrows, nrhs, ldb, b_values_d,
                                             CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, ncols, nrhs, ldx, x_values_d,
                                             CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. The Laplacian generator
       stores the full matrix, so it is described as general with a full view. */
    cudssMatrix_t         A;
    cudssMatrixType_t     mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t      base  = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDA_R_32I,
                                              CUDA_R_32I, CUDA_R_64F, mtype, mview,
                                              base),
                         status, "cudssMatrixCreateCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b),
        status, "cudssExecute for analysis");

    size_t  sizeWritten = 0;
    int64_t device_memory_min = 0;
    CUDSS_CALL_AND_CHECK(
        cudssDataGet(handle, solverData, CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN,
                     &device_memory_min, sizeof(device_memory_min), &sizeWritten),
        status, "cudssDataGet for CUDSS_DATA_HYBRID_DEVICE_MEMORY_MIN");
    printf("cuDSS hybrid minimum device memory is %lld bytes (%.2f MiB)\n",
           static_cast<long long>(device_memory_min),
           double(device_memory_min) / (1024.0 * 1024.0));
    CUDSS_CALL_AND_CHECK(
        cudssConfigSet(solverConfig, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
                       &device_memory_min, sizeof(device_memory_min)),
        status, "cudssConfigSet for CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT");
    printf("Approximate GPU memory covered by user allocations plus the cuDSS "
           "hybrid limit: %.2f MiB\n",
           (double(user_device_bytes) + double(device_memory_min)) /
               (1024.0 * 1024.0));

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b),
                         status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b),
        status, "cudssExecute for solve");

    /* (optional) Getting a user-defined memory pool from the library handle */
    cudssPinnedMemHandler_t returned_handler;
    CUDSS_CALL_AND_CHECK(cudssGetPinnedMemHandler(handle, &returned_handler), status,
                         "cudssGetPinnedMemHandler");
    printf("Pinned host memory handler name: %s\n", returned_handler.name);

    int passed = 1;

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status,
                         "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Compare against the exact solution x = 1. */
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy for x_values");

    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(x_values_h[i] - 1.0);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > 1.e-8) {
            passed = 0;
        }
    }
    printf("max |x[i] - 1| = %1.3e\n", max_diff);

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
