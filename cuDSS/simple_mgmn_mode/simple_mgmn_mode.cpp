/*
 * Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
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
#include <string.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>

#include "cudss.h"

#ifdef USE_MPI
#  include "mpi.h"
#  ifdef USE_NCCL
#    include "nccl.h"
#  endif
#  if !defined(USE_OPENMPI) && !defined(USE_NCCL)
#    error "With USE_MPI, either USE_OPENMPI or USE_NCCL must be defined"
#  endif
#  if defined(USE_OPENMPI) && defined(USE_NCCL)
#    error "With USE_MPI, exactly one of USE_OPENMPI and USE_NCCL must be defined"
#  endif
#else
#  error "This example needs to be compiled with MPI (and optionally with NCCL as well)"
#endif

/*
    This example demonstrates usage of MGMN mode in cuDSS for solving
    a system of linear algebraic equations with a sparse matrix:
                                Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
    Note: in this example A, b and x are assumed to be fully present on
    the root process, the rest of the processes are assumed to have
    correct matrix shapes only.

    Note: The MGMN mode is intended to be used for solving large systems.
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
            return -3; \
        } \
    } while(0);

#ifdef USE_MPI
#  define MPI_CALL_AND_CHECK(call, msg) \
    do { \
        mpi_error = call; \
        if (mpi_error != 0) { \
            printf("Example FAILED: MPI call returned error = %d, details: " #msg "\n", mpi_error); \
            return -4; \
        } \
    } while(0);
#  ifdef USE_NCCL
#    define NCCL_CALL_AND_CHECK(call, msg) \
    do { \
        nccl_result = call; \
        if (nccl_result != ncclSuccess) { \
            printf("Example FAILED: NCCL call returned error = %d, details: " #msg "\n", nccl_result); \
            return -5; \
        } \
    } while(0);
#  endif
#endif


#define CUDSS_CALL_AND_CHECK(call, status, msg) \
    do { \
        status = call; \
        if (status != CUDSS_STATUS_SUCCESS) { \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, details: " #msg "\n", status); \
            CUDSS_EXAMPLE_FREE; \
            return -6; \
        } \
    } while(0);


int main (int argc, char *argv[]) {
    cudaError_t   cuda_error = cudaSuccess;
    cudssStatus_t status     = CUDSS_STATUS_SUCCESS;
#ifdef USE_MPI
    int           mpi_error  = 0;
#  ifdef USE_NCCL
    ncclResult_t  nccl_result = ncclSuccess;
#  endif
#endif

    /* Initializing the communication backend
       Note: cuDSS can work with any CUDA-aware communication backend through the
       user-defined communication layers (see the documentation), but this example
       demonstrates the mode using pre-built communication layers for OpenMPI and
       NCCL.
       Therefore, as both of them rely on MPI_Init/MPI_Finalize, we call it here.
       For a different communication backend, a different initialization/cleanup
       APIs might be needed. */
#ifdef USE_MPI
    MPI_CALL_AND_CHECK(MPI_Init(&argc, &argv), "MPI_Init");
#endif

    /* Identifying the root process as the one with rank equal to 0 */
    int rank = 0, size = 1;
#ifdef USE_MPI
    MPI_CALL_AND_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    MPI_CALL_AND_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size");
#endif
    if (rank == 0) {
        printf("---------------------------------------------------------\n");
        printf("cuDSS example: this example will be run with %d processes\n", size);
        printf("Note: number of processes must not exceed the number of\n"
               "GPU devices available\n");
        printf("---------------------------------------------------------\n");
        printf("cuDSS example: solving a real linear 5x5 system\n"
               "with a symmetric positive-definite matrix using \n"
               "the distributed memory mode.\n");
        printf("---------------------------------------------------------\n");
        fflush(0);
    }

    /* Binding each process to a specific GPU device under the assumption
       that the number of processes does not exceed the number of devices */
    int device_count = 0;
    CUDA_CALL_AND_CHECK(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
        printf("Error: no GPU devices have been found\n");
        fflush(0);
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return -1;
    }
    int device_id = rank % device_count;
    CUDA_CALL_AND_CHECK(cudaSetDevice(device_id), "cudaSetDevice");

    /* Parsing the communication layer information from the input parameters */
    char comm_backend_name[1024];
    char comm_layer_lib[1024];
    if (argc > 2) {
        strcpy(comm_backend_name, argv[1]);
        if (rank == 0) printf("Communication backend name is: %s\n", comm_backend_name);
        strcpy(comm_layer_lib, argv[2]);
        printf("Communication layer library name is: %s\n", comm_layer_lib);
        fflush(0);
    } else {
        if (rank == 0) {
            printf("Error: this example requires passing:\n"
                    "a) the communication backend name (openmpi or nccl)\n"
                    "b) the communication layer library (full name with the path)\n"
                    "Note: backend should match the communication layer library\n");
            fflush(0);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return -2;
    }


    /* Creating a communicator of the type matching the communication backend */
#ifdef USE_MPI
    MPI_Comm *mpi_comm = NULL;

    #if USE_OPENMPI
    if (strcmp(comm_backend_name,"openmpi") == 0) {
        mpi_comm = (MPI_Comm*) malloc(sizeof(MPI_Comm));
        mpi_comm[0] = MPI_COMM_WORLD;
    }
    #endif
    #if USE_NCCL
    ncclComm_t *nccl_comm = NULL;
    if (strcmp(comm_backend_name,"nccl") == 0) {
        nccl_comm = (ncclComm_t*) malloc(sizeof(ncclComm_t));
        ncclUniqueId id;
        if (rank == 0) {
            NCCL_CALL_AND_CHECK(ncclGetUniqueId(&id), "ncclGetUniqueId");
        }
        MPI_CALL_AND_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD),
                           "MPI_Bcast for id");
        NCCL_CALL_AND_CHECK(ncclCommInitRank(nccl_comm, size, id, rank), "ncclCommInitRank");
    }
    #endif
#endif

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

    /* We only allocate host and device memory  for A,x and b for the root process */
    if (rank == 0) {
        /* Allocate host memory for the sparse input matrix A,
           right-hand side x and solution b*/
        csr_offsets_h = (int*)malloc((n + 1) * sizeof(int));
        csr_columns_h = (int*)malloc(nnz * sizeof(int));
        csr_values_h = (double*)malloc(nnz * sizeof(double));
        x_values_h = (double*)malloc(nrhs * n * sizeof(double));
        b_values_h = (double*)malloc(nrhs * n * sizeof(double));

        if (!csr_offsets_h || ! csr_columns_h || !csr_values_h ||
            !x_values_h || !b_values_h) {
            printf("Error: host memory allocation failed\n");fflush(0);
            return -2;
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
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, nnz * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, nnz * sizeof(double),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
        CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h, nrhs * n * sizeof(double),
                            cudaMemcpyHostToDevice), "cudaMemcpy for b_values");
    }

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* Set the full name of the cuDSS communication layer library. 
      Note: if comm_layer_lib = NULL then cudssSetCommLayer takes
      the communication layer library name from the environment variable
      "CUDSS_COMM_LIB“ */
    cudssSetCommLayer(handle, comm_layer_lib);

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices).
       Note: currently, solution and right0hand side arrays must be fully present only on
       the root process (rank = 0). */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix.
       Note: currently, matrix A must be fully present on the root process (rank = 0),
       and the rest of the processes should have correct shape of the matrix (but it is
       fine to have NULL pointers for the data arrays). */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                         csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
                         base), status, "cudssMatrixCreateCsr");

    /* Setting communicator to be used by MGMN mode of cuDSS */
#ifdef USE_MPI
    #if USE_OPENMPI
    if (strcmp(comm_backend_name,"openmpi") == 0) {
        CUDSS_CALL_AND_CHECK(cudssDataSet(handle, solverData, CUDSS_DATA_COMM,
                                          mpi_comm, sizeof(MPI_Comm*)),
                                          status, "cudssDataSet for OpenMPI comm");
    }
    #endif
    #if USE_NCCL
    if (strcmp(comm_backend_name,"nccl") == 0) {
        CUDSS_CALL_AND_CHECK(cudssDataSet(handle, solverData, CUDSS_DATA_COMM,
                             nccl_comm, sizeof(ncclComm_t*)),
                             status, "cudssDataSet for NCCL comm");
    }
    #endif
#endif

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData,
                         A, x, b), status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                         solverData, A, x, b), status, "cudssExecute for factor");

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

    /* (optional) For the root process, print the solution and compare against the exact solution */
    int passed = 1;
    if (rank == 0) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                            cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

        for (int i = 0; i < n; i++) {
            printf("x[%d] = %1.4f expected %1.4f\n", i, x_values_h[i], double(i+1));
            if (fabs(x_values_h[i] - (i + 1)) > 2.e-15)
            passed = 0;
        }
    }

    /* (optional) For the root process, print the solution and compare against the exact solution */
    if (rank == 0) {
        /* Release the data allocated on the user side */

        CUDSS_EXAMPLE_FREE;
    }

    /* Deleting the memory allocated for the communicator */
#if USE_MPI
    if (mpi_comm != NULL) free(mpi_comm);
#endif
#if USE_NCCL
    if (nccl_comm != NULL) NCCL_CALL_AND_CHECK(ncclCommDestroy(*nccl_comm),"ncclCommDestroy");
    if (nccl_comm != NULL) free(nccl_comm);
#endif

    /* Cleanup for the communication backend
       See comments about calling MPI_Init() above */
#ifdef USE_MPI
    MPI_Finalize();
#endif

    if (status == CUDSS_STATUS_SUCCESS && passed) {
        if (rank == 0)
            printf("Example PASSED\n");
        return 0;
    } else {
        if (rank == 0)
            printf("Example FAILED\n");
        return -3;
    }
}
