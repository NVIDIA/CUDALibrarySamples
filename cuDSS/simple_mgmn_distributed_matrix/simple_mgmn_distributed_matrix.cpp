/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
        A is the sparse input matrix distributed among 2 mpi ranks,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
    Note: in this example A, b and x are distributed between 2 ranks,
    the rest of the processes have empty input. However all ranks are assumed
    to have correct global matrix shapes (n, nnz, nrhs).

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
            if (rank == 0 || rank == 1) { \
                CUDSS_EXAMPLE_FREE; \
            } \
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
               "with a distributed symmetric positive-definite matrix using \n"
               "the MGMN mode.\n");
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
    char comm_layer_libname[1024];
    if (argc > 2) {
        strncpy(comm_backend_name, argv[1], sizeof(comm_backend_name));
        if (rank == 0) printf("Communication backend name is: %s\n", comm_backend_name);
        strncpy(comm_layer_libname, argv[2], sizeof(comm_layer_libname));
        printf("Communication layer library name is: %s\n", comm_layer_libname);
        fflush(0);
    } else if (argc == 2) {
        strncpy(comm_backend_name, argv[1], sizeof(comm_backend_name));
        if (rank == 0) printf("Communication backend name is: %s\n", comm_backend_name);
        if (rank == 0)
            printf("Since communication layer library name was not provided as input \n"
                    "argument for the executable, the layer library name will be \n"
                    "taken from CUDSS_COMM_LIB environment variable\n");
        fflush(0);
    } else {
        if (rank == 0) {
            printf("Error: this example requires passing at least one argument:\n"
                    "the communication backend name (openmpi or nccl)\n"
                    "and, optionally, the second argument for the communication \n"
                    "layer library (full name with the path)\n"
                    "If the second argument is not present, the layer library name will be \n"
                    "taken from CUDSS_COMM_LIB environment variable\n"
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

    int matrix_row_start = 0;
    int matrix_row_end   = 0;
    int b_row_start = 0;
    int b_row_end   = 0;

    //Note that distribution of the solution can be different from the matrix
    //    and right-hand side
    //In this example we set full overlapping for the solution vector that means
    //    all ranks will have full results
    int x_row_start = 0;
    int x_row_end   = n - 1;
    x_values_h = (double*)malloc(nrhs * n * sizeof(double));
    if (!x_values_h) {
        printf("Error: host memory allocation failed\n");fflush(0);
        return -2;
    }
    /* Allocate device memory for x and b */
    CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, nrhs * n * sizeof(double)),
        "cudaMalloc for x_values");

    /* We only allocate host and device memory A for rank 0 and 1*/
    if (rank == 0) {
        //Matrix row start and end should be always in 0-based
        matrix_row_start = 0;
        matrix_row_end   = size == 1 ? n - 1 : 3;

        //Note that distribution of right-hand can be different from the matrix.
        b_row_start = 0;
        b_row_end   = size == 1 ? n - 1 : 2;
        int local_b_n = b_row_end - b_row_start + 1;

        int local_n = matrix_row_end - matrix_row_start + 1;
        int local_nnz = size == 1 ? nnz : 5;

        /* Allocate host memory for the sparse input matrix A and right-hand b*/
        csr_offsets_h = (int*)malloc((local_n + 1) * sizeof(int));
        csr_columns_h = (int*)malloc(local_nnz * sizeof(int));
        csr_values_h = (double*)malloc(local_nnz * sizeof(double));

        if (!csr_offsets_h || ! csr_columns_h || !csr_values_h) {
            printf("Error: host memory allocation failed\n");fflush(0);
            return -2;
        }

        /* Allocate host memory for the right-hand side b*/
        b_values_h = (double*)malloc(nrhs * local_b_n * sizeof(double));
        if (!b_values_h) {
            printf("Error: host memory allocation failed\n");fflush(0);
            return -2;
        }

        /* Initialize host memory for A and b */
        if (size == 1) {
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
             * to the exact solution vector {1, 2, 3, 4, 5} */
            i = 0;
            b_values_h[i++] = 7.0;
            b_values_h[i++] = 12.0;
            b_values_h[i++] = 25.0;
            b_values_h[i++] = 4.0;
            b_values_h[i++] = 13.0;
        } else {
            int i = 0;
            csr_offsets_h[i++] = 0;
            csr_offsets_h[i++] = 2;
            csr_offsets_h[i++] = 3;
            csr_offsets_h[i++] = 4;
            csr_offsets_h[i++] = 5;

            i = 0;
            csr_columns_h[i++] = 0; csr_columns_h[i++] = 2;
            csr_columns_h[i++] = 2;
            csr_columns_h[i++] = 2;
            csr_columns_h[i++] = 3;

            i = 0;
            csr_values_h[i++] = 4.0; csr_values_h[i++] = 1.0;
            csr_values_h[i++] = 2.0;
            csr_values_h[i++] = 5.0;
            csr_values_h[i++] = 0.5;

            i = 0;
            b_values_h[i++] = 7.0;
            b_values_h[i++] = 12.0;
            b_values_h[i++] = 12.5;
        }

        /* Allocate device memory for A */
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (local_n + 1) * sizeof(int)),
                            "cudaMalloc for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, local_nnz * sizeof(int)),
                            "cudaMalloc for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, local_nnz * sizeof(double)),
                            "cudaMalloc for csr_values");
        /* Allocate device memory for b */
        CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * local_b_n * sizeof(double)),
                            "cudaMalloc for b_values");

        /* Copy host memory to device for A*/
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (local_n + 1) * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, local_nnz * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, local_nnz * sizeof(double),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
        /* Copy host memory to device for b */
        CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h,
            nrhs * local_b_n * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy for b_values");
    } else if (rank == 1) {
        //Matrix row start and end should always be 0-based
        matrix_row_start = 1;
        matrix_row_end   = 4;
        //Note that distribution of right-hand can be different from the matrix.
        b_row_start = 2;
        b_row_end   = 4;
        int local_b_n = b_row_end - b_row_start + 1;

        int local_n = matrix_row_end - matrix_row_start + 1;
        int local_nnz = 4;

        /* Allocate host memory for the sparse input matrix A and right-hand side b*/
        csr_offsets_h = (int*)malloc((local_n + 1) * sizeof(int));
        csr_columns_h = (int*)malloc(local_nnz * sizeof(int));
        csr_values_h = (double*)malloc(local_nnz * sizeof(double));

        if (!csr_offsets_h || ! csr_columns_h || !csr_values_h) {
            printf("Error: host memory allocation failed\n");fflush(0);
            return -2;
        }

        /* Allocate host memory for the right-hand side b*/
        b_values_h = (double*)malloc(nrhs * local_b_n * sizeof(double));
        if (!b_values_h) {
            printf("Error: host memory allocation failed\n");fflush(0);
            return -2;
        }

        /* Initialize host memory for A and b */
        int i = 0;
        csr_offsets_h[i++] = 0;
        csr_offsets_h[i++] = 1;
        csr_offsets_h[i++] = 2;
        csr_offsets_h[i++] = 3;
        csr_offsets_h[i++] = 4;

        i = 0;
        csr_columns_h[i++] = 1;
        csr_columns_h[i++] = 4;
        csr_columns_h[i++] = 3;
        csr_columns_h[i++] = 4;

        i = 0;
        csr_values_h[i++] = 3.0;
        csr_values_h[i++] = 1.0;
        csr_values_h[i++] = 0.5;
        csr_values_h[i++] = 2.0;

        /* Note: Right-hand side b is initialized with values which correspond
         * to the exact solution vector {1, 2, 3, 4, 5} */
        i = 0;
        b_values_h[i++] = 12.5;
        b_values_h[i++] = 4.0;
        b_values_h[i++] = 13.0;

        /* Allocate device memory for A */
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (local_n + 1) * sizeof(int)),
                            "cudaMalloc for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, local_nnz * sizeof(int)),
                            "cudaMalloc for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, local_nnz * sizeof(double)),
                            "cudaMalloc for csr_values");
        /* Allocate device memory for b */
        CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d, nrhs * local_b_n * sizeof(double)),
                            "cudaMalloc for b_values");

        /* Copy host memory to device for A*/
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, csr_offsets_h, (local_n + 1) * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_offsets");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, csr_columns_h, local_nnz * sizeof(int),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_columns");
        CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, csr_values_h, local_nnz * sizeof(double),
                            cudaMemcpyHostToDevice), "cudaMemcpy for csr_values");
        /* Copy host memory to device for b */
        CUDA_CALL_AND_CHECK(cudaMemcpy(b_values_d, b_values_h,
            nrhs * local_b_n * sizeof(double), cudaMemcpyHostToDevice),
            "cudaMemcpy for b_values");
    } else {
        //Ranks > 1 in this example do not have a piece of the input matrix and.
        //    right-hand side. Thus setting any pair of numbers such that
        //    matrix_row_start > matrix_row_end will do.
        matrix_row_start = 1;
        matrix_row_end   = 0;
        b_row_start = 1;
        b_row_end   = 0;
    }

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* Set the full name of the cuDSS communication layer library.
      Note: if comm_layer_libname = NULL then cudssSetCommLayer takes
      the communication layer library name from the environment variable
      "CUDSS_COMM_LIB" */
    cudssSetCommLayer(handle, argc > 2 ? comm_layer_libname : NULL);

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices).
       Note: currently, solution and right-hand side arrays must be fully present only on
       the root process (rank = 0). */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    int ldb = ncols, ldx = nrows;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for b");
    /* Set row-wise 1D distribution for  X */
    CUDSS_CALL_AND_CHECK(cudssMatrixSetDistributionRow1d(b,
        b_row_start, b_row_end), status, "cudssMatrixSetDistributionRow1d");

    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                         CUDSS_LAYOUT_COL_MAJOR), status, "cudssMatrixCreateDn for x");
    /* Set row-wise 1D distribution for  X */
    CUDSS_CALL_AND_CHECK(cudssMatrixSetDistributionRow1d(x,
        x_row_start, x_row_end), status, "cudssMatrixSetDistributionRow1d");

    /* Create a matrix object for the sparse input matrix.
       Note: matrix A is distributed between rank = 0 and rank = 1,
       and the rest of the processes should have correct shape of the matrix
       and matrix_row_start > matrix_row_end (and it is
       fine to have NULL pointers for the data arrays). */
    cudssMatrix_t A;
    cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssIndexBase_t base       = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d,
        NULL, csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mview,
        base), status, "cudssMatrixCreateCsr");
    /* Set row-wise 1D distribution for matrix A */
    CUDSS_CALL_AND_CHECK(cudssMatrixSetDistributionRow1d(A,
        matrix_row_start, matrix_row_end), status, "cudssMatrixSetDistributionRow1d");

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

    /* (optional) Print the solution and compare against the exact solution */
    int passed = 1;
    CUDA_CALL_AND_CHECK(cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
        cudaMemcpyDeviceToHost), "cudaMemcpy for x_values");

    for (int i = 0; i < n; i++) {
        printf("RANK = %d x[%d] = %1.4f expected %1.4f\n",
            rank, i, x_values_h[i], double(i+1));
        if (fabs(x_values_h[i] - (i + 1)) > 2.e-15) passed = 0;
    }

    /* (optional) For the root process, print the solution and compare against the exact solution */
    if (rank == 0 || rank == 1) {
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
