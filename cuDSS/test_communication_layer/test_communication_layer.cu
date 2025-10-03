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

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#error                                                                                   \
    "This sample does not support Windows right now (replace dlopen()/dlsym() with equivalent Windows API functions)"
#endif

#include <dlfcn.h> // required for dlopen and dlsym; only works on Linux

#include "cudss_distributed_interface.h"

#ifdef USE_MPI
#include "mpi.h"
#ifdef USE_NCCL
#include "nccl.h"
#endif
#if !defined(USE_OPENMPI) && !defined(USE_NCCL)
#error "With USE_MPI, either USE_OPENMPI or USE_NCCL must be defined"
#endif
#if defined(USE_OPENMPI) && defined(USE_NCCL)
#error "With USE_MPI, exactly one of USE_OPENMPI and USE_NCCL must be defined"
#endif
#else
#error "This example needs to be compiled with MPI (and optionally with NCCL as well)"
#endif

#ifdef USE_MPI
#define MPI_CALL_AND_CHECK(call, msg)                                                    \
    do {                                                                                 \
        mpi_error = call;                                                                \
        if (mpi_error != 0) {                                                            \
            printf("Example FAILED: MPI call returned error = %d, details: " #msg "\n",  \
                   mpi_error);                                                           \
            return -4;                                                                   \
        }                                                                                \
    } while (0);
#ifdef USE_NCCL
#define NCCL_CALL_AND_CHECK(call, msg)                                                   \
    do {                                                                                 \
        nccl_result = call;                                                              \
        if (nccl_result != ncclSuccess) {                                                \
            printf("Example FAILED: NCCL call returned error = %d, details: " #msg "\n", \
                   nccl_result);                                                         \
            return -5;                                                                   \
        }                                                                                \
    } while (0);
#endif
#endif

#define CUDA_CALL_AND_CHECK(call, msg)                                                   \
    do {                                                                                 \
        cuda_error = call;                                                               \
        if (cuda_error != cudaSuccess) {                                                 \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n",  \
                   cuda_error);                                                          \
            return -3;                                                                   \
        }                                                                                \
    } while (0);


/* Fills the host array data_h with number_elements values, which depend on the seed */
void fill_host_data(int *data_h, int number_elements, int seed) {
    for (int i = 0; i < number_elements; ++i) {
        data_h[i] = i + seed;
    }
}

/* Checks the values of the host array data_h against the expected values that were
 * inserted with fill_host_data. If it matches, returns 1, if there is at least one
 * mismatch, returns 0.
 */
int validate_host_data(const int *data_h, int number_elements, int seed) {
    for (int i = 0; i < number_elements; ++i) {
        if (data_h[i] != i + seed) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int passed = 1;

    cudaError_t cuda_error         = cudaSuccess;
    cudaError_t cuda_error_reduced = cudaSuccess;

#ifdef USE_MPI
    int mpi_error         = 0;
    int mpi_error_reduced = 0;
#ifdef USE_NCCL
    ncclResult_t nccl_result         = ncclSuccess;
    ncclResult_t nccl_result_reduced = ncclSuccess;
#endif
#endif

    /*
     * TEST No.1: initializing the distributed communication environment
     */

    /* Initializing the communication backend
       Note: cuDSS can work with any CUDA-aware communication backend through the
       user-defined communication layers (see the documentation), but this example
       demonstrates the mode using pre-built communication layers for OpenMPI and
       NCCL.
       Therefore, as both of them rely on MPI_Init/MPI_Finalize, we call it here.
       For a different communication backend, a different initialization/cleanup
       APIs might be needed.
     */

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
        printf("This sample will be run with %d processes\n", size);
        printf("Note: number of processes must not exceed the number of\n"
               "GPU devices available for NCCL backend\n");
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
        strcpy(comm_backend_name, argv[1]);
        if (rank == 0)
            printf("Communication backend name is: %s\n", comm_backend_name);
        strcpy(comm_layer_libname, argv[2]);
        printf("Communication layer library name is: %s\n", comm_layer_libname);
        fflush(0);
    } else {
        if (rank == 0) {
            printf("Error: this example requires passing at least two arguments:\n"
                   "the communication backend name (openmpi or nccl)\n"
                   "and the communication layer library (full name with the path)\n");
            fflush(0);
        }
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return -2;
    }

    /* Allocate host memory and set data sizes */
    const int number_elements = 256 * 1024 * 1024;
    int      *data_h          = NULL;
    data_h                    = (int *)malloc(number_elements * sizeof(int));
    if (data_h == NULL) {
        printf("Failed to allocate host memory. exiting...\nTest FAILED\n");
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return -3;
    }
    /* Should change between tests in order to use different data patterns. */
    int seed = 1;

    /* Device memory */
    int         *data_d = NULL;
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating a communicator of the type matching the communication backend */
#ifdef USE_MPI
    MPI_Comm *mpi_comm = NULL;
    mpi_comm           = (MPI_Comm *)malloc(sizeof(MPI_Comm));
    mpi_comm[0]        = MPI_COMM_WORLD;

#if USE_NCCL
    ncclComm_t *nccl_comm = NULL;
    if (strcmp(comm_backend_name, "nccl") == 0) {
        nccl_comm = (ncclComm_t *)malloc(sizeof(ncclComm_t));
        ncclUniqueId id;
        if (rank == 0) {
            NCCL_CALL_AND_CHECK(ncclGetUniqueId(&id), "ncclGetUniqueId");
        }
        MPI_CALL_AND_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD),
                           "MPI_Bcast for id");
        NCCL_CALL_AND_CHECK(ncclCommInitRank(nccl_comm, size, id, rank),
                            "ncclCommInitRank");
    }
#endif
#endif

    // If we are here and there are no errors, test No. 1 has passed
    if (cuda_error == cudaSuccess) {
#ifdef USE_MPI
        if (mpi_error == 0)
#endif
        {
#ifdef USE_NCCL
            if (nccl_result == ncclSuccess)
#endif
            {
                if (rank == 0)
                    printf("Test No. 1 [initialize environment and communicator] PASSED "
                           "on the root process\n");
                fflush(0);
            }
        }
    } else {
        printf("rank %d: Test No. 1 [initialize environment and communicator] FAILED\n",
               rank);
        fflush(0);
#ifdef USE_MPI
        printf("rank %d: Failure details: mpi_error: %d\n", rank, mpi_error);
        fflush(0);
#endif
#ifdef USE_NCCL
        printf("rank %d: Failure details: nccl_result: %d\n", rank, nccl_result);
        fflush(0);
#endif
        printf("rank %d: Failure details: cuda_error: %d\n", rank, cuda_error);
        fflush(0);
        passed = -11;
    }

    /*
     * TEST No.2: calling a collective API (without using the cuDSS communication layer)
     *            with a cudaMalloc'ed buffer (to check that the communication backend is
     * GPU-enabled)
     */

    CUDA_CALL_AND_CHECK(cudaMalloc(&data_d, number_elements * sizeof(int)), "cudaMalloc");
    if (rank == 0) {
        fill_host_data(data_h, number_elements, seed);
        CUDA_CALL_AND_CHECK(cudaMemcpy(data_d, data_h, number_elements * sizeof(int),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy");
    }
    /* Resets data_h, so the test would fail if it didn't copy back the GPU memory */
    memset(data_h, 0, number_elements * sizeof(int));

#if USE_OPENMPI
    if (strcmp(comm_backend_name, "openmpi") == 0) {
        MPI_CALL_AND_CHECK(
            MPI_Bcast(data_d, number_elements * sizeof(int), MPI_BYTE, 0, mpi_comm[0]),
            "MPI_Bcast");
        CUDA_CALL_AND_CHECK(cudaMemcpy(data_h, data_d, number_elements * sizeof(int),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy");
        if (!validate_host_data(data_h, number_elements, seed)) {
            mpi_error = 1;
            printf("rank %d: Failure details: mpi_error: %d\n", rank, mpi_error);
        }
        memset(data_h, 0, number_elements * sizeof(int));
    }
#endif
#if USE_NCCL
    if (strcmp(comm_backend_name, "nccl") == 0) {
        NCCL_CALL_AND_CHECK(ncclBcast(data_d, number_elements * sizeof(int), ncclChar, 0,
                                      nccl_comm[0], stream),
                            "ncclBcast");
        CUDA_CALL_AND_CHECK(cudaMemcpy(data_h, data_d, number_elements * sizeof(int),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy");
        if (!validate_host_data(data_h, number_elements, seed)) {
            mpi_error = 1;
            printf("rank %d: Failure details: mpi_error: %d\n", rank, mpi_error);
        }
        memset(data_h, 0, number_elements * sizeof(int));
    }
#endif

    CUDA_CALL_AND_CHECK(cudaFree(data_d), "cudaFree");
    data_d = NULL;

#ifdef USE_MPI
    MPI_CALL_AND_CHECK(
        MPI_Reduce(&mpi_error, &mpi_error_reduced, 1, MPI_INT, MPI_SUM, 0, mpi_comm[0]),
        "MPI_Reduce for mpi_error");
#ifdef USE_NCCL
    MPI_CALL_AND_CHECK(MPI_Reduce(&nccl_result, &nccl_result_reduced, 1, MPI_INT, MPI_SUM,
                                  0, mpi_comm[0]),
                       "MPI_Reduce for nccl_result");
#endif
    MPI_CALL_AND_CHECK(
        MPI_Reduce(&cuda_error, &cuda_error_reduced, 1, MPI_INT, MPI_SUM, 0, mpi_comm[0]),
        "MPI_Reduce for cuda_error");
#endif

    // If we are here and there are no errors, test No. 2 has passed
    if (cuda_error_reduced == cudaSuccess
#ifdef USE_MPI
        && mpi_error_reduced == 0
#endif
    ) {
#ifdef USE_NCCL
        if (nccl_result_reduced == ncclSuccess)
#endif
        {
            if (rank == 0)
                printf("Test No. 2 [calling a collective API with a cudaMalloc'ed "
                       "buffer] PASSED\n");
            fflush(0);
        }
    } else {
        passed = -22;
        if (rank == 0) {
            printf("Test No. 2 [calling a collective API with a cudaMalloc'ed buffer] "
                   "FAILED\n");
            fflush(0);
#ifdef USE_MPI
            printf("rank %d: Failure details: mpi_error_reduced: %d\n", rank,
                   mpi_error_reduced);
            fflush(0);
#endif
#ifdef USE_NCCL
            printf("rank %d: Failure details: nccl_result: %d\n", rank, nccl_result);
            fflush(0);
#endif
            printf("rank %d: Failure details: cuda_error: %d\n", rank, cuda_error);
            fflush(0);
        }
    }

    /*
     * TEST No.3: calling a collective API using the cuDSS communication layer library
     */

    seed++;
    mpi_error = 0;
    CUDA_CALL_AND_CHECK(cudaMalloc(&data_d, number_elements * sizeof(int)), "cudaMalloc");
    if (rank == 0) {
        fill_host_data(data_h, number_elements, seed);
        CUDA_CALL_AND_CHECK(cudaMemcpy(data_d, data_h, number_elements * sizeof(int),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy");
    }
    /* Resets data_h, so the test would fail if it didn't copy back the GPU memory */
    memset(data_h, 0, number_elements * sizeof(int));

    cudssDistributedInterface_t *commIface    = NULL;
    void                        *commIfaceLib = NULL;

    commIfaceLib = static_cast<void *>(dlopen(comm_layer_libname, RTLD_NOW));
    if (commIfaceLib == NULL) {
        printf("rank %d: Error: failed to open the communication layer library %s\n",
               rank, comm_layer_libname);
        fflush(0);
        return -33;
    }

    commIface =
        (cudssDistributedInterface_t *)dlsym(commIfaceLib, "cudssDistributedInterface");

    if (commIface == NULL) {
        printf("rank %d: Error: failed to find the symbol cudssDistributedInterface_t in "
               "the communication layer library %s\n",
               rank, comm_layer_libname);
        fflush(0);
        return -34;
    }

#if USE_OPENMPI
    if (strcmp(comm_backend_name, "openmpi") == 0) {
        commIface->cudssBcast(data_d, number_elements, CUDA_R_32I, 0, mpi_comm, stream);
        CUDA_CALL_AND_CHECK(cudaMemcpy(data_h, data_d, number_elements * sizeof(int),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy");
        if (!validate_host_data(data_h, number_elements, seed)) {
            mpi_error = 1;
        }
        memset(data_h, 0, number_elements * sizeof(int));
    }
#endif
#if USE_NCCL
    if (strcmp(comm_backend_name, "nccl") == 0) {
        commIface->cudssBcast(data_d, number_elements, CUDA_R_32I, 0, nccl_comm, stream);
        CUDA_CALL_AND_CHECK(cudaMemcpy(data_h, data_d, number_elements * sizeof(int),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy");
        if (!validate_host_data(data_h, number_elements, seed)) {
            mpi_error = 1;
        }
        memset(data_h, 0, number_elements * sizeof(int));
    }
#endif

    dlclose(commIfaceLib);

    CUDA_CALL_AND_CHECK(cudaFree(data_d), "cudaFree");
    data_d = NULL;
    CUDA_CALL_AND_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");
    stream = NULL;

#ifdef USE_MPI
    MPI_CALL_AND_CHECK(
        MPI_Reduce(&mpi_error, &mpi_error_reduced, 1, MPI_INT, MPI_SUM, 0, mpi_comm[0]),
        "MPI_Reduce for mpi_error");
#ifdef USE_NCCL
    MPI_CALL_AND_CHECK(MPI_Reduce(&nccl_result, &nccl_result_reduced, 1, MPI_INT, MPI_SUM,
                                  0, mpi_comm[0]),
                       "MPI_Reduce for nccl_result");
#endif
    MPI_CALL_AND_CHECK(
        MPI_Reduce(&cuda_error, &cuda_error_reduced, 1, MPI_INT, MPI_SUM, 0, mpi_comm[0]),
        "MPI_Reduce for cuda_error");
#endif

    // If we are here and there are no errors, test No. 3 has passed
    if (cuda_error_reduced == cudaSuccess) {
#ifdef USE_MPI
        if (mpi_error_reduced == 0)
#endif
        {
#ifdef USE_NCCL
            if (nccl_result_reduced == ncclSuccess)
#endif
            {
                if (rank == 0)
                    printf("Test No. 3 [calling a collective API using the cuDSS "
                           "communication layer library] PASSED\n");
                fflush(0);
            }
        }
    } else {
        if (rank == 0) {
            printf("Test No. 3 [calling a collective API using the cuDSS communication "
                   "layer library] FAILED\n");
            fflush(0);
#ifdef USE_MPI
            printf("rank %d: Failure details: mpi_error: %d\n", rank, mpi_error);
            fflush(0);
#endif
#ifdef USE_NCCL
            printf("rank %d: Failure details: nccl_result: %d\n", rank, nccl_result);
            fflush(0);
#endif
            printf("rank %d: Failure details: cuda_error: %d\n", rank, cuda_error);
            fflush(0);
        }
        passed = -35;
    }

    /* Deleting the memory allocated for the communicator */
#if USE_MPI
    if (mpi_comm != NULL)
        free(mpi_comm);
#endif
#if USE_NCCL
    if (nccl_comm != NULL) {
        NCCL_CALL_AND_CHECK(ncclCommDestroy(*nccl_comm), "ncclCommDestroy");
        free(nccl_comm);
    }
#endif

    // Cleanup for the communication backend
    // See comments about calling MPI_Init() above
#ifdef USE_MPI
    MPI_Finalize();
#endif
    free(data_h);

    if (passed == 1) {
        if (rank == 0)
            printf("Example PASSED\n");
        return 0;
    } else {
        if (rank == 0)
            printf("Example FAILED\n");
        return passed;
    }
}
