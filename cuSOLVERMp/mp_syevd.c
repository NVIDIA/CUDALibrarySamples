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



#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

static void generate_diagonal_dominant_symmetric_matrix(int64_t n, double* A, int64_t lda)
{
    /* set A[0:n, 0:n] = 0 */
    for (int64_t j = 0; j < n; j++)
    {
        double sum = 0;
        for (int64_t i = 0; i < n; i++)
        {
            if (i < j)
            {
                A[i + j * lda] = A[j + i * lda];
            }
            else
            {
                A[i + j * lda] = (double)(rand()) / RAND_MAX;
            }
            sum += A[i + j * lda];
        }

        A[j + j * lda] = 2 * sum;
    }
}

/* Print matrix */
static void print_host_matrix(int64_t m, int64_t n, double* A, int64_t lda, const char* msg)
{
    printf("print_host_matrix : %s\n", msg);

    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
        {
            printf("%.2lf  ", A[i + j * lda]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    Options opts = { .m           = 1,
                     .n           = 128,
                     .nrhs        = 1,
                     .mbA         = 32,
                     .nbA         = 32,
                     .mbB         = 32,
                     .nbB         = 32,
                     .mbQ         = 32,
                     .nbQ         = 32,
                     .ia          = 1,
                     .ja          = 1,
                     .ib          = 1,
                     .jb          = 1,
                     .iq          = 1,
                     .jq          = 1,
                     .p           = 2,
                     .q           = 1,
                     .grid_layout = 'C',
                     .verbose     = false };

    parse(&opts, argc, argv);
    validate(&opts);
    print(&opts);

    /* Initialize MPI library */
    MPI_Init(NULL, NULL);

    char compz = 'Z';

    /* Define dimensions, block sizes and offsets of A and B matrices */
    const int64_t n = opts.n;

    /* Offsets of A and B matrices (base-1) */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t iq = opts.iq;
    const int64_t jq = opts.jq;

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbQ = opts.mbQ;
    const int64_t nbQ = opts.nbQ;

    /* Define grid of processors */
    const int numRowDevices = opts.p;
    const int numColDevices = opts.q;

    /* Convert grid layout to cusolverMp grid mapping */
    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t rsrca = 0;
    const uint32_t csrca = 0;
    const uint32_t rsrcq = 0;
    const uint32_t csrcq = 0;

    /* Get rank id and rank size of the comm. */
    int mpiCommSize, mpiRank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiCommSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

    /*
     * Initialize device context for this process
     */
    int         localRank = getLocalRank();
    cudaError_t cudaStat  = cudaSetDevice(localRank);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    assert(cudaStat == cudaSuccess);

    {
        const int rank     = mpiRank;
        const int commSize = mpiCommSize;

        /* Error codes */
        cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
        ncclResult_t     ncclStat     = ncclSuccess;
        cudaError_t      cudaStat     = cudaSuccess;

        cudaStat = cudaSetDevice(localRank);
        assert(cudaStat == cudaSuccess);

        /* Create communicator */
        ncclUniqueId id;

        if (rank == 0)
        {
            ncclGetUniqueId(&id);
        }

        MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

        ncclComm_t comm;
        ncclStat = ncclCommInitRank(&comm, commSize, id, rank);
        assert(ncclStat == ncclSuccess);

        /* Create local stream */
        cudaStream_t localStream = NULL;
        cudaStat                 = cudaStreamCreate(&localStream);
        assert(cudaStat == cudaSuccess);

        /* Initialize cusolverMp library handle */
        cusolverMpHandle_t cusolverMpHandle = NULL;
        cusolverStat                        = cusolverMpCreate(&cusolverMpHandle, localRank, localStream);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* cusolverMp grids */
        cusolverMpGrid_t gridA = NULL;
        cusolverMpGrid_t gridQ = NULL;

        /* cusolverMp matrix descriptors */
        cusolverMpMatrixDescriptor_t descA = NULL;
        cusolverMpMatrixDescriptor_t descQ = NULL;

        /* Distributed matrices */
        void* d_A = NULL;
        void* d_D = NULL;
        void* d_Q = NULL;

        /* Distributed device workspace */
        void* d_syevdWork = NULL;

        /* Distributed host workspace */
        void* h_syevdWork = NULL;

        /* size of workspace on device */
        size_t syevdWorkspaceInBytesOnDevice = 0;

        /* size of workspace on host */
        size_t syevdWorkspaceInBytesOnHost = 0;

        /* error codes from cusolverMp (device) */
        int* d_syevdInfo = NULL;

        /* error codes from cusolverMp (host) */
        int h_syevdInfo = 0;

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        /* Single process per device */
        assert((numRowDevices * numColDevices) <= commSize);

        /* =========================================== */
        /*          Create inputs on master rank       */
        /* =========================================== */

        const int64_t lda   = (ia - 1) + n;
        const int64_t colsA = (ja - 1) + n;
        const int64_t ldq   = (iq - 1) + n;
        const int64_t colsQ = (jq - 1) + n;

        double* h_A = NULL;
        double* h_Q = NULL;
        double* h_D = NULL;

        if (rank == 0)
        {
            /* allocate host workspace */
            h_A = (double*)malloc(lda * colsA * sizeof(double));
            h_Q = (double*)malloc(ldq * colsQ * sizeof(double));
            h_D = (double*)malloc(n * 1 * sizeof(double));

            /* reset host workspace */
            memset(h_A, 0xFF, lda * colsA * sizeof(double));
            memset(h_Q, 0xFF, ldq * colsQ * sizeof(double));
            memset(h_D, 0xFF, n * 1 * sizeof(double));

            double* _A = &h_A[(ia - 1) + (ja - 1) * lda];

            /* Set A[ia:ia+n, ja:ja+n] = diagonal dominant lower triangular matrix */
            generate_diagonal_dominant_symmetric_matrix(n, _A, lda);

            /* print input matrices */
            if (opts.verbose)
            {
                print_host_matrix(lda, colsA, h_A, lda, "Input matrix A");
            }
        }

        /* compute process grid index */
        int myRowRank, myColRank;
        if (gridLayout == CUSOLVERMP_GRID_MAPPING_COL_MAJOR)
        {
            myRowRank = rank % numRowDevices;
            myColRank = rank / numRowDevices;
        }
        else
        {
            myRowRank = rank / numColDevices;
            myColRank = rank % numColDevices;
        }

        /* compute the load leading dimension of the device buffers */
        const int64_t llda       = cusolverMpNUMROC(lda, mbA, myRowRank, rsrca, numRowDevices);
        const int64_t localColsA = cusolverMpNUMROC(colsA, nbA, myColRank, csrca, numColDevices);

        const int64_t lldq       = cusolverMpNUMROC(ldq, mbQ, myRowRank, rsrcq, numRowDevices);
        const int64_t localColsQ = cusolverMpNUMROC(colsQ, nbQ, myColRank, csrcq, numColDevices);

        /* Allocate global d_A */
        cudaStat = cudaMalloc((void**)&d_A, llda * localColsA * sizeof(double));
        assert(cudaStat == cudaSuccess);

        /* Allocate global d_B */
        cudaStat = cudaMalloc((void**)&d_Q, lldq * localColsQ * sizeof(double));
        assert(cudaStat == cudaSuccess);

        /* Allocate global d_D */
        cudaStat = cudaMalloc((void**)&d_D, n * 1 * sizeof(double));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*          CREATE GRID DESCRIPTORS            */
        /* =========================================== */
        cusolverStat =
                cusolverMpCreateDeviceGrid(cusolverMpHandle, &gridA, comm, numRowDevices, numColDevices, gridLayout);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat =
                cusolverMpCreateDeviceGrid(cusolverMpHandle, &gridQ, comm, numRowDevices, numColDevices, gridLayout);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*        CREATE MATRIX DESCRIPTORS            */
        /* =========================================== */
        cusolverStat = cusolverMpCreateMatrixDesc(&descA, gridA, CUDA_R_64F, lda, colsA, mbA, nbA, rsrca, csrca, llda);

        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpCreateMatrixDesc(&descQ, gridQ, CUDA_R_64F, ldq, colsQ, mbQ, nbQ, rsrcq, csrcq, lldq);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*             ALLOCATE D_INFO                 */
        /* =========================================== */

        cudaStat = cudaMalloc((void**)&d_syevdInfo, sizeof(int));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*                RESET D_INFO                 */
        /* =========================================== */

        cudaStat = cudaMemset(d_syevdInfo, 0, sizeof(int));
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
        /* =========================================== */

        cusolverStat = cusolverMpSyevd_bufferSize(cusolverMpHandle,
                                                  &compz,
                                                  CUBLAS_FILL_MODE_LOWER,
                                                  n,
                                                  d_A,
                                                  ia,
                                                  ja,
                                                  descA,
                                                  d_D,
                                                  d_Q,
                                                  iq,
                                                  jq,
                                                  descQ,
                                                  CUDA_R_64F,
                                                  &syevdWorkspaceInBytesOnDevice,
                                                  &syevdWorkspaceInBytesOnHost);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*         ALLOCATE Psyevd WORKSPACE            */
        /* =========================================== */

        cudaStat = cudaMalloc((void**)&d_syevdWork, syevdWorkspaceInBytesOnDevice);
        assert(cudaStat == cudaSuccess);

        h_syevdWork = (void*)malloc(syevdWorkspaceInBytesOnHost);
        assert(h_syevdWork != NULL);

        /* =========================================== */
        /*      SCATTER MATRICES A AND B FROM MASTER   */
        /* =========================================== */
        cusolverStat = cusolverMpMatrixScatterH2D(cusolverMpHandle,
                                                  lda,
                                                  colsA,
                                                  (void*)d_A,
                                                  1,
                                                  1,
                                                  descA,
                                                  0, /* root rank */
                                                  (void*)h_A,
                                                  lda);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpMatrixScatterH2D(cusolverMpHandle,
                                                  ldq,
                                                  colsQ,
                                                  (void*)d_Q,
                                                  1,
                                                  1,
                                                  descQ,
                                                  0, /* root rank */
                                                  (void*)h_Q,
                                                  ldq);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync wait for data to arrive to device */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*                   CALL psyevd               */
        /* =========================================== */

        cusolverStat = cusolverMpSyevd(cusolverMpHandle,
                                       &compz,
                                       CUBLAS_FILL_MODE_LOWER,
                                       n,
                                       d_A,
                                       ia,
                                       ja,
                                       descA,
                                       d_D,
                                       d_Q,
                                       iq,
                                       jq,
                                       descQ,
                                       CUDA_R_64F,
                                       d_syevdWork,
                                       syevdWorkspaceInBytesOnDevice,
                                       h_syevdWork,
                                       syevdWorkspaceInBytesOnHost,
                                       d_syevdInfo);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync after cusolverMpsyevd */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* copy d_syevdInfo to host */
        cudaStat = cudaMemcpyAsync(&h_syevdInfo, d_syevdInfo, sizeof(int), cudaMemcpyDeviceToHost, localStream);
        assert(cudaStat == cudaSuccess);

        /* wait for d_syevdInfo copy */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* check return value of cusolverMpsyevd */
        assert(h_syevdInfo == 0);

        /* =========================================== */
        /*      GATHER MATRICES A AND Q FROM MASTER    */
        /* =========================================== */

        /* Copy solution to h_Q */
        cusolverStat = cusolverMpMatrixGatherD2H(cusolverMpHandle,
                                                 ldq,
                                                 colsQ,
                                                 (void*)d_Q,
                                                 1,
                                                 1,
                                                 descQ,
                                                 0, /* master rank, destination */
                                                 (void*)h_Q,
                                                 ldq);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync wait for data to arrive to host */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* =========================================== */
        /*        CLEAN UP HOST WORKSPACE ON MASTER    */
        /* =========================================== */
        if (rank == 0)
        {
            if (h_A)
            {
                free(h_A);
                h_A = NULL;
            }

            if (h_Q)
            {
                free(h_Q);
                h_Q = NULL;
            }

            if (h_D)
            {
                free(h_D);
                h_D = NULL;
            }
        }

        /* =========================================== */
        /*           DESTROY MATRIX DESCRIPTORS        */
        /* =========================================== */

        cusolverStat = cusolverMpDestroyMatrixDesc(descA);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyMatrixDesc(descQ);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*             DESTROY MATRIX GRIDS            */
        /* =========================================== */

        cusolverStat = cusolverMpDestroyGrid(gridA);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverMpDestroyGrid(gridQ);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* =========================================== */
        /*          DEALLOCATE DEVICE WORKSPACE        */
        /* =========================================== */

        if (d_A)
        {
            cudaStat = cudaFree(d_A);
            assert(cudaStat == cudaSuccess);
            d_A = NULL;
        }

        if (d_Q)
        {
            cudaStat = cudaFree(d_Q);
            assert(cudaStat == cudaSuccess);
            d_Q = NULL;
        }

        if (d_D)
        {
            cudaStat = cudaFree(d_D);
            assert(cudaStat == cudaSuccess);
            d_D = NULL;
        }

        if (d_syevdWork)
        {
            cudaStat = cudaFree(d_syevdWork);
            assert(cudaStat == cudaSuccess);
            d_syevdWork = NULL;
        }

        if (d_syevdInfo)
        {
            cudaStat = cudaFree(d_syevdInfo);
            assert(cudaStat == cudaSuccess);
            d_syevdInfo = NULL;
        }

        /* =========================================== */
        /*         DEALLOCATE HOST WORKSPACE           */
        /* =========================================== */
        if (h_syevdWork)
        {
            free(h_syevdWork);
            h_syevdWork = NULL;
        }

        /* =========================================== */
        /*                      CLEANUP                */
        /* =========================================== */

        /* Destroy cusolverMp handle */
        cusolverStat = cusolverMpDestroy(cusolverMpHandle);
        assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* sync before ncclCommDestroy */
        cudaStat = cudaStreamSynchronize(localStream);
        assert(cudaStat == cudaSuccess);

        /* destroy nccl communicator */
        ncclStat = ncclCommDestroy(comm);
        assert(ncclStat == ncclSuccess);

        /* destroy user stream */
        cudaStat = cudaStreamDestroy(localStream);
        assert(cudaStat == cudaSuccess);
    }

    /* MPI barrier before MPI_Finalize */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Finalize MPI environment */
    MPI_Finalize();

    if (mpiRank == 0)
    {
        printf("[SUCCEEDED]\n");
    }

    return 0;
};