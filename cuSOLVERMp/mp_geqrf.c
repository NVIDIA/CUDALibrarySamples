/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

static void gen_matrix(int64_t M, int64_t N, double* A, int64_t lda)
{
    /* set A[0:N, 0:N] = 0 */
    for (int64_t J = 0; J < N; J++)
    {
        for (int64_t I = 0; I < M; I++)
        {
            A[I + J * lda] = J % 6 + I % 3 + 2 * I / 7 + 3 * J / 6;
        }
    }

    /* set entries */
    const int64_t M_N_min = (M < N) ? M : N;
    for (int J = 0; J < M_N_min; J++)
    {
        /* main diagonal */
        A[((M - 1) - J) + J * lda] = 2.0;
        A[J + J * lda]             = 2.0;

        /* upper diagonal */
        if (J > 0)
        {
            A[((M - 1) - (J - 1)) + J * lda] = -1.0;
        }
        /* lower diagonal */
        if (J < (N - 1))
        {
            A[((M - 1) - (J + 1)) + J * lda] = -1.0;
        }
    }
}

/* Print matrix */
static void print_host_matrix(int64_t M, int64_t N, double* A, int64_t lda, const char* msg)
{
    if (M * N > 2000) return;
    printf("print_host_matrix : %s\n", msg);

    for (int64_t i = 0; i < M; i++)
    {
        for (int64_t j = 0; j < N; j++)
        {
            printf("%.2lf  ", A[i + j * lda]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    Options opts = { .m           = 10,
                     .n           = 10,
                     .nrhs        = 1,
                     .mbA         = 24,
                     .nbA         = 24,
                     .mbB         = 24,
                     .nbB         = 24,
                     .mbQ         = 24,
                     .nbQ         = 24,
                     .mbZ         = 24,
                     .nbZ         = 24,
                     .ia          = 1,
                     .ja          = 1,
                     .ib          = 1,
                     .jb          = 1,
                     .iq          = 1,
                     .jq          = 1,
                     .iz          = 1,
                     .jz          = 1,
                     .p           = 2,
                     .q           = 1,
                     .grid_layout = 'C',
                     .verbose     = false };

    parse(&opts, argc, argv);
    validate(&opts);

    int sample_ok = 1;

    /* Initialize MPI library */
    MPI_Init(NULL, NULL);

    /* Define dimensions, block sizes and offsets of A and B matrices */
    const int64_t M = opts.m;
    const int64_t N = opts.n;

    /* Tile sizes */
    const int64_t MA = opts.mbA;
    const int64_t NA = opts.nbA;

    /* Offsets of A and B matrices (base-1) */
    const int64_t IA = opts.ia;
    const int64_t JA = opts.ja;

    /* Define grid of processors */
    const int nprow = opts.p;
    const int npcol = opts.q;

    /* Convert grid layout to cusolverMp grid mapping */
    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t RSRCA = 0;
    const uint32_t CSRCA = 0;
    SAMPLE_ASSERT(RSRCA == 0 && CSRCA == 0); // only RSRCA==0 and CSRC==0 are supported

    /* Get MPI rank id and communicator size. */
    int commSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) print(&opts);

    /* Library handles */
    cusolverMpHandle_t handle = NULL;

    /* Error codes */
    cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
    ncclResult_t     ncclStat     = ncclSuccess;
    cudaError_t      cudaStat     = cudaSuccess;

    /* User defined stream */
    cudaStream_t stream = NULL;

    /*
     * localDeviceId is the deviceId from rank's point of view. This is
     * system-dependent. For example, setting one device per process,
     * Summit always sees the local device as device 0.
     */
    const int localDeviceId = getLocalRank();

    cudaStat = cudaSetDevice(localDeviceId);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Create communicator */
    ncclComm_t ncclComm = createNcclComm(commSize, rank);

    /* Create local stream */
    cudaStat = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Initialize cusolverMp library handle */
    cusolverStat = cusolverMpCreate(&handle, localDeviceId, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* cusolverMp grids */
    cusolverMpGrid_t gridA = NULL;

    /* cusolverMp matrix descriptors */
    cusolverMpMatrixDescriptor_t descrA = NULL;

    /* Distributed matrices */
    void* d_A   = NULL;
    void* d_tau = NULL;

    /* Distributed device workspace */
    void* d_work_geqrf = NULL;

    /* Distributed host workspace */
    void* h_work_geqrf = NULL;

    /* size of workspace on device */
    size_t workspaceInBytesOnDevice_geqrf = 0;

    /* size of workspace on host */
    size_t workspaceInBytesOnHost_geqrf = 0;

    /* error codes from cusolverMp (device) */
    int* d_info_geqrf = NULL;

    /* error codes from cusolverMp (host) */
    int h_info_geqrf = 0;

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    /* Single process per device */
    SAMPLE_ASSERT((nprow * npcol) == commSize);

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    const int64_t lda   = (IA - 1) + M;
    const int64_t colsA = (JA - 1) + N;

    double* h_A  = NULL;
    double* h_QR = NULL;

    void* d_global_Q   = NULL;
    void* d_global_tau = NULL;

    if (rank == 0)
    {
        /* allocate host workspace */
        h_A  = (double*)malloc(lda * colsA * sizeof(double));
        h_QR = (double*)malloc(lda * colsA * sizeof(double));
        memset(h_A, 0, lda * colsA * sizeof(double));
        double* _A = &h_A[(IA - 1) + (JA - 1) * lda]; // first entry of A
        gen_matrix(M, N, _A, lda);
        if (opts.verbose)
        {
            print_host_matrix(lda, colsA, h_A, lda, "Input matrix A");
        }
    }

    /* =========================================== */
    /*            COMPUTE LLDA AND LLDB            */
    /* =========================================== */

    /* compute process grid index */
    int myRowRank, myColRank;
    if (gridLayout == CUSOLVERMP_GRID_MAPPING_COL_MAJOR)
    {
        myRowRank = rank % nprow;
        myColRank = rank / nprow;
    }
    else
    {
        myRowRank = rank / npcol;
        myColRank = rank % npcol;
    }

    /*
     * Compute number of tiles per rank to store local portion of A
     *
     * Current implementation has the following restrictions on the size of
     * the device buffer size:
     *  - Rows of device buffer is a multiple of MA
     *  - Cols of device buffer is a multiple of NA
     *
     * This limitation will be removed on the official release.
     */
    const int64_t LLDA       = cusolverMpNUMROC(lda, MA, myRowRank, RSRCA, nprow);
    const int64_t localColsA = cusolverMpNUMROC(colsA, NA, myColRank, CSRCA, npcol);

    /*
     * Compute number of tiles per rank to store local portion of B
     *
     * Current implementation has the following restrictions on the size of
     * the device buffer size:
     *  - Rows of device buffer is a multiple of MB
     *  - Cols of device buffer is a multiple of NB
     *
     * This limitation will be removed on the official release.
     */
    /* Allocate global d_A */
    cudaStat = cudaMalloc((void**)&d_A, localColsA * LLDA * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*          CREATE GRID DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateDeviceGrid(handle, &gridA, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*        CREATE MATRIX DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateMatrixDesc(
            &descrA, gridA, CUDA_R_64F, (IA - 1) + M, (JA - 1) + N, MA, NA, RSRCA, CSRCA, LLDA);

    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Allocate global d_tau */
    cudaStat = cudaMalloc((void**)&d_tau, localColsA * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_info_geqrf, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset(d_info_geqrf, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
    /* =========================================== */

    cusolverStat = cusolverMpGeqrf_bufferSize(handle,
                                              M,
                                              N,
                                              d_A,
                                              IA,
                                              JA,
                                              descrA,
                                              CUDA_R_64F,
                                              &workspaceInBytesOnDevice_geqrf,
                                              &workspaceInBytesOnHost_geqrf);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         ALLOCATE Pgeqrf WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_work_geqrf, workspaceInBytesOnDevice_geqrf);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_work_geqrf = (void*)malloc(workspaceInBytesOnHost_geqrf);
    SAMPLE_ASSERT(h_work_geqrf != NULL);


    /* =========================================== */
    /*      SCATTER MATRICES A AND B FROM MASTER   */
    /* =========================================== */
    cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                              lda,
                                              colsA,
                                              (void*)d_A, /* routine requires void** */
                                              1,
                                              1,
                                              descrA,
                                              0, /* root rank */
                                              (void*)h_A,
                                              lda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to device */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* =========================================== */
    /*                   CALL Pgeqrf               */
    /* =========================================== */

    cusolverStat = cusolverMpGeqrf(handle,
                                   M,
                                   N,
                                   d_A,
                                   IA,
                                   JA,
                                   descrA,
                                   d_tau,
                                   CUDA_R_64F,
                                   d_work_geqrf,
                                   workspaceInBytesOnDevice_geqrf,
                                   h_work_geqrf,
                                   workspaceInBytesOnHost_geqrf,
                                   d_info_geqrf);

    /* sync after cusolverMpgeqrf */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* copy d_info_geqrf to host */
    cudaStat = cudaMemcpyAsync(&h_info_geqrf, d_info_geqrf, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* make h_info_geqrf visible on the host, then verify the factorization succeeded */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info_geqrf == 0);

    /* =========================================== */
    /*      GATHER MATRICES A AND B FROM MASTER    */
    /* =========================================== */

    /* Copy solution to h_A */
    cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                             lda,
                                             colsA,
                                             (void*)d_A,
                                             1,
                                             1,
                                             descrA,
                                             0, /* master rank, destination */
                                             (void*)h_QR,
                                             lda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to host */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*            PRINT A ON RANK 0                */
    /* =========================================== */
    if (rank == 0)
    {
        if (opts.verbose)
        {
            print_host_matrix(lda, colsA, h_QR, lda, "Output matrix QR");
        }
    }

    // allocate global GPU arrays and copy h_A to d_global_Q
    if (rank == 0)
    {
        cudaStat = cudaMalloc((void**)&d_global_Q, lda * colsA * sizeof(double));
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        cudaStat = cudaMalloc((void**)&d_global_tau, colsA * sizeof(double));
        SAMPLE_ASSERT(cudaStat == cudaSuccess);

        cudaStat = cudaMemcpy(d_global_Q, h_A, sizeof(double) * lda * colsA, cudaMemcpyHostToDevice);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);

        // compare to 1 gpu cusolverDnGeqrf
        cusolverDnParams_t dn_geqrf_params = NULL;
        cusolverStat                       = cusolverDnCreateParams(&dn_geqrf_params);

        cusolverDnHandle_t cudenseHandle;
        cusolverStat = cusolverDnCreate(&cudenseHandle);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        cusolverStat = cusolverDnSetStream(cudenseHandle, stream);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        const void* d_global_Q_ptr = (double*)d_global_Q + ((IA - 1) + (JA - 1) * lda);

        size_t workspaceInBytesOnDevice_geqrf_dn = 0;
        size_t workspaceInBytesOnHost_geqrf_dn   = 0;

        cusolverStat = cusolverDnXgeqrf_bufferSize(cudenseHandle,
                                                   dn_geqrf_params,
                                                   M,
                                                   N,
                                                   CUDA_R_64F,
                                                   (void*)d_global_Q_ptr,
                                                   lda,
                                                   CUDA_R_64F,
                                                   (void*)d_global_tau,
                                                   CUDA_R_64F,
                                                   &workspaceInBytesOnDevice_geqrf_dn,
                                                   &workspaceInBytesOnHost_geqrf_dn);

        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        if (workspaceInBytesOnDevice_geqrf_dn > workspaceInBytesOnDevice_geqrf)
        {
            cudaStat = cudaFree(d_work_geqrf);
            SAMPLE_ASSERT(cudaStat == cudaSuccess);
            cudaStat = cudaMalloc((void**)&d_work_geqrf, workspaceInBytesOnDevice_geqrf_dn);
            SAMPLE_ASSERT(cudaStat == cudaSuccess);
            workspaceInBytesOnDevice_geqrf = workspaceInBytesOnDevice_geqrf_dn;
        }

        if (workspaceInBytesOnHost_geqrf_dn > workspaceInBytesOnHost_geqrf)
        {
            free(h_work_geqrf);
            h_work_geqrf = (void*)malloc(workspaceInBytesOnHost_geqrf_dn);
            SAMPLE_ASSERT(h_work_geqrf != NULL);
            workspaceInBytesOnHost_geqrf = workspaceInBytesOnHost_geqrf_dn;
        }


        cusolverStat = cusolverDnXgeqrf( // overwrites A
                cudenseHandle,
                dn_geqrf_params,
                M,
                N,
                CUDA_R_64F,
                (void*)d_global_Q_ptr, // in/out
                lda,
                CUDA_R_64F,
                (void*)d_global_tau,
                CUDA_R_64F,
                (void*)d_work_geqrf,
                workspaceInBytesOnDevice_geqrf,
                (void*)h_work_geqrf,
                workspaceInBytesOnHost_geqrf,
                d_info_geqrf);

        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cudaStat = cudaMemcpy(h_A, d_global_Q, sizeof(double) * lda * colsA, cudaMemcpyDeviceToHost);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);

        if (opts.verbose)
        {
            print_host_matrix(lda, colsA, h_A, lda, "A after cusolverDnXgeqrf");
        }

        int passed     = 0;
        int failed     = 0;
        int passed_abs = 0;
        int failed_abs = 0;

        for (int i = 0; i < (M + IA - 1) * (N + JA - 1); i++)
        {
            if (fabs(h_A[i] - h_QR[i]) < 0.001)
                passed++;
            else
                failed++;

            if (fabs(fabs(h_A[i]) - fabs(h_QR[i])) < 0.001)
                passed_abs++;
            else
                failed_abs++;

            h_A[i] = fabs(h_A[i] - h_QR[i]);
        }

        if (opts.verbose)
        {
            print_host_matrix(lda, colsA, h_A, lda, "difference");
        }

        sample_ok = sample_ok && (failed_abs == 0);
        printf("passed_abs %d failed_abs %d passed %d failed %d\n", passed_abs, failed_abs, passed, failed);
        printf("  GEQRF check: %s\n", failed_abs == 0 ? "PASS" : "FAIL");

        /* Clean up the additional GPU arrays */
        if (d_global_Q)
        {
            cudaStat = cudaFree(d_global_Q);
            SAMPLE_ASSERT(cudaStat == cudaSuccess);
            d_global_Q = NULL;
        }

        if (d_global_tau)
        {
            cudaStat = cudaFree(d_global_tau);
            SAMPLE_ASSERT(cudaStat == cudaSuccess);
            d_global_tau = NULL;
        }

        /* Clean up cusolverDn handle */
        cusolverStat = cusolverDnDestroy(cudenseHandle);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cusolverStat = cusolverDnDestroyParams(dn_geqrf_params);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    }

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

        if (h_QR)
        {
            free(h_QR);
            h_QR = NULL;
        }
    }

    /* =========================================== */
    /*           DESTROY MATRIX DESCRIPTORS        */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyMatrixDesc(descrA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid(gridA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*          DEALLOCATE DEVICE WORKSPACE        */
    /* =========================================== */

    if (d_A != NULL)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_A = NULL;
    }

    if (d_tau != NULL)
    {
        cudaStat = cudaFree(d_tau);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_tau = NULL;
    }

    if (d_work_geqrf != NULL)
    {
        cudaStat = cudaFree(d_work_geqrf);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_work_geqrf = NULL;
    }

    if (d_info_geqrf != NULL)
    {
        cudaStat = cudaFree(d_info_geqrf);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_info_geqrf = NULL;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */
    /* =========================================== */
    if (h_work_geqrf)
    {
        free(h_work_geqrf);
        h_work_geqrf = NULL;
    }

    /* =========================================== */
    /*                      CLEANUP                */
    /* =========================================== */

    /* Destroy cusolverMp handle */
    cusolverStat = cusolverMpDestroy(handle);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync before ncclCommDestroy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* destroy nccl communicator */
    ncclStat = ncclCommDestroy(ncclComm);
    SAMPLE_ASSERT(ncclStat == ncclSuccess);

    /* destroy user stream */
    cudaStat = cudaStreamDestroy(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* MPI barrier before MPI_Finalize */
    MPI_Barrier(MPI_COMM_WORLD);
    sample_ok = sample_all_ranks_succeeded(sample_ok);

    /* Finalize MPI environment */
    MPI_Finalize();

    if (rank == 0)
    {
        printf("%s\n", sample_ok ? "[SUCCEEDED]" : "[FAILED]");
    }

    return sample_ok ? 0 : 1;
}
