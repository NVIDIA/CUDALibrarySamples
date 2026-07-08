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


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* A is a diagonal weighted matrix */
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

        A[j + j * lda] = 10 * sum;
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
            printf("%.6e ", A[i + j * lda]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    /* Options used in this sample are marked */
    Options opts = { .m           = 256,
                     .mbA         = 32,
                     .nbA         = 32,
                     .mbB         = 1,
                     .nbB         = 1,
                     .mbZ         = 1,
                     .nbZ         = 1,
                     .ia          = 1,
                     .ja          = 1,
                     .ib          = 1,
                     .jb          = 1,
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
    const int64_t m = opts.m;

    /* Offsets of A and B matrices (base-1) */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;
    const int64_t iz = opts.iz;
    const int64_t jz = opts.jz;

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbB = opts.mbB == 1 ? mbA : opts.mbB;
    const int64_t nbB = opts.nbB == 1 ? nbA : opts.nbB;
    const int64_t mbZ = opts.mbZ == 1 ? mbA : opts.mbZ;
    const int64_t nbZ = opts.nbZ == 1 ? nbA : opts.nbZ;

    /* Define grid of processors */
    const int                     nprow = opts.p;
    const int                     npcol = opts.q;
    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t rsrca = 0;
    const uint32_t csrca = 0;
    const uint32_t rsrcb = 0;
    const uint32_t csrcb = 0;
    const uint32_t rsrcz = 0;
    const uint32_t csrcz = 0;

    /* Get MPI rank id and communicator size. */
    int commSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int verbose = opts.verbose;

    /* input argument used in this sample */
    if (rank == 0)
    {
        print(&opts);
    }

    /* error check of implementation restrictions */
    {
        /* check using the same square block sizes for all */
        const bool use_same_square_blocksize =
                ((mbA == nbA) && (mbA == mbB && mbA == nbB) && (mbA == mbZ && mbA == nbZ));
        SAMPLE_ASSERT(use_same_square_blocksize && "SYGVD sample requires matching square A/B/Z block sizes");

        /* current implementation constraint using ia=1, ja=1 */
        const bool use_unit_global_offsets = ((ia == 1 && ja == 1) && (ib == 1 && jb == 1) && (iz == 1 && jz == 1));
        SAMPLE_ASSERT(use_unit_global_offsets && "SYGVD sample requires ia=ja=ib=jb=iz=jz=1");
    }

    /*
     * Initialize device context for this process
     */
    int         localRank = getLocalRank();
    cudaError_t cudaStat  = cudaSetDevice(localRank);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

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

    /* Error codes */
    cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
    ncclResult_t     ncclStat     = ncclSuccess;

    /* Create communicator */
    ncclComm_t ncclComm = createNcclComm(commSize, rank);

    /* Create local stream */
    cudaStream_t stream = NULL;

    cudaStat = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Initialize cusolverMp library handle */
    cusolverMpHandle_t handle = NULL;

    cusolverStat = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* cusolverMp grid */
    cusolverMpGrid_t grid = NULL;

    /* cusolverMp matrix descriptors */
    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverMpMatrixDescriptor_t descB = NULL;
    cusolverMpMatrixDescriptor_t descZ = NULL;

    /* Distributed matrices */
    void* d_A = NULL;
    void* d_B = NULL;
    void* d_D = NULL;
    void* d_Z = NULL;

    /* Distributed device workspace */
    void* d_sygvdWork = NULL;

    /* Distributed host workspace */
    void* h_sygvdWork = NULL;

    /* size of workspace on device */
    size_t sygvdWorkspaceInBytesOnDevice = 0;

    /* size of workspace on host */
    size_t sygvdWorkspaceInBytesOnHost = 0;

    /* error codes from cusolverMp (device) */
    int* d_sygvdInfo = NULL;

    /* error codes from cusolverMp (host) */
    int h_sygvdInfo = 0;

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    /* Single process per device */
    SAMPLE_ASSERT((nprow * npcol) == commSize);

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    /* current implementation requires A, B, and Z matrices are aligned each other i.e., use the same ia, ja */
    const int64_t m_global = (ia - 1) + m;
    const int64_t n_global = (ja - 1) + m;

    double* h_A = NULL;
    double* h_B = NULL;
    double* h_D = NULL;
    double* h_Z = NULL;

    if (rank == 0)
    {
        /* allocate host workspace */
        h_A = (double*)malloc(m_global * n_global * sizeof(double));
        h_B = (double*)malloc(m_global * n_global * sizeof(double));
        h_D = (double*)malloc(m_global * sizeof(double));
        h_Z = (double*)malloc(m_global * n_global * sizeof(double));

        /* clean the allocated memory */
        memset(h_A, 0, m_global * n_global * sizeof(double));
        memset(h_B, 0, m_global * n_global * sizeof(double));
        memset(h_D, 0, m_global * sizeof(double));
        memset(h_Z, 0, m_global * n_global * sizeof(double));

        /* pointer offsets */
        double* AA = &h_A[(ia - 1) + (ja - 1) * m_global];
        double* BB = &h_B[(ib - 1) + (jb - 1) * m_global];

        /* set A[ia:ia+n, ja:ja+n] = diagonal dominant random lower triangular matrix */
        generate_diagonal_dominant_symmetric_matrix(m, AA, m_global);
        generate_diagonal_dominant_symmetric_matrix(m, BB, m_global);

        /* print input matrices */
        if (verbose)
        {
            print_host_matrix(m_global, n_global, h_A, m_global, "Input matrix A");
            print_host_matrix(m_global, n_global, h_B, m_global, "Input matrix B");
        }
    }

    /* compute the local dimensions device buffers */
    const int64_t m_local_A = cusolverMpNUMROC(m_global, mbA, myRowRank, rsrca, nprow);
    const int64_t n_local_A = cusolverMpNUMROC(n_global, nbA, myColRank, csrca, npcol);

    const int64_t m_local_B = cusolverMpNUMROC(m_global, mbB, myRowRank, rsrcb, nprow);
    const int64_t n_local_B = cusolverMpNUMROC(n_global, nbB, myColRank, csrcb, npcol);

    const int64_t m_local_Z = cusolverMpNUMROC(m_global, mbZ, myRowRank, rsrcz, nprow);
    const int64_t n_local_Z = cusolverMpNUMROC(n_global, nbZ, myColRank, csrcz, npcol);

    /* Allocate local d_A, d_B, d_D, d_Z */
    cudaStat = cudaMalloc((void**)&d_A, m_local_A * n_local_A * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&d_B, m_local_B * n_local_B * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&d_D, m_global * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&d_Z, m_local_Z * n_local_Z * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*          CREATE GRID DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateDeviceGrid(handle, &grid, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*        CREATE MATRIX DESCRIPTORS            */
    /* =========================================== */
    cusolverStat =
            cusolverMpCreateMatrixDesc(&descA, grid, CUDA_R_64F, m_global, n_global, mbA, nbA, rsrca, csrca, m_local_A);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat =
            cusolverMpCreateMatrixDesc(&descB, grid, CUDA_R_64F, m_global, n_global, mbB, nbB, rsrcb, csrcb, m_local_B);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat =
            cusolverMpCreateMatrixDesc(&descZ, grid, CUDA_R_64F, m_global, n_global, mbZ, nbZ, rsrcz, csrcz, m_local_Z);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_sygvdInfo, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset(d_sygvdInfo, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*      SCATTER MATRICES A AND B FROM MASTER   */
    /* =========================================== */
    cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                              m_global,
                                              n_global,
                                              (void*)d_A,
                                              ia,
                                              ja,
                                              descA,
                                              0, /* root rank */
                                              (void*)h_A,
                                              m_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to device */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                              m_global,
                                              n_global,
                                              (void*)d_B,
                                              ib,
                                              jb,
                                              descB,
                                              0, /* root rank */
                                              (void*)h_B,
                                              m_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to device */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
    /* =========================================== */

    cusolverStat = cusolverMpSygvd_bufferSize(handle,
                                              CUSOLVER_EIG_TYPE_1,
                                              CUSOLVER_EIG_MODE_VECTOR,
                                              CUBLAS_FILL_MODE_LOWER,
                                              m,
                                              ia,
                                              ja,
                                              descA,
                                              ib,
                                              jb,
                                              descB,
                                              iz,
                                              jz,
                                              descZ,
                                              CUDA_R_64F,
                                              &sygvdWorkspaceInBytesOnDevice,
                                              &sygvdWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         ALLOCATE Psygvd WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_sygvdWork, sygvdWorkspaceInBytesOnDevice);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_sygvdWork = (void*)malloc(sygvdWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(h_sygvdWork != NULL);

    /* sync wait for data to arrive to device */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*                   CALL psygvd               */
    /* =========================================== */

    cusolverStat = cusolverMpSygvd(handle,
                                   CUSOLVER_EIG_TYPE_1,
                                   CUSOLVER_EIG_MODE_VECTOR,
                                   CUBLAS_FILL_MODE_LOWER,
                                   m,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_B,
                                   ib,
                                   jb,
                                   descB,
                                   d_D,
                                   d_Z,
                                   iz,
                                   jz,
                                   descZ,
                                   CUDA_R_64F,
                                   d_sygvdWork,
                                   sygvdWorkspaceInBytesOnDevice,
                                   h_sygvdWork,
                                   sygvdWorkspaceInBytesOnHost,
                                   d_sygvdInfo);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMpsygvd */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMpsygvd */
    cudaStat = cudaMemcpyAsync(&h_sygvdInfo, d_sygvdInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_sygvdInfo == 0);

    /* =================================== */
    /*      GATHER MATRICES Z TO MASTER    */
    /* =================================== */

    /* copy eigen vectors to h_Z */
    cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                             m_global,
                                             n_global,
                                             (void*)d_Z,
                                             iz,
                                             jz,
                                             descZ,
                                             0, /* master rank, destination */
                                             (void*)h_Z,
                                             m_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    if (rank == 0)
    {
        /* copy d_D to host */
        cudaStat = cudaMemcpyAsync(h_D, d_D, m_global * sizeof(double), cudaMemcpyDeviceToHost, stream);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }

    /* sync wait for data to arrive to host */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0)
    {
        if (verbose)
        {
            print_host_matrix(m_global, 1, h_D, m_global, "Output eigen values");
            print_host_matrix(m_global, n_global, h_Z, m_global, "Output matrix eigen vectors");
        }
    }

    /* =========================================== */
    /*        CLEAN UP HOST MATRICES ON MASTER     */
    /* =========================================== */
    if (rank == 0)
    {
        if (h_A)
        {
            free(h_A);
            h_A = NULL;
        }

        if (h_B)
        {
            free(h_B);
            h_B = NULL;
        }

        if (h_D)
        {
            free(h_D);
            h_D = NULL;
        }

        if (h_Z)
        {
            free(h_Z);
            h_Z = NULL;
        }
    }

    /* The SYGVD host workspace is allocated independently on every rank. */
    if (h_sygvdWork)
    {
        free(h_sygvdWork);
        h_sygvdWork = NULL;
    }

    /* =========================================== */
    /*           DESTROY MATRIX DESCRIPTORS        */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyMatrixDesc(descB);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyMatrixDesc(descZ);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid(grid);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*          DEALLOCATE DEVICE WORKSPACE        */
    /* =========================================== */

    if (d_A)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_A = NULL;
    }

    if (d_B)
    {
        cudaStat = cudaFree(d_B);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_B = NULL;
    }

    if (d_D)
    {
        cudaStat = cudaFree(d_D);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_D = NULL;
    }

    if (d_Z)
    {
        cudaStat = cudaFree(d_Z);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_Z = NULL;
    }

    if (d_sygvdWork)
    {
        cudaStat = cudaFree(d_sygvdWork);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_sygvdWork = NULL;
    }

    if (d_sygvdInfo)
    {
        cudaStat = cudaFree(d_sygvdInfo);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_sygvdInfo = NULL;
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
