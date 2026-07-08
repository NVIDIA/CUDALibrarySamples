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


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <mpi.h>

#include <cuda_bf16.h>
#include <cusolverMp.h>

#include "helpers.h"

/* types */
typedef __nv_bfloat16 value_type;
typedef float         compute_type;
cudaDataType_t        cudaValueType   = CUDA_R_16BF;
cudaDataType_t        cudaComputeType = CUDA_R_32F;

/* set random matrix A, dim(A) = [m,n] */
static void generate_random_matrix(const int64_t m, const int64_t n, value_type* A, const int64_t lda)
{
    for (int64_t j = 0; j < n; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            A[i + j * lda] = static_cast<value_type>((double)(rand()) / RAND_MAX);
        }
    }
}

/* print matrix */
static void print_host_matrix(int64_t m, int64_t n, value_type* A, int64_t lda, const char* msg)
{
    printf("print_host_matrix : %s\n", msg);

    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
        {
            printf("%.2lf  ", static_cast<double>(A[i + j * lda]));
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    Options opts = { .m           = 16384,
                     .n           = 8192,
                     .mbA         = 32,
                     .nbA         = 32,
                     .ia          = 1,
                     .ja          = 1,
                     .p           = 2,
                     .q           = 1,
                     .grid_layout = 'C',
                     .verbose     = false };

    parse(&opts, argc, argv);
    validate(&opts);

    int sample_ok = 1;

    /* Initialize MPI library */
    MPI_Init(NULL, NULL);

    /* Define a problem; A is a m x n matrix */
    const int64_t m = opts.m;
    const int64_t n = opts.n;

    /* Offsets of A and B matrices (base-1) */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;

    /* Define grid of processors */
    const int nprow = opts.p;
    const int npcol = opts.q;

    /* Convert grid layout to cusolverMp grid mapping */
    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t rsrca = 0;
    const uint32_t csrca = 0;

    /* Get MPI rank id and communicator size. */
    int commSize, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) print(&opts);

    /*
     * Initialize device context for this process
     */
    int         localRank = getLocalRank();
    cudaError_t cudaStat  = cudaSetDevice(localRank);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Error codes */
    cusolverStatus_t cusolverStat;
    ncclResult_t     ncclStat;

    /* Create communicator */
    ncclComm_t ncclComm = createNcclComm(commSize, rank);

    /* Create local stream */
    cudaStream_t stream = NULL;
    cudaStat            = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Initialize cusolverMp library handle */
    cusolverMpHandle_t handle = NULL;
    cusolverStat              = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* cusolverMp grids */
    cusolverMpGrid_t gridA = NULL;

    /* cusolverMp matrix descriptors */
    cusolverMpMatrixDescriptor_t descA = NULL;

    /* cusolverMp NS descriptor */
    cusolverMpNewtonSchulzDescriptor_t descNS = NULL;

    /* Distributed matrices */
    void* d_A = NULL;

    /* Distributed device workspace */
    void* d_work = NULL;

    /* Distributed host workspace */
    void* h_work = NULL;

    /* size of workspace on device */
    size_t workspaceInBytesOnDevice = 0;

    /* size of workspace on host */
    size_t workspaceInBytesOnHost = 0;

    /* error codes from cusolverMp (device) */
    int* d_nsInfo = NULL;

    /* error codes from cusolverMp (host) */
    int h_nsInfo = 0;

    /* Single process per device */
    SAMPLE_ASSERT((nprow * npcol) == commSize);
    SAMPLE_ASSERT(npcol == 1); /* the current impl supports 1d distribution only */

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */
    const int64_t lda   = (ia - 1) + m;
    const int64_t colsA = (ja - 1) + n;

    value_type* h_A = NULL;

    if (rank == 0)
    {
        /* allocate host workspace */
        h_A = (value_type*)malloc(lda * colsA * sizeof(value_type));
        SAMPLE_ASSERT(h_A != NULL);

        /* reset host workspace */
        memset(h_A, 0xFF, lda * colsA * sizeof(value_type));

        value_type* _A = &h_A[(ia - 1) + (ja - 1) * lda];

        /* Set A[ia:ia+n, ja:ja+n] = diagonal dominant lower triangular matrix */
        generate_random_matrix(m, n, _A, lda);

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
        myRowRank = rank % nprow;
        myColRank = rank / nprow;
    }
    else
    {
        myRowRank = rank / npcol;
        myColRank = rank % npcol;
    }

    /* compute the load leading dimension of the device buffers */
    const int64_t llda       = cusolverMpNUMROC(lda, mbA, myRowRank, rsrca, nprow);
    const int64_t localColsA = cusolverMpNUMROC(colsA, nbA, myColRank, csrca, npcol);

    /* Allocate global d_A */
    cudaStat = cudaMalloc((void**)&d_A, llda * localColsA * sizeof(value_type));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*          CREATE GRID DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateDeviceGrid(handle, &gridA, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*        CREATE MATRIX DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateMatrixDesc(&descA, gridA, cudaValueType, lda, colsA, mbA, nbA, rsrca, csrca, llda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*      SCATTER MATRICES A FROM MASTER         */
    /* =========================================== */
    cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                              lda,
                                              colsA,
                                              (void*)d_A,
                                              1,
                                              1,
                                              descA,
                                              0, /* root rank */
                                              (void*)h_A,
                                              lda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to device */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*        CREATE NS DESCRIPTOR                 */
    /* =========================================== */
    cusolverStat = cusolverMpNewtonSchulzDescriptorCreate(&descNS);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Use default param values using null pointer or empty descNS.
     * The following block is intentionally disabled; it shows non-default
     * descriptor values for reference. */
#if 0
        /* Disable input normalization (default: 1). Use only when the caller
         * has already normalized X by its Frobenius norm. */
        int normalize = 0;
        cusolverStat = cusolverMpNewtonSchulzDescriptorSetAttribute(
                descNS,
                CUSOLVERMP_NEWTON_SCHULZ_DESCRIPTOR_ATTRIBUTE_NORMALIZE,
                &normalize,
                sizeof(int));
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        /* Use compute type for the intermediate X^T X reduction
         * (default: 0, uses the value type). */
        int reduce_via_compute_type = 1;
        cusolverStat = cusolverMpNewtonSchulzDescriptorSetAttribute(
                descNS,
                CUSOLVERMP_NEWTON_SCHULZ_DESCRIPTOR_ATTRIBUTE_REDUCE_VIA_COMPUTE_TYPE,
                &reduce_via_compute_type,
                sizeof(int));
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
#endif

    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_nsInfo, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset(d_nsInfo, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* quintic iterations and its params */
    /* https://leloykun.github.io/ponder/muon-opt-coeffs/#how-do-we-optimize-the-coefficients */
    /* https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L44 */
    const int64_t numNewtonSchulzIterations = 5;
    float         h_coeffs[15]              = { 4.0848, -6.8946, 2.9270,  3.9505, -6.3029, 2.6377,  3.7418, -5.5913,
                                                2.3037, 2.8769,  -3.1427, 1.2046, 2.8366,  -3.0525, 1.2012 };

    /* =========================================== */
    /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
    /* =========================================== */
    cusolverStat = cusolverMpNewtonSchulz_bufferSize(handle,
                                                     descNS,
                                                     m,
                                                     n,
                                                     d_A,
                                                     ia,
                                                     ja,
                                                     descA,
                                                     numNewtonSchulzIterations,
                                                     h_coeffs,
                                                     cudaComputeType,
                                                     &workspaceInBytesOnDevice,
                                                     &workspaceInBytesOnHost);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*          ALLOCATE and REGISTER WORKSPACE    */
    /* =========================================== */
    /* Try ncclMemAlloc + window register for best collective performance.
     * Fall back to plain cudaMalloc when symmetric memory is not supported. */
    bool useNcclMem = false;
    if (workspaceInBytesOnDevice > 0)
    {
        ncclStat = ncclMemAlloc((void**)&d_work, workspaceInBytesOnDevice);
        if (ncclStat == ncclSuccess)
        {
            cusolverStat = cusolverMpBufferRegister(gridA, d_work, workspaceInBytesOnDevice);
            if (cusolverStat == CUSOLVER_STATUS_SUCCESS)
            {
                useNcclMem = true;
            }
            else
            {
                /* Registration not supported — free symmetric memory and fall back */
                ncclMemFree(d_work);
                d_work = NULL;
            }
        }

        if (!useNcclMem)
        {
            if (rank == 0)
            {
                printf("NCCL symmetric memory not available, falling back to cudaMalloc for workspace\n");
            }
            cudaStat = cudaMalloc((void**)&d_work, workspaceInBytesOnDevice);
            SAMPLE_ASSERT(cudaStat == cudaSuccess);
        }
    }

    if (workspaceInBytesOnHost > 0)
    {
        h_work = (void*)malloc(workspaceInBytesOnHost);
        SAMPLE_ASSERT(h_work != NULL);
    }

    /* =========================================== */
    /*                   CALL NewtonSchulz         */
    /* =========================================== */
    cusolverStat = cusolverMpNewtonSchulz(handle,
                                          descNS,
                                          m,
                                          n,
                                          d_A,
                                          ia,
                                          ja,
                                          descA,
                                          numNewtonSchulzIterations,
                                          h_coeffs,
                                          cudaComputeType,
                                          d_work,
                                          workspaceInBytesOnDevice,
                                          h_work,
                                          workspaceInBytesOnHost,
                                          d_nsInfo);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMpNewtonSchulz */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* copy d_nsInfo to host */
    cudaStat = cudaMemcpyAsync(&h_nsInfo, d_nsInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* wait for d_nsInfo copy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMpNewtonSchulz */
    SAMPLE_ASSERT(h_nsInfo == 0);

    /* =========================================== */
    /*      GATHER MATRICES A TO MASTER            */
    /* =========================================== */
    cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                             lda,
                                             colsA,
                                             (void*)d_A,
                                             1,
                                             1,
                                             descA,
                                             0, /* master rank, destination */
                                             (void*)h_A,
                                             lda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to host */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0 && opts.verbose)
    {
        print_host_matrix(lda, colsA, h_A, lda, "Output matrix A");
    }

    /* =========================================== */
    /*          DEALLOCATE DEVICE WORKSPACE        */
    /* =========================================== */

    if (d_A)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_A = NULL;
    }

    if (d_work)
    {
        if (useNcclMem)
        {
            cusolverStat = cusolverMpBufferDeregister(gridA, d_work);
            SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

            ncclStat = ncclMemFree(d_work);
            SAMPLE_ASSERT(ncclStat == ncclSuccess);
        }
        else
        {
            cudaStat = cudaFree(d_work);
            SAMPLE_ASSERT(cudaStat == cudaSuccess);
        }
        d_work = NULL;
    }

    if (d_nsInfo)
    {
        cudaStat = cudaFree(d_nsInfo);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_nsInfo = NULL;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */
    /* =========================================== */
    if (h_work)
    {
        free(h_work);
        h_work = NULL;
    }

    /* =========================================== */
    /*             DESTROY NS DESCRIPTOR           */
    /* =========================================== */
    cusolverStat = cusolverMpNewtonSchulzDescriptorDestroy(descNS);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);


    /* =========================================== */
    /*           DESTROY MATRIX DESCRIPTORS        */
    /* =========================================== */
    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid(gridA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

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
