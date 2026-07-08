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

/* compute |x|_inf */
static float normI(int64_t m, int64_t n, int64_t lda, const float* A)
{
    float max_nrm = 0;

    for (int64_t col = 0; col < n; col++)
    {
        float err = 0;
        for (int64_t row = 0; row < m; row++)
        {
            err += fabsf(A[row + col * lda]);
        }

        max_nrm = fmaxf(max_nrm, err);
    }

    return max_nrm;
}

static void generate_diagonal_dominant_symmetric_matrix(int64_t n, float* A, int64_t lda)
{
    /* set A[0:n, 0:n] = 0 */
    for (int64_t j = 0; j < n; j++)
    {
        float sum = 0;
        for (int64_t i = 0; i < n; i++)
        {
            if (i < j)
            {
                A[i + j * lda] = A[j + i * lda];
            }
            else
            {
                A[i + j * lda] = (float)(rand()) / RAND_MAX;
            }
            sum += A[i + j * lda];
        }

        A[j + j * lda] = 2 * sum;
    }
}

/* Print matrix */
static void print_host_matrix(int64_t m, int64_t n, float* A, int64_t lda, const char* msg)
{
    printf("print_host_matrix : %s\n", msg);

    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
        {
            printf("%.2f  ", A[i + j * lda]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    Options opts = { .m           = 1,
                     .n           = 10,
                     .nrhs        = 1,
                     .mbA         = 2,
                     .nbA         = 2,
                     .mbB         = 2,
                     .nbB         = 2,
                     .mbQ         = 2,
                     .nbQ         = 2,
                     .ia          = 3,
                     .ja          = 1,
                     .ib          = 3,
                     .jb          = 1,
                     .iq          = 1,
                     .jq          = 1,
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
    const int64_t n    = opts.n;
    const int64_t nrhs = opts.nrhs;

    /* Offsets of A and B matrices (base-1) */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbB = opts.mbB;
    const int64_t nbB = opts.nbB;

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
    const uint32_t rsrcb = 0;
    const uint32_t csrcb = 0;

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
    cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
    ncclResult_t     ncclStat     = ncclSuccess;

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
    cusolverMpGrid_t gridB = NULL;

    /* cusolverMp matrix descriptors */
    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverMpMatrixDescriptor_t descB = NULL;

    /* Distributed matrices */
    void* d_A = NULL;
    void* d_B = NULL;

    /* Distributed device workspace */
    void* d_potrfWork = NULL;
    void* d_potrsWork = NULL;

    /* Distributed host workspace */
    void* h_potrfWork = NULL;
    void* h_potrsWork = NULL;

    /* size of workspace on device */
    size_t potrfWorkspaceInBytesOnDevice = 0;
    size_t potrsWorkspaceInBytesOnDevice = 0;

    /* size of workspace on host */
    size_t potrfWorkspaceInBytesOnHost = 0;
    size_t potrsWorkspaceInBytesOnHost = 0;

    /* error codes from cusolverMp (device) */
    int* d_potrfInfo = NULL;
    int* d_potrsInfo = NULL;

    /* error codes from cusolverMp (host) */
    int h_potrfInfo = 0;
    int h_potrsInfo = 0;

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    /* cusolverMppotrs only supports nrhs == 1 at this point. */
    SAMPLE_ASSERT(nrhs == 1);

    /* Single process per device */
    SAMPLE_ASSERT((nprow * npcol) == commSize);

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    const int64_t lda   = (ia - 1) + n;
    const int64_t colsA = (ja - 1) + n;
    const int64_t ldb   = (ib - 1) + n;
    const int64_t colsB = (jb - 1) + nrhs;

    float* h_A = NULL;
    float* h_B = NULL;
    float* h_X = NULL;

    if (rank == 0)
    {
        /* allocate host workspace */
        h_A = (float*)malloc(lda * colsA * sizeof(float));
        h_X = (float*)malloc(ldb * colsB * sizeof(float));
        h_B = (float*)malloc(ldb * colsB * sizeof(float));

        /* reset host workspace */
        memset(h_A, 0xFF, lda * colsA * sizeof(float));
        memset(h_X, 0xFF, ldb * colsB * sizeof(float));
        memset(h_B, 0xFF, ldb * colsB * sizeof(float));

        /* pointers to the first valid entry of A and B */
        float* _A = &h_A[(ia - 1) + (ja - 1) * lda];
        float* _B = &h_B[(ib - 1) + (jb - 1) * ldb];

        /* Set A[ia:ia+n, ja:ja+n] = diagonal dominant lower triangular matrix */
        generate_diagonal_dominant_symmetric_matrix(n, _A, lda);

        /* Set B[ib:ib+n, jb] = 1 */
        for (int64_t i = 0; i < n; i++)
        {
            _B[i] = 1.0f;
        }

        /* print input matrices */
        if (opts.verbose)
        {
            print_host_matrix(lda, colsA, h_A, lda, "Input matrix A");
            print_host_matrix(ldb, colsB, h_X, ldb, "Input matrix X");
            print_host_matrix(ldb, colsB, h_B, ldb, "Input matrix B");
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

    const int64_t lldb       = cusolverMpNUMROC(ldb, mbB, myRowRank, rsrcb, nprow);
    const int64_t localColsB = cusolverMpNUMROC(colsB, nbB, myColRank, csrcb, npcol);

    /* Allocate global d_A */
    cudaStat = cudaMalloc((void**)&d_A, llda * localColsA * sizeof(float));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Allocate global d_B */
    cudaStat = cudaMalloc((void**)&d_B, lldb * localColsB * sizeof(float));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*          CREATE GRID DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateDeviceGrid(handle, &gridA, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpCreateDeviceGrid(handle, &gridB, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*        CREATE MATRIX DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateMatrixDesc(&descA, gridA, CUDA_R_32F, lda, colsA, mbA, nbA, rsrca, csrca, llda);

    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpCreateMatrixDesc(&descB, gridB, CUDA_R_32F, ldb, colsB, mbB, nbB, rsrcb, csrcb, lldb);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_potrfInfo, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&d_potrsInfo, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset(d_potrfInfo, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMemset(d_potrsInfo, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
    /* =========================================== */

    cusolverStat = cusolverMpPotrf_bufferSize(handle,
                                              CUBLAS_FILL_MODE_LOWER,
                                              n,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              CUDA_R_32F,
                                              &potrfWorkspaceInBytesOnDevice,
                                              &potrfWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpPotrs_bufferSize(handle,
                                              CUBLAS_FILL_MODE_LOWER,
                                              n,
                                              nrhs,
                                              (const void*)d_A,
                                              ia,
                                              ja,
                                              descA,
                                              d_B,
                                              ib,
                                              jb,
                                              descB,
                                              CUDA_R_32F,
                                              &potrsWorkspaceInBytesOnDevice,
                                              &potrsWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         ALLOCATE Ppotrf WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_potrfWork, potrfWorkspaceInBytesOnDevice);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_potrfWork = (void*)malloc(potrfWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(h_potrfWork != NULL);

    /* =========================================== */
    /*         ALLOCATE Ppotrs WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_potrsWork, potrsWorkspaceInBytesOnDevice);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_potrsWork = (void*)malloc(potrsWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(h_potrsWork != NULL);

    /* =========================================== */
    /*      SCATTER MATRICES A AND B FROM MASTER   */
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

    cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                              ldb,
                                              colsB,
                                              (void*)d_B,
                                              1,
                                              1,
                                              descB,
                                              0, /* root rank */
                                              (void*)h_B,
                                              ldb);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to device */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* =========================================== */
    /*                   CALL Ppotrf               */
    /* =========================================== */

    cusolverStat = cusolverMpPotrf(handle,
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   CUDA_R_32F,
                                   d_potrfWork,
                                   potrfWorkspaceInBytesOnDevice,
                                   h_potrfWork,
                                   potrfWorkspaceInBytesOnHost,
                                   d_potrfInfo);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMppotrf */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* copy d_potrfInfo to host */
    cudaStat = cudaMemcpyAsync(&h_potrfInfo, d_potrfInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* wait for d_potrfInfo copy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMppotrf */
    SAMPLE_ASSERT(h_potrfInfo == 0);

    /* =========================================== */
    /*                   CALL Ppotrs               */
    /* =========================================== */

    cusolverStat = cusolverMpPotrs(handle,
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   nrhs,
                                   (const void*)d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_B,
                                   ib,
                                   jb,
                                   descB,
                                   CUDA_R_32F,
                                   d_potrsWork,
                                   potrsWorkspaceInBytesOnDevice,
                                   h_potrsWork,
                                   potrsWorkspaceInBytesOnHost,
                                   d_potrsInfo);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMppotrs */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* copy d_potrsInfo to host */
    cudaStat = cudaMemcpyAsync(&h_potrsInfo, d_potrsInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* wait for d_potrsInfo copy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMppotrf */
    SAMPLE_ASSERT(h_potrsInfo == 0);

    /* =========================================== */
    /*      GATHER MATRICES A AND B FROM MASTER    */
    /* =========================================== */

    /* Copy solution to h_X */
    cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                             ldb,
                                             colsB,
                                             (void*)d_B,
                                             1,
                                             1,
                                             descB,
                                             0, /* master rank, destination */
                                             (void*)h_X,
                                             ldb);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to host */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*            CHECK RESIDUAL ON MASTER         */
    /* =========================================== */
    if (rank == 0)
    {
        /* print input matrices */
        if (opts.verbose)
        {
            print_host_matrix(ldb, colsB, h_X, ldb, "Output matrix X");
        }

        /* pointers to the first valid entry of A, B and X */
        float* _A = &h_A[(ia - 1) + (ja - 1) * lda];
        float* _X = &h_X[(ib - 1) + (jb - 1) * ldb];
        float* _B = &h_B[(ib - 1) + (jb - 1) * ldb];

        /* measure residual error |b - A*x| */
        float max_err = 0;
        for (int row = 0; row < n; row++)
        {
            float sum = 0.0f;
            for (int col = 0; col < n; col++)
            {
                float Aij = _A[row + col * lda];
                float xj  = _X[col];
                sum += Aij * xj;
            }
            float bi  = _B[row];
            float err = fabsf(bi - sum);

            max_err = fmaxf(max_err, err);
        }

        float x_nrm_inf = normI(n, 1, ldb, _X);
        float b_nrm_inf = normI(n, 1, ldb, _B);
        float A_nrm_inf = normI(n, n, lda, _A);
        float rel_err   = max_err / (A_nrm_inf * x_nrm_inf + b_nrm_inf);
        float tol       = 1.0e-4f;
        int   ok        = (rel_err < tol);

        printf("\n|b - A*x|_inf = %E\n", max_err);
        printf("|x|_inf = %E\n", x_nrm_inf);
        printf("|b|_inf = %E\n", b_nrm_inf);
        printf("|A|_inf = %E\n", A_nrm_inf);

        /* relative error is around machine zero  */
        /* the user can use |b - A*x|/(n*|A|*|x|+|b|) as well */
        printf("|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);
        printf("  POTRF/POTRS check: %s  (threshold: %E)\n", ok ? "PASS" : "FAIL", tol);
        sample_ok = sample_ok && ok;
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
        if (h_X)
        {
            free(h_X);
            h_X = NULL;
        }
        if (h_B)
        {
            free(h_B);
            h_B = NULL;
        }
    }

    /* =========================================== */
    /*           DESTROY MATRIX DESCRIPTORS        */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyMatrixDesc(descB);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid(gridA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyGrid(gridB);
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

    if (d_potrfWork)
    {
        cudaStat = cudaFree(d_potrfWork);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_potrfWork = NULL;
    }

    if (d_potrsWork)
    {
        cudaStat = cudaFree(d_potrsWork);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_potrsWork = NULL;
    }

    if (d_potrfInfo)
    {
        cudaStat = cudaFree(d_potrfInfo);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_potrfInfo = NULL;
    }

    if (d_potrsInfo)
    {
        cudaStat = cudaFree(d_potrsInfo);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_potrsInfo = NULL;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */
    /* =========================================== */
    if (h_potrfWork)
    {
        free(h_potrfWork);
        h_potrfWork = NULL;
    }
    if (h_potrsWork)
    {
        free(h_potrsWork);
        h_potrsWork = NULL;
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
