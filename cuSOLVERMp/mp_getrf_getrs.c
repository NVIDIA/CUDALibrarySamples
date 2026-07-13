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
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* compute |x|_inf */
static double vec_nrm_inf(int64_t N, const double* X)
{
    double max_nrm = 0;

    for (int64_t row = 0; row < N; row++)
    {
        double xi = X[row];
        max_nrm   = (max_nrm > fabs(xi)) ? max_nrm : fabs(xi);
    }

    return max_nrm;
}

/* A is 1D laplacian, return A[N:-1:1, :] */
static void gen_1d_laplacian_perm(int64_t N, double* A, int64_t lda)
{
    /* set A[0:N, 0:N] = 0 */
    for (int64_t J = 0; J < N; J++)
    {
        for (int64_t I = 0; I < N; I++)
        {
            A[I + J * lda] = 0.0;
        }
    }

    /* set entries */
    for (int J = 0; J < N; J++)
    {
        /* main diagonal */
        A[((N - 1) - J) + J * lda] = 2.0;

        /* upper diagonal */
        if (J > 0)
        {
            A[((N - 1) - (J - 1)) + J * lda] = -1.0;
        }
        /* lower diagonal */
        if (J < (N - 1))
        {
            A[((N - 1) - (J + 1)) + J * lda] = -1.0;
        }
    }
}

/* Print matrix */
static void print_host_matrix(int64_t M, int64_t N, double* A, int64_t lda, const char* msg)
{
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
                     .ja          = 3,
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

    int sample_ok = 1;

    /* Initialize MPI library */
    MPI_Init(NULL, NULL);

    /* Define dimensions, block sizes and offsets of A and B matrices */
    const int64_t N    = opts.n;
    const int64_t NRHS = opts.nrhs;

    /* Enable / disable pivoting */
    const int enable_pivoting = 1;

    /* Offsets of A and B matrices (base-1) */
    const int64_t IA = opts.ia;
    const int64_t JA = opts.ja;
    const int64_t IB = opts.ib;
    const int64_t JB = opts.jb;

    /* Tile sizes */
    const int64_t MA = opts.mbA;
    const int64_t NA = opts.nbA;
    const int64_t MB = opts.mbB;
    const int64_t NB = opts.nbB;

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
    const uint32_t RSRCB = 0;
    const uint32_t CSRCB = 0;

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

    /* Library handles */
    cusolverMpHandle_t handle = NULL;

    /* Error codes */
    cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
    ncclResult_t     ncclStat     = ncclSuccess;

    /* User defined stream */
    cudaStream_t stream = NULL;

    /* Create communicator */
    ncclComm_t ncclComm = createNcclComm(commSize, rank);

    /* Create local stream */
    cudaStat = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Initialize cusolverMp library handle */
    cusolverStat = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* cusolverMp grids */
    cusolverMpGrid_t gridA = NULL;
    cusolverMpGrid_t gridB = NULL;

    /* cusolverMp matrix descriptors */
    cusolverMpMatrixDescriptor_t descrA = NULL;
    cusolverMpMatrixDescriptor_t descrB = NULL;

    /* Distributed matrices */
    void*    d_A    = NULL;
    int64_t* d_ipiv = NULL;
    void*    d_B    = NULL;

    /* Distributed device workspace */
    void* d_work_getrf = NULL;
    void* d_work_getrs = NULL;

    /* Distributed host workspace */
    void* h_work_getrf = NULL;
    void* h_work_getrs = NULL;

    /* size of workspace on device */
    size_t workspaceInBytesOnDevice_getrf = 0;
    size_t workspaceInBytesOnDevice_getrs = 0;

    /* size of workspace on host */
    size_t workspaceInBytesOnHost_getrf = 0;
    size_t workspaceInBytesOnHost_getrs = 0;

    /* error codes from cusolverMp (device) */
    int* d_info_getrf = NULL;
    int* d_info_getrs = NULL;

    /* error codes from cusolverMp (host) */
    int h_info_getrf = 0;
    int h_info_getrs = 0;

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    /* cusolverMpGetrs only supports NRHS == 1 at this point. */
    SAMPLE_ASSERT(NRHS == 1);

    /* Single process per device */
    SAMPLE_ASSERT((nprow * npcol) == commSize);

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    const int64_t lda   = (IA - 1) + N;
    const int64_t colsA = (JA - 1) + N;
    const int64_t ldb   = (IB - 1) + N;
    const int64_t colsB = (JB - 1) + NRHS;

    double* h_A = NULL;
    double* h_B = NULL;
    double* h_X = NULL;

    if (rank == 0)
    {
        /* allocate host workspace */
        h_A = (double*)malloc(lda * colsA * sizeof(double));
        h_X = (double*)malloc(ldb * colsB * sizeof(double));
        h_B = (double*)malloc(ldb * colsB * sizeof(double));

        /* reset host workspace */
        memset(h_A, 0xFF, lda * colsA * sizeof(double));
        memset(h_X, 0xFF, ldb * colsB * sizeof(double));
        memset(h_B, 0xFF, ldb * colsB * sizeof(double));

        /* pointers to the first valid entry of A, B and X */
        double* _A = &h_A[(IA - 1) + (JA - 1) * lda];
        double* _X = &h_X[(IB - 1) + (JB - 1) * ldb];
        double* _B = &h_B[(IB - 1) + (JB - 1) * ldb];

        /* Set B[IB:IB+N, JB] = 1 */
        for (int64_t i = 0; i < N; i++)
        {
            _B[i] = 1.0;
        }

        /* Set X[IB:IB+N, JB] = 1 */
        for (int64_t i = 0; i < N; i++)
        {
            _X[i] = 1.0;
        }

        /* Set A[IA:IA+N, JA:JA+N] = permuted laplacian */
        gen_1d_laplacian_perm(N, _A, lda);

        /* print input matrices */
        if (opts.verbose)
        {
            print_host_matrix(lda, colsA, h_A, lda, "Input matrix A");
            print_host_matrix(ldb, colsB, h_X, ldb, "Input matrix X");
            print_host_matrix(ldb, colsB, h_B, ldb, "Input matrix B");
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
    const int64_t LLDB       = cusolverMpNUMROC(ldb, MB, myRowRank, RSRCB, nprow);
    const int64_t localColsB = cusolverMpNUMROC(colsB, NB, myColRank, CSRCB, npcol);

    /* Allocate global d_A */
    cudaStat = cudaMalloc((void**)&d_A, localColsA * LLDA * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Allocate global d_B */
    cudaStat = cudaMalloc((void**)&d_B, localColsB * LLDB * sizeof(double));
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
    cusolverStat = cusolverMpCreateMatrixDesc(
            &descrA, gridA, CUDA_R_64F, (IA - 1) + N, (JA - 1) + N, MA, NA, RSRCA, CSRCA, LLDA);

    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpCreateMatrixDesc(
            &descrB, gridB, CUDA_R_64F, (IB - 1) + N, (JB - 1) + 1, MB, NB, RSRCB, CSRCB, LLDB);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Allocate global d_ipiv */
    if (enable_pivoting)
    {
        /* REMARK : ipiv overlaps A[IA, JA:JA+N] as in Netlib's ScaLAPACK */
        cudaStat = cudaMalloc((void**)&d_ipiv, localColsA * sizeof(int64_t));
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }


    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_info_getrf, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&d_info_getrs, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset(d_info_getrf, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMemset(d_info_getrs, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*     QUERY WORKSPACE SIZE FOR MP ROUTINES    */
    /* =========================================== */

    cusolverStat = cusolverMpGetrf_bufferSize(handle,
                                              N,
                                              N,
                                              d_A,
                                              IA,
                                              JA,
                                              descrA,
                                              d_ipiv,
                                              CUDA_R_64F,
                                              &workspaceInBytesOnDevice_getrf,
                                              &workspaceInBytesOnHost_getrf);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpGetrs_bufferSize(handle,
                                              CUBLAS_OP_N, /* only non-transposed is supported */
                                              N,
                                              NRHS,
                                              (const void*)d_A,
                                              IA,
                                              JA,
                                              descrA,
                                              (const int64_t*)d_ipiv,
                                              d_B,
                                              IB,
                                              JB,
                                              descrB,
                                              CUDA_R_64F,
                                              &workspaceInBytesOnDevice_getrs,
                                              &workspaceInBytesOnHost_getrs);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         ALLOCATE PGETRF WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_work_getrf, workspaceInBytesOnDevice_getrf);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_work_getrf = (void*)malloc(workspaceInBytesOnHost_getrf);
    SAMPLE_ASSERT(h_work_getrf != NULL);


    /* =========================================== */
    /*         ALLOCATE PGETRS WORKSPACE            */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_work_getrs, workspaceInBytesOnDevice_getrs);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_work_getrs = (void*)malloc(workspaceInBytesOnHost_getrs);
    SAMPLE_ASSERT(h_work_getrs != NULL);

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

    cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                              ldb,
                                              colsB,
                                              (void*)d_B, /* routine requires void** */
                                              1,
                                              1,
                                              descrB,
                                              0, /* root rank */
                                              (void*)h_B,
                                              ldb);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to device */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* =========================================== */
    /*                   CALL PGETRF               */
    /* =========================================== */

    cusolverStat = cusolverMpGetrf(handle,
                                   N,
                                   N,
                                   d_A,
                                   IA,
                                   JA,
                                   descrA,
                                   d_ipiv,
                                   CUDA_R_64F,
                                   d_work_getrf,
                                   workspaceInBytesOnDevice_getrf,
                                   h_work_getrf,
                                   workspaceInBytesOnHost_getrf,
                                   d_info_getrf);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMpGetrf */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* copy d_info_getrf to host */
    cudaStat = cudaMemcpyAsync(&h_info_getrf, d_info_getrf, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* wait for d_info_getrf copy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMpGetrf */
    SAMPLE_ASSERT(h_info_getrf == 0);


    /* =========================================== */
    /*                   CALL PGETRS               */
    /* =========================================== */

    cusolverStat = cusolverMpGetrs(handle,
                                   CUBLAS_OP_N, /* only non-transposed is supported */
                                   N,
                                   NRHS,
                                   (const void*)d_A,
                                   IA,
                                   JA,
                                   descrA,
                                   (const int64_t*)d_ipiv,
                                   d_B,
                                   IB,
                                   JB,
                                   descrB,
                                   CUDA_R_64F,
                                   d_work_getrs,
                                   workspaceInBytesOnDevice_getrs,
                                   h_work_getrs,
                                   workspaceInBytesOnHost_getrs,
                                   d_info_getrs);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMpGetrs */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* copy d_info_getrs to host */
    cudaStat = cudaMemcpyAsync(&h_info_getrs, d_info_getrs, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* wait for d_info_getrs copy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMpGetrs */
    SAMPLE_ASSERT(h_info_getrs == 0);

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
                                             descrB,
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
        double* _A = &h_A[(IA - 1) + (JA - 1) * lda];
        double* _X = &h_X[(IB - 1) + (JB - 1) * ldb];
        double* _B = &h_B[(IB - 1) + (JB - 1) * ldb];

        /* measure residual error |b - A*x| */
        double max_err = 0;
        for (int row = 0; row < N; row++)
        {
            double sum = 0.0;
            for (int col = 0; col < N; col++)
            {
                double Aij = _A[row + col * lda];
                double xj  = _X[col];
                sum += Aij * xj;
            }
            double bi  = _B[row];
            double err = fabs(bi - sum);

            max_err = (max_err > err) ? max_err : err;
        }

        double x_nrm_inf = vec_nrm_inf(N, _X);
        double b_nrm_inf = vec_nrm_inf(N, _B);
        double A_nrm_inf = 4.0;
        double rel_err   = max_err / (A_nrm_inf * x_nrm_inf + b_nrm_inf);
        double tol       = 1.0e-10;
        int    ok        = (rel_err < tol);

        printf("\n|b - A*x|_inf = %E\n", max_err);
        printf("|x|_inf = %E\n", x_nrm_inf);
        printf("|b|_inf = %E\n", b_nrm_inf);
        printf("|A|_inf = %E\n", A_nrm_inf);

        /* relative error is around machine zero  */
        /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
        printf("|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);
        printf("  GETRF/GETRS check: %s  (threshold: %E)\n", ok ? "PASS" : "FAIL", tol);
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

    cusolverStat = cusolverMpDestroyMatrixDesc(descrA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyMatrixDesc(descrB);
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

    if (d_A != NULL)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_A = NULL;
    }

    if (d_B != NULL)
    {
        cudaStat = cudaFree(d_B);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_B = NULL;
    }

    if (d_ipiv != NULL)
    {
        cudaStat = cudaFree(d_ipiv);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_ipiv = NULL;
    }

    if (d_work_getrf != NULL)
    {
        cudaStat = cudaFree(d_work_getrf);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_work_getrf = NULL;
    }

    if (d_work_getrs != NULL)
    {
        cudaStat = cudaFree(d_work_getrs);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_work_getrs = NULL;
    }

    if (d_info_getrf != NULL)
    {
        cudaStat = cudaFree(d_info_getrf);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_info_getrf = NULL;
    }

    if (d_info_getrs != NULL)
    {
        cudaStat = cudaFree(d_info_getrs);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_info_getrs = NULL;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */
    /* =========================================== */
    if (h_work_getrf)
    {
        free(h_work_getrf);
        h_work_getrf = NULL;
    }
    if (h_work_getrs)
    {
        free(h_work_getrs);
        h_work_getrs = NULL;
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
