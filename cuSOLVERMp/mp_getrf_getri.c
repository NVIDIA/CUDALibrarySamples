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
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* compute |A|_inf: the largest absolute row sum */
static double mat_nrm_inf(int64_t m, int64_t n, int64_t lda, const double* A)
{
    double max_nrm = 0;

    for (int64_t row = 0; row < m; row++)
    {
        double row_sum = 0;
        for (int64_t col = 0; col < n; col++)
        {
            row_sum += fabs(A[row + col * lda]);
        }

        max_nrm = fmax(max_nrm, row_sum);
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

    /* tridiagonal band along the anti-diagonal */
    for (int64_t J = 0; J < N; J++)
    {
        /* anti-diagonal */
        A[((N - 1) - J) + J * lda] = 2.0;

        /* one below the anti-diagonal */
        if (J > 0)
        {
            A[((N - 1) - (J - 1)) + J * lda] = -1.0;
        }
        /* one above the anti-diagonal */
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

    /* Define dimensions, block sizes and offsets of the A matrix */
    const int64_t N = opts.n;

    /* Enable / disable pivoting (the inverse is computed either way) */
    const int enable_pivoting = 1;

    /* Offsets of A matrix (base-1). cusolverMpGetri requires tile-aligned
     * offsets, i.e. (IA - 1) must be a multiple of MA and (JA - 1) a
     * multiple of NA. */
    const int64_t IA = opts.ia;
    const int64_t JA = opts.ja;

    /* Tile sizes; cusolverMpGetri requires square tiles (MA == NA) */
    const int64_t MA = opts.mbA;
    const int64_t NA = opts.nbA;

    /* Define grid of processors */
    const int nprow = opts.p;
    const int npcol = opts.q;

    /* Convert grid layout to cuSOLVERMp grid mapping */
    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t RSRCA = 0;
    const uint32_t CSRCA = 0;

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

    /* Initialize cuSOLVERMp library handle */
    cusolverStat = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* cuSOLVERMp grids */
    cusolverMpGrid_t gridA = NULL;

    /* cuSOLVERMp matrix descriptors */
    cusolverMpMatrixDescriptor_t descrA = NULL;

    /* Distributed matrices */
    void*    d_A    = NULL;
    int64_t* d_ipiv = NULL;

    /* Distributed device workspace */
    void* d_work_getrf = NULL;
    void* d_work_getri = NULL;

    /* Distributed host workspace */
    void* h_work_getrf = NULL;
    void* h_work_getri = NULL;

    /* size of workspace on device */
    size_t workspaceInBytesOnDevice_getrf = 0;
    size_t workspaceInBytesOnDevice_getri = 0;

    /* size of workspace on host */
    size_t workspaceInBytesOnHost_getrf = 0;
    size_t workspaceInBytesOnHost_getri = 0;

    /* error codes from cuSOLVERMp (device) */
    int* d_info_getrf = NULL;
    int* d_info_getri = NULL;

    /* error codes from cuSOLVERMp (host) */
    int h_info_getrf = 0;
    int h_info_getri = 0;

    /* Single process per device */
    SAMPLE_ASSERT((nprow * npcol) == commSize);

    /* cusolverMpGetri layout requirements: square tiles and tile-aligned
     * submatrix offsets */
    SAMPLE_ASSERT(MA == NA);
    SAMPLE_ASSERT(((IA - 1) % MA) == 0);
    SAMPLE_ASSERT(((JA - 1) % NA) == 0);

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    const int64_t lda   = (IA - 1) + N;
    const int64_t colsA = (JA - 1) + N;

    double* h_A    = NULL;
    double* h_Ainv = NULL;

    if (rank == 0)
    {
        /* allocate host workspace */
        h_A    = (double*)malloc(lda * colsA * sizeof(double));
        h_Ainv = (double*)malloc(lda * colsA * sizeof(double));
        SAMPLE_ASSERT(h_A != NULL);
        SAMPLE_ASSERT(h_Ainv != NULL);

        /* reset host workspace */
        memset(h_A, 0xFF, lda * colsA * sizeof(double));
        memset(h_Ainv, 0xFF, lda * colsA * sizeof(double));

        /* pointer to the first valid entry of A */
        double* _A = &h_A[(IA - 1) + (JA - 1) * lda];

        /* Set A[IA:IA+N, JA:JA+N] = permuted laplacian (its rows are
         * reversed, so the LU factorization performs real row interchanges
         * that cusolverMpGetri has to undo on the columns of the inverse) */
        gen_1d_laplacian_perm(N, _A, lda);

        /* print input matrix */
        if (opts.verbose)
        {
            print_host_matrix(lda, colsA, h_A, lda, "Input matrix A");
        }
    }

    /* =========================================== */
    /*                COMPUTE LLDA                 */
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
            &descrA, gridA, CUDA_R_64F, (IA - 1) + N, (JA - 1) + N, MA, NA, RSRCA, CSRCA, LLDA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Allocate global d_ipiv */
    if (enable_pivoting)
    {
        /* REMARK : ipiv overlaps A[IA, JA:JA+N] as in Netlib's ScaLAPACK.
         * Allocate at least one entry: d_ipiv must not be NULL on a process
         * without local columns, because d_ipiv must be NULL on all
         * processes or on none. */
        cudaStat = cudaMalloc((void**)&d_ipiv, (localColsA > 0 ? localColsA : 1) * sizeof(int64_t));
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }


    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_info_getrf, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&d_info_getri, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset(d_info_getrf, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMemset(d_info_getri, 0, sizeof(int));
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

    cusolverStat = cusolverMpGetri_bufferSize(handle,
                                              N,
                                              (const void*)d_A,
                                              IA,
                                              JA,
                                              descrA,
                                              (const int64_t*)d_ipiv,
                                              CUDA_R_64F,
                                              &workspaceInBytesOnDevice_getri,
                                              &workspaceInBytesOnHost_getri);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         ALLOCATE PGETRF WORKSPACE           */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_work_getrf, workspaceInBytesOnDevice_getrf);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_work_getrf = (void*)malloc(workspaceInBytesOnHost_getrf);
    SAMPLE_ASSERT(h_work_getrf != NULL);


    /* =========================================== */
    /*         ALLOCATE PGETRI WORKSPACE           */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_work_getri, workspaceInBytesOnDevice_getri);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_work_getri = (void*)malloc(workspaceInBytesOnHost_getri);
    SAMPLE_ASSERT(h_work_getri != NULL);

    /* =========================================== */
    /*          SCATTER MATRIX A FROM MASTER       */
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
    /*                   CALL PGETRI               */
    /* =========================================== */

    cusolverStat = cusolverMpGetri(handle,
                                   N,
                                   d_A,
                                   IA,
                                   JA,
                                   descrA,
                                   (const int64_t*)d_ipiv,
                                   CUDA_R_64F,
                                   d_work_getri,
                                   workspaceInBytesOnDevice_getri,
                                   h_work_getri,
                                   workspaceInBytesOnHost_getri,
                                   d_info_getri);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMpGetri */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* copy d_info_getri to host */
    cudaStat = cudaMemcpyAsync(&h_info_getri, d_info_getri, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* wait for d_info_getri copy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMpGetri: info > 0 would report the
     * global row index of an exactly-zero diagonal element of U (singular
     * factor), with sub(A) left unmodified */
    SAMPLE_ASSERT(h_info_getri == 0);

    /* =========================================== */
    /*        GATHER MATRIX A FROM MASTER          */
    /* =========================================== */

    /* Copy inverse to h_Ainv; sub(A) holds the full inverse */
    cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                             lda,
                                             colsA,
                                             (void*)d_A,
                                             1,
                                             1,
                                             descrA,
                                             0, /* master rank, destination */
                                             (void*)h_Ainv,
                                             lda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync wait for data to arrive to host */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);


    /* =========================================== */
    /*            CHECK RESIDUAL ON MASTER         */
    /* =========================================== */
    if (rank == 0)
    {
        /* print output matrix */
        if (opts.verbose)
        {
            print_host_matrix(lda, colsA, h_Ainv, lda, "Output matrix inv(A)");
        }

        /* pointers to the first valid entry of A and inv(A) */
        double* _A    = &h_A[(IA - 1) + (JA - 1) * lda];
        double* _Ainv = &h_Ainv[(IA - 1) + (JA - 1) * lda];

        /* measure residual error |A * inv(A) - I|_inf: the largest absolute
         * row sum of the residual matrix */
        double max_err = 0;
        for (int64_t row = 0; row < N; row++)
        {
            double row_sum = 0.0;
            for (int64_t col = 0; col < N; col++)
            {
                double sum = 0.0;
                for (int64_t k = 0; k < N; k++)
                {
                    sum += _A[row + k * lda] * _Ainv[k + col * lda];
                }

                double expected = (row == col ? 1.0 : 0.0);
                row_sum += fabs(sum - expected);
            }

            /* fmax ignores a NaN. Keep the NaN, so that a NaN or Inf inverse fails the check. */
            if (isnan(row_sum) || row_sum > max_err)
            {
                max_err = row_sum;
            }
        }

        double A_nrm_inf    = mat_nrm_inf(N, N, lda, _A);
        double Ainv_nrm_inf = mat_nrm_inf(N, N, lda, _Ainv);
        /* An Inf denominator gives rel_err = 0 when max_err is finite, so ok also needs a finite denominator. */
        double denom   = A_nrm_inf * Ainv_nrm_inf;
        double rel_err = max_err / denom;
        double tol     = 1.0e-10;
        int    ok      = isfinite(denom) && (rel_err < tol);

        printf("\n|A * inv(A) - I|_inf = %E\n", max_err);
        printf("|A|_inf = %E\n", A_nrm_inf);
        printf("|inv(A)|_inf = %E\n", Ainv_nrm_inf);

        /* relative error is around machine zero */
        printf("|A * inv(A) - I|/(|A|*|inv(A)|) = %E\n\n", rel_err);
        printf("  GETRF/GETRI check: %s  (threshold: %E)\n", ok ? "PASS" : "FAIL", tol);
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
        if (h_Ainv)
        {
            free(h_Ainv);
            h_Ainv = NULL;
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

    if (d_work_getri != NULL)
    {
        cudaStat = cudaFree(d_work_getri);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_work_getri = NULL;
    }

    if (d_info_getrf != NULL)
    {
        cudaStat = cudaFree(d_info_getrf);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_info_getrf = NULL;
    }

    if (d_info_getri != NULL)
    {
        cudaStat = cudaFree(d_info_getri);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_info_getri = NULL;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */
    /* =========================================== */
    if (h_work_getrf)
    {
        free(h_work_getrf);
        h_work_getrf = NULL;
    }
    if (h_work_getri)
    {
        free(h_work_getri);
        h_work_getri = NULL;
    }

    /* =========================================== */
    /*                      CLEANUP                */
    /* =========================================== */

    /* Destroy cuSOLVERMp handle */
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
