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

#include <cusolverMp.h>

#include "helpers.h"

/* compute |x|_inf */
static double normI(int64_t m, int64_t n, int64_t lda, const double* A)
{
    double max_nrm = 0;

    for (int64_t col = 0; col < n; col++)
    {
        double err = 0;
        for (int64_t row = 0; row < m; row++)
        {
            err += fabs(A[row + col * lda]);
        }

        max_nrm = fmax(max_nrm, err);
    }

    return max_nrm;
}

static void generate_diagonal_dominant_symmetric_matrix(int64_t n, double* A, int64_t lda)
{
    /* fill A[0:n, 0:n] with a random symmetric diagonally dominant matrix */
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

    /* Define dimensions, block sizes and offsets of the A matrix */
    const int64_t n = opts.n;

    /* Offsets of A matrix (base-1). cusolverMpPotri requires tile-aligned
     * offsets, i.e. (ia - 1) must be a multiple of mbA and (ja - 1) a
     * multiple of nbA. */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;

    /* Tile sizes; cusolverMpPotri requires square tiles (mbA == nbA) */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;

    /* Define grid of processors */
    const int nprow = opts.p;
    const int npcol = opts.q;

    /* Convert grid layout to cuSOLVERMp grid mapping */
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
    cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
    ncclResult_t     ncclStat     = ncclSuccess;

    /* Create communicator */
    ncclComm_t ncclComm = createNcclComm(commSize, rank);

    /* Create local stream */
    cudaStream_t stream = NULL;
    cudaStat            = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Initialize cuSOLVERMp library handle */
    cusolverMpHandle_t handle = NULL;
    cusolverStat              = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* cuSOLVERMp grids */
    cusolverMpGrid_t gridA = NULL;

    /* cuSOLVERMp matrix descriptors */
    cusolverMpMatrixDescriptor_t descA = NULL;

    /* Distributed matrices */
    void* d_A = NULL;

    /* Distributed device workspace */
    void* d_potrfWork = NULL;
    void* d_potriWork = NULL;

    /* Distributed host workspace */
    void* h_potrfWork = NULL;
    void* h_potriWork = NULL;

    /* size of workspace on device */
    size_t potrfWorkspaceInBytesOnDevice = 0;
    size_t potriWorkspaceInBytesOnDevice = 0;

    /* size of workspace on host */
    size_t potrfWorkspaceInBytesOnHost = 0;
    size_t potriWorkspaceInBytesOnHost = 0;

    /* error codes from cuSOLVERMp (device) */
    int* d_potrfInfo = NULL;
    int* d_potriInfo = NULL;

    /* error codes from cuSOLVERMp (host) */
    int h_potrfInfo = 0;
    int h_potriInfo = 0;

    /* Single process per device */
    SAMPLE_ASSERT((nprow * npcol) == commSize);

    /* =========================================== */
    /*          Create inputs on master rank       */
    /* =========================================== */

    const int64_t lda   = (ia - 1) + n;
    const int64_t colsA = (ja - 1) + n;

    double* h_A    = NULL;
    double* h_Ainv = NULL;

    if (rank == 0)
    {
        /* allocate host workspace */
        h_A    = (double*)malloc(lda * colsA * sizeof(double));
        h_Ainv = (double*)malloc(lda * colsA * sizeof(double));

        /* reset host workspace */
        memset(h_A, 0xFF, lda * colsA * sizeof(double));
        memset(h_Ainv, 0xFF, lda * colsA * sizeof(double));

        /* pointer to the first valid entry of A */
        double* _A = &h_A[(ia - 1) + (ja - 1) * lda];

        /* Set A[ia:ia+n, ja:ja+n] = diagonal dominant symmetric matrix */
        generate_diagonal_dominant_symmetric_matrix(n, _A, lda);

        /* print input matrix */
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
    cudaStat = cudaMalloc((void**)&d_A, llda * localColsA * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*          CREATE GRID DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateDeviceGrid(handle, &gridA, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*        CREATE MATRIX DESCRIPTORS            */
    /* =========================================== */
    cusolverStat = cusolverMpCreateMatrixDesc(&descA, gridA, CUDA_R_64F, lda, colsA, mbA, nbA, rsrca, csrca, llda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_potrfInfo, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMalloc((void**)&d_potriInfo, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*                RESET D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMemset(d_potrfInfo, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMemset(d_potriInfo, 0, sizeof(int));
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
                                              CUDA_R_64F,
                                              &potrfWorkspaceInBytesOnDevice,
                                              &potrfWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpPotri_bufferSize(handle,
                                              CUBLAS_FILL_MODE_LOWER,
                                              n,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              CUDA_R_64F,
                                              &potriWorkspaceInBytesOnDevice,
                                              &potriWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         ALLOCATE Ppotrf WORKSPACE           */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_potrfWork, potrfWorkspaceInBytesOnDevice);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_potrfWork = (void*)malloc(potrfWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(h_potrfWork != NULL);

    /* =========================================== */
    /*         ALLOCATE Ppotri WORKSPACE           */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_potriWork, potriWorkspaceInBytesOnDevice);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_potriWork = (void*)malloc(potriWorkspaceInBytesOnHost);
    SAMPLE_ASSERT(h_potriWork != NULL);

    /* =========================================== */
    /*          SCATTER MATRIX A FROM MASTER       */
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
    /*                   CALL Ppotrf               */
    /* =========================================== */

    cusolverStat = cusolverMpPotrf(handle,
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   CUDA_R_64F,
                                   d_potrfWork,
                                   potrfWorkspaceInBytesOnDevice,
                                   h_potrfWork,
                                   potrfWorkspaceInBytesOnHost,
                                   d_potrfInfo);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMpPotrf */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* copy d_potrfInfo to host */
    cudaStat = cudaMemcpyAsync(&h_potrfInfo, d_potrfInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* wait for d_potrfInfo copy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMpPotrf */
    SAMPLE_ASSERT(h_potrfInfo == 0);

    /* =========================================== */
    /*                   CALL Ppotri               */
    /* =========================================== */

    cusolverStat = cusolverMpPotri(handle,
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   CUDA_R_64F,
                                   d_potriWork,
                                   potriWorkspaceInBytesOnDevice,
                                   h_potriWork,
                                   potriWorkspaceInBytesOnHost,
                                   d_potriInfo);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* sync after cusolverMpPotri */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* copy d_potriInfo to host */
    cudaStat = cudaMemcpyAsync(&h_potriInfo, d_potriInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* wait for d_potriInfo copy */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* check return value of cusolverMpPotri: info > 0 would report the
     * global row index of an exactly-zero diagonal element of the factor */
    SAMPLE_ASSERT(h_potriInfo == 0);

    /* =========================================== */
    /*        GATHER MATRIX A FROM MASTER          */
    /* =========================================== */

    /* Copy inverse to h_Ainv; the lower triangle of sub(A) holds the
     * inverse, the strict upper triangle is not referenced or modified */
    cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                             lda,
                                             colsA,
                                             (void*)d_A,
                                             1,
                                             1,
                                             descA,
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
            print_host_matrix(lda, colsA, h_Ainv, lda, "Output matrix inv(A) (lower triangle)");
        }

        /* pointers to the first valid entry of A and inv(A) */
        double* _A    = &h_A[(ia - 1) + (ja - 1) * lda];
        double* _Ainv = &h_Ainv[(ia - 1) + (ja - 1) * lda];

        /* symmetrize the inverse from its lower triangle */
        for (int64_t j = 0; j < n; j++)
        {
            for (int64_t i = 0; i < j; i++)
            {
                _Ainv[i + j * lda] = _Ainv[j + i * lda];
            }
        }

        /* measure residual error |A * inv(A) - I|_inf: the largest absolute
         * row sum of the residual matrix */
        double max_err = 0;
        for (int64_t row = 0; row < n; row++)
        {
            double row_sum = 0.0;
            for (int64_t col = 0; col < n; col++)
            {
                double sum = 0.0;
                for (int64_t k = 0; k < n; k++)
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

        /* normI accumulates column sums (the 1-norm); for the symmetric A and
         * inv(A) used here it equals the infinity norm. */
        double A_nrm_inf    = normI(n, n, lda, _A);
        double Ainv_nrm_inf = normI(n, n, lda, _Ainv);
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
        printf("  POTRF/POTRI check: %s  (threshold: %E)\n", ok ? "PASS" : "FAIL", tol);
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

    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid(gridA);
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

    if (d_potrfWork)
    {
        cudaStat = cudaFree(d_potrfWork);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_potrfWork = NULL;
    }

    if (d_potriWork)
    {
        cudaStat = cudaFree(d_potriWork);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_potriWork = NULL;
    }

    if (d_potrfInfo)
    {
        cudaStat = cudaFree(d_potrfInfo);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_potrfInfo = NULL;
    }

    if (d_potriInfo)
    {
        cudaStat = cudaFree(d_potriInfo);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_potriInfo = NULL;
    }

    /* =========================================== */
    /*         DEALLOCATE HOST WORKSPACE           */
    /* =========================================== */
    if (h_potrfWork)
    {
        free(h_potrfWork);
        h_potrfWork = NULL;
    }
    if (h_potriWork)
    {
        free(h_potriWork);
        h_potriWork = NULL;
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
