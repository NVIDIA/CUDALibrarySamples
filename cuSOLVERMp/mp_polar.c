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


/*
 * cusolverMpPolar sample: distributed QDWH polar decomposition
 *
 * Computes A = Up * H where:
 *   - Up is M-by-N with orthonormal columns (overwrites A)
 *   - H is N-by-N Hermitian positive semi-definite (optional)
 *
 * The QDWH (QR-based Dynamically Weighted Halley) algorithm is used,
 * with automatic QR/Cholesky iteration selection based on the estimated
 * condition number.
 *
 * Usage:
 *   mpirun -np 4 mp_polar -m 64 -n 64 -mbA 32 -nbA 32 -p 2 -q 2
 *   mpirun -np 2 mp_polar -m 128 -n 64 -mbA 32 -nbA 32 -p 2 -q 1
 *
 * Parameters:
 *   -m, -n     : matrix dimensions (m >= n)
 *   -mbA, -nbA : tile sizes for 2D block-cyclic distribution
 *   -ia, -ja   : 1-based submatrix offsets (default 1)
 *   -p, -q     : process grid dimensions (p rows, q columns)
 *   -grid_layout : 'C' (column-major) or 'R' (row-major)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cusolverMp.h>

#include "helpers.h"

/* Generate a random M-by-N matrix with well-conditioned singular values.
 * Adds a diagonal boost to ensure the matrix is not near-singular,
 * which helps the QDWH algorithm converge in few iterations. */
static void generate_random_matrix(int64_t m, int64_t n, double* A, int64_t lda)
{
    srand(42);
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < m; i++)
        {
            A[i + j * lda] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    /* Diagonal boost for well-conditioning */
    int64_t minmn = (m < n) ? m : n;
    for (int64_t i = 0; i < minmn; i++)
    {
        A[i + i * lda] += (double)n;
    }
}

int main(int argc, char* argv[])
{
    Options opts = { .m           = 64,
                     .n           = 64,
                     .nrhs        = 1,
                     .mbA         = 32,
                     .nbA         = 32,
                     .mbB         = 32,
                     .nbB         = 32,
                     .mbQ         = 32,
                     .nbQ         = 32,
                     .mbZ         = 32,
                     .nbZ         = 32,
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

    /* =========================================== */
    /*           POLAR-SPECIFIC ALIASES            */
    /* =========================================== */
    const int64_t m     = opts.m;   /* rows of A */
    const int64_t n     = opts.n;   /* cols of A (m >= n required) */
    const int64_t mbA   = opts.mbA; /* row tile size */
    const int64_t nbA   = opts.nbA; /* col tile size */
    const int64_t ia    = opts.ia;  /* 1-based row offset for sub(A) */
    const int64_t ja    = opts.ja;  /* 1-based col offset for sub(A) */
    const int     nprow = opts.p;   /* process grid rows */
    const int     npcol = opts.q;   /* process grid cols */

    /* H matrix uses the same tile sizes and starts at (ia, ja) */
    const int64_t ih = ia;
    const int64_t jh = ja;

    /* =========================================== */
    /*                INITIALIZE MPI               */
    /* =========================================== */
    MPI_Init(NULL, NULL);

    int rank, commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Sample-configuration checks run after MPI_Init so a failure routes
     * through MPI_Abort (via SAMPLE_ASSERT) rather than a bare abort(). */
    SAMPLE_ASSERT(m >= n && "m must be >= n for polar decomposition");
    SAMPLE_ASSERT(mbA == nbA && "tile sizes must be equal (mbA == nbA)");
    SAMPLE_ASSERT((ia - 1) % mbA == 0 && "ia must start at a row tile boundary");
    SAMPLE_ASSERT((ja - 1) % nbA == 0 && "ja must start at a column tile boundary");
    SAMPLE_ASSERT(commSize == nprow * npcol);

    if (rank == 0)
    {
        printf("cusolverMpPolar sample: A (%ld x %ld) = Up * H\n", (long)m, (long)n);
        printf("  tile size: %ld x %ld, grid: %d x %d, offsets: ia=%ld ja=%ld\n",
               (long)mbA,
               (long)nbA,
               nprow,
               npcol,
               (long)ia,
               (long)ja);
        printf("  ranks: %d\n\n", commSize);
    }

    /* =========================================== */
    /*              SET UP CUDA DEVICE             */
    /* =========================================== */
    int         localRank = getLocalRank();
    cudaError_t cudaStat  = cudaSetDevice(localRank);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0); /* force device init */
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*            CREATE NCCL COMMUNICATOR         */
    /* =========================================== */
    ncclComm_t   ncclComm = createNcclComm(commSize, rank);
    ncclResult_t ncclStat = ncclSuccess;

    /* =========================================== */
    /*          CREATE STREAM AND HANDLE           */
    /* =========================================== */
    cudaStream_t stream = NULL;
    cudaStat            = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cusolverMpHandle_t handle       = NULL;
    cusolverStatus_t   cusolverStat = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*              CREATE PROCESS GRID            */
    /* =========================================== */
    cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'R') ? CUSOLVERMP_GRID_MAPPING_ROW_MAJOR : CUSOLVERMP_GRID_MAPPING_COL_MAJOR;

    cusolverMpGrid_t grid = NULL;
    cusolverStat          = cusolverMpCreateDeviceGrid(handle, &grid, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*           COMPUTE LOCAL DIMENSIONS          */
    /* =========================================== */
    /* Global matrix A has dimensions (ia-1+m) x (ja-1+n) to account
     * for the 1-based submatrix offset. Each rank owns a subset of
     * tiles determined by NUMROC (number of rows/cols of this rank). */
    const int rsrc = 0, csrc = 0; /* source process for first tile */

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

    /* A is M-by-N with offset (ia, ja) */
    const int64_t globalRowsA         = (ia - 1) + m;
    const int64_t globalColsA         = (ja - 1) + n;
    const int64_t localRowsA          = cusolverMpNUMROC(globalRowsA, mbA, myRowRank, rsrc, nprow);
    const int64_t localColsA          = cusolverMpNUMROC(globalColsA, nbA, myColRank, csrc, npcol);
    const int64_t lldA                = (localRowsA > 0) ? localRowsA : 1;

    /* H is N-by-N with offset (ih, jh) */
    const int64_t globalRowsH         = (ih - 1) + n;
    const int64_t globalColsH         = (jh - 1) + n;
    const int64_t localRowsH          = cusolverMpNUMROC(globalRowsH, mbA, myRowRank, rsrc, nprow);
    const int64_t localColsH          = cusolverMpNUMROC(globalColsH, nbA, myColRank, csrc, npcol);
    const int64_t lldH                = (localRowsH > 0) ? localRowsH : 1;

    /* =========================================== */
    /*            ALLOCATE DEVICE MEMORY           */
    /* =========================================== */
    double* d_A    = NULL;
    double* d_H    = NULL;
    int*    d_info = NULL;

    /* Distributed matrix pointers are kept non-NULL on every rank.
     * Empty-owner ranks use a one-element dummy allocation. */
    const size_t elemsA = (size_t)(lldA * localColsA);
    const size_t elemsH = (size_t)(lldH * localColsH);
    const size_t bytesA = ((elemsA > 0) ? elemsA : 1) * sizeof(double);
    const size_t bytesH = ((elemsH > 0) ? elemsH : 1) * sizeof(double);

    cudaStat = cudaMalloc((void**)&d_A, bytesA);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void**)&d_H, bytesH);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void**)&d_info, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*          CREATE MATRIX DESCRIPTORS          */
    /* =========================================== */
    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverMpMatrixDescriptor_t descH = NULL;

    cusolverStat =
            cusolverMpCreateMatrixDesc(&descA, grid, CUDA_R_64F, globalRowsA, globalColsA, mbA, nbA, rsrc, csrc, lldA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat =
            cusolverMpCreateMatrixDesc(&descH, grid, CUDA_R_64F, globalRowsH, globalColsH, mbA, nbA, rsrc, csrc, lldH);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*           GENERATE HOST MATRIX              */
    /* =========================================== */
    double* h_A    = NULL; /* host copy of A (rank 0 only) */
    double* h_Aref = NULL; /* reference copy for verification */

    if (rank == 0)
    {
        h_A    = (double*)malloc(globalRowsA * globalColsA * sizeof(double));
        h_Aref = (double*)malloc(globalRowsA * globalColsA * sizeof(double));
        SAMPLE_ASSERT(h_A != NULL && h_Aref != NULL);

        /* Zero the full buffer, then fill the submatrix at (ia, ja) */
        memset(h_A, 0, globalRowsA * globalColsA * sizeof(double));
        generate_random_matrix(m, n, &h_A[(ia - 1) + (ja - 1) * globalRowsA], globalRowsA);

        /* Save reference copy for later verification */
        memcpy(h_Aref, h_A, globalRowsA * globalColsA * sizeof(double));
    }

    /* =========================================== */
    /*        SCATTER A FROM MASTER TO RANKS       */
    /* =========================================== */
    cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                              globalRowsA,
                                              globalColsA,
                                              (void*)d_A,
                                              1,
                                              1,
                                              descA,
                                              0, /* root rank */
                                              (void*)h_A,
                                              globalRowsA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*          CONFIGURE POLAR PARAMETERS         */
    /* =========================================== */
    /* uplo: CUBLAS_FILL_MODE_FULL — input A is a general dense matrix.
     *       Use CUBLAS_FILL_MODE_UPPER if A is already upper triangular
     *       (e.g., from a prior QR factorization) to skip internal QR.
     *
     * polarDesc: optional descriptor for perturbation control and diagnostics. */
    cublasFillMode_t            uplo        = CUBLAS_FILL_MODE_FULL;  /* general input matrix */
    cudaDataType_t              computeType = CUDA_R_64F;
    cusolverMpPolarDescriptor_t polarDesc   = NULL;

    cusolverStat = cusolverMpPolarDescriptorCreate(&polarDesc);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Requested perturbation. Values <= 0 request no perturbation.
     * Float input is accepted and stored by the descriptor as double. */
    float requested_ksi = 0.0f;
    cusolverStat        = cusolverMpPolarDescriptorSetAttribute(
            polarDesc, CUSOLVERMP_POLAR_DESCRIPTOR_ATTRIBUTE_REQUESTED_KSI, &requested_ksi, sizeof(requested_ksi));
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*            QUERY WORKSPACE SIZE             */
    /* =========================================== */
    size_t d_workSize = 0, h_workSize = 0;

    cusolverStat = cusolverMpPolar_bufferSize(handle,
                                              polarDesc,
                                              uplo,
                                              m,
                                              n,
                                              (const void*)d_A,
                                              ia,
                                              ja,
                                              descA,
                                              (const void*)d_H,
                                              ih,
                                              jh,
                                              descH,
                                              computeType,
                                              &d_workSize,
                                              &h_workSize);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    if (rank == 0)
    {
        printf("Workspace: device=%zu bytes, host=%zu bytes\n", d_workSize, h_workSize);
    }

    /* =========================================== */
    /*             ALLOCATE WORKSPACE              */
    /* =========================================== */
    void* d_work = NULL;
    void* h_work = NULL;

    if (d_workSize > 0)
    {
        cudaStat = cudaMalloc(&d_work, d_workSize);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }
    if (h_workSize > 0)
    {
        h_work = malloc(h_workSize);
        SAMPLE_ASSERT(h_work != NULL);
    }

    /* =========================================== */
    /*        EXECUTE POLAR DECOMPOSITION          */
    /* =========================================== */
    /* On output:
     *   d_A contains Up (M-by-N, orthonormal columns)
     *   d_H contains H (N-by-N Hermitian PSD, symmetrized)
     *   polarDesc contains scalar diagnostics */
    cusolverStat = cusolverMpPolar(handle,
                                   polarDesc,
                                   uplo,
                                   m,
                                   n,
                                   (void*)d_A,
                                   ia,
                                   ja,
                                   descA,
                                   (void*)d_H,
                                   ih,
                                   jh,
                                   descH,
                                   computeType,
                                   d_work,
                                   d_workSize,
                                   h_work,
                                   h_workSize,
                                   d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*                CHECK D_INFO                 */
    /* =========================================== */
    int h_info = 0;
    cudaStat   = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    double requested_ksi_attr = 0.0;
    double a_nrmF             = 0.0;
    double rcond              = 0.0;
    size_t written            = 0;
    cusolverStat              = cusolverMpPolarDescriptorGetAttribute(polarDesc,
                                                         CUSOLVERMP_POLAR_DESCRIPTOR_ATTRIBUTE_REQUESTED_KSI,
                                                         &requested_ksi_attr,
                                                         sizeof(requested_ksi_attr),
                                                         &written);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    SAMPLE_ASSERT(written == sizeof(requested_ksi_attr));
    cusolverStat = cusolverMpPolarDescriptorGetAttribute(
            polarDesc, CUSOLVERMP_POLAR_DESCRIPTOR_ATTRIBUTE_A_NORM_FROBENIUS,
            &a_nrmF, sizeof(a_nrmF), &written);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    SAMPLE_ASSERT(written == sizeof(a_nrmF));
    cusolverStat = cusolverMpPolarDescriptorGetAttribute(
            polarDesc, CUSOLVERMP_POLAR_DESCRIPTOR_ATTRIBUTE_RCOND_ESTIMATE,
            &rcond, sizeof(rcond), &written);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    SAMPLE_ASSERT(written == sizeof(rcond));
    double applied_ksi = (requested_ksi_attr > 0.0) ? requested_ksi_attr : 0.0;

    if (rank == 0)
    {
        printf("\nResults:\n");
        printf("  d_info = %d (%s)\n",
               h_info,
               h_info == 0 ? "success" : (h_info < 0 ? "invalid argument" : "convergence failure"));
        printf("  ||A||_F = %.6e\n", a_nrmF);
        printf("  rcond   = %.6e\n", rcond);
        printf("  ksi     = %.6e%s\n", applied_ksi, applied_ksi > 0 ? " (perturbation applied)" : "");
    }

    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    SAMPLE_ASSERT(h_info == 0);

    /* =========================================== */
    /*              GATHER RESULTS                 */
    /* =========================================== */
    double* h_Up = NULL;
    double* h_H  = NULL;

    if (rank == 0)
    {
        h_Up = (double*)malloc(globalRowsA * globalColsA * sizeof(double));
        h_H  = (double*)malloc(globalRowsH * globalColsH * sizeof(double));
        SAMPLE_ASSERT(h_Up != NULL && h_H != NULL);
    }

    /* Gather Up (M-by-N) */
    cusolverStat = cusolverMpMatrixGatherD2H(
            handle, globalRowsA, globalColsA, (void*)d_A, 1, 1, descA, 0, (void*)h_Up, globalRowsA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Gather H (N-by-N) */
    cusolverStat = cusolverMpMatrixGatherD2H(
            handle, globalRowsH, globalColsH, (void*)d_H, 1, 1, descH, 0, (void*)h_H, globalRowsH);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*             VERIFY ON MASTER                */
    /* =========================================== */
    if (rank == 0)
    {
        double* Up = &h_Up[(ia - 1) + (ja - 1) * globalRowsA];

        /* Check orthogonality: ||Up^T * Up - I||_F.
         * This sample is real double-only; a complex variant must use Up^H. */
        double ortho_err = 0.0;
        for (int64_t j = 0; j < n; j++)
        {
            for (int64_t i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int64_t k = 0; k < m; k++)
                {
                    sum += Up[k + i * globalRowsA] * Up[k + j * globalRowsA];
                }
                double target = (i == j) ? 1.0 : 0.0;
                double diff   = sum - target;
                ortho_err += diff * diff;
            }
        }
        ortho_err = sqrt(ortho_err);

        printf("\nVerification:\n");
        printf("  ||Up^T*Up - I||_F = %.6e\n", ortho_err);
        sample_ok = sample_ok && (ortho_err < n * 1e-12);
        printf("  Polar check: %s\n", sample_ok ? "PASS" : "FAIL");
    }

    /* =========================================== */
    /*                  CLEANUP                    */
    /* =========================================== */
    if (rank == 0)
    {
        if (h_A) free(h_A);
        if (h_Aref) free(h_Aref);
        if (h_Up) free(h_Up);
        if (h_H) free(h_H);
    }

    if (d_work != NULL)
    {
        cudaStat = cudaFree(d_work);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_work = NULL;
    }
    if (h_work != NULL)
    {
        free(h_work);
        h_work = NULL;
    }

    if (d_A != NULL)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_A = NULL;
    }
    if (d_H != NULL)
    {
        cudaStat = cudaFree(d_H);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_H = NULL;
    }
    if (d_info != NULL)
    {
        cudaStat = cudaFree(d_info);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_info = NULL;
    }

    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpDestroyMatrixDesc(descH);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyGrid(grid);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpPolarDescriptorDestroy(polarDesc);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroy(handle);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    ncclStat = ncclCommDestroy(ncclComm);
    SAMPLE_ASSERT(ncclStat == ncclSuccess);
    cudaStat = cudaStreamDestroy(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* MPI barrier before MPI_Finalize */
    MPI_Barrier(MPI_COMM_WORLD);
    sample_ok = sample_all_ranks_succeeded(sample_ok);
    MPI_Finalize();

    if (rank == 0)
    {
        printf("%s\n", sample_ok ? "[SUCCEEDED]" : "[FAILED]");
    }

    return sample_ok ? 0 : 1;
}
