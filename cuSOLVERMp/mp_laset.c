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
 * cuSOLVERMp LASET sample
 *
 * Demonstrates cusolverMpLaset which initializes a distributed M-by-N matrix:
 *   - Off-diagonal elements in the specified triangular region are set to alpha
 *   - Diagonal elements are set to beta
 *
 * The sample:
 *   1. Creates a distributed matrix filled with a sentinel value
 *   2. Calls cusolverMpLaset with CUBLAS_FILL_MODE_FULL to set all elements
 *   3. Gathers the result to rank 0 and verifies correctness
 *
 * Usage: mpirun -n 2 ./mp_laset
 *        mpirun -n 4 ./mp_laset -p 2 -q 2 -m 20 -n 15 -mbA 4 -nbA 4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

int main(int argc, char* argv[])
{
    Options opts = { .m           = 10,
                     .n           = 10,
                     .nrhs        = 1,
                     .mbA         = 3,
                     .nbA         = 3,
                     .mbB         = 1,
                     .nbB         = 1,
                     .mbQ         = 1,
                     .nbQ         = 1,
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

    int sample_ok = 1;

    /* Initialize MPI library */
    MPI_Init(NULL, NULL);

    /* Matrix dimensions and offsets */
    const int64_t m  = opts.m;
    const int64_t n  = opts.n;
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;

    /* Process grid */
    const int nprow = opts.p;
    const int npcol = opts.q;

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

    /* Single process per device */
    SAMPLE_ASSERT((nprow * npcol) == commSize);

    /* Create NCCL communicator */
    ncclComm_t ncclComm = createNcclComm(commSize, rank);

    /* Create local stream */
    cudaStream_t stream = NULL;
    cudaStat            = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Initialize cusolverMp library handle */
    cusolverMpHandle_t handle = NULL;
    cusolverStat              = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*          CREATE GRID AND DESCRIPTORS        */
    /* =========================================== */

    /* Global matrix dimensions: large enough to hold submatrix at (IA, JA) */
    const int64_t lda   = (ia - 1) + m;
    const int64_t colsA = (ja - 1) + n;

    cusolverMpGrid_t gridA = NULL;
    cusolverStat           = cusolverMpCreateDeviceGrid(handle, &gridA, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverMpMatrixDescriptor_t descA = NULL;

    /* Compute process grid index */
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

    /* Compute local leading dimension */
    const int64_t llda       = cusolverMpNUMROC(lda, mbA, myRowRank, rsrca, nprow);
    const int64_t localColsA = cusolverMpNUMROC(colsA, nbA, myColRank, csrca, npcol);

    cusolverStat = cusolverMpCreateMatrixDesc(&descA, gridA, CUDA_R_64F, lda, colsA, mbA, nbA, rsrca, csrca, llda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         ALLOCATE DISTRIBUTED MATRIX         */
    /* =========================================== */

    void* d_A = NULL;
    cudaStat  = cudaMalloc(&d_A, llda * localColsA * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Allocate device info */
    int* d_info = NULL;
    cudaStat    = cudaMalloc((void**)&d_info, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*      INITIALIZE MATRIX WITH SENTINEL        */
    /* =========================================== */

    /* Fill entire distributed matrix with a sentinel value (-999.0)
     * so we can verify that LASET only touches the intended region. */
    const double sentinel = -999.0;

    /* Scatter a host matrix filled with sentinel to the distributed matrix */
    double* h_sentinel_matrix = NULL;
    if (rank == 0)
    {
        h_sentinel_matrix = (double*)malloc(lda * colsA * sizeof(double));
        SAMPLE_ASSERT(h_sentinel_matrix != NULL);
        for (int64_t i = 0; i < lda * colsA; i++)
        {
            h_sentinel_matrix[i] = sentinel;
        }
    }

    cusolverStat = cusolverMpMatrixScatterH2D(handle, lda, colsA, d_A, 1, 1, descA, 0, h_sentinel_matrix, lda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0)
    {
        free(h_sentinel_matrix);
        h_sentinel_matrix = NULL;
    }

    /* =========================================== */
    /*            CALL cusolverMpLaset             */
    /* =========================================== */

    /* Set off-diagonal to alpha=3.14, diagonal to beta=2.71.
     * Alpha and beta pointers can reside on the host or the device;
     * here we pass host pointers for simplicity. */
    const double alpha = 3.14;
    const double beta  = 2.71;

    cusolverStat = cusolverMpLaset(handle,
                                   CUBLAS_FILL_MODE_FULL,
                                   m,
                                   n,
                                   &alpha, /* host or device pointer */
                                   &beta,  /* host or device pointer */
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Sync after LASET */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Check info */
    int h_info = 0;
    cudaStat   = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    /* =========================================== */
    /*      GATHER RESULT AND VERIFY ON RANK 0     */
    /* =========================================== */

    double* h_A = NULL;
    if (rank == 0)
    {
        h_A = (double*)malloc(lda * colsA * sizeof(double));
        SAMPLE_ASSERT(h_A != NULL);
    }

    cusolverStat = cusolverMpMatrixGatherD2H(handle, lda, colsA, d_A, 1, 1, descA, 0, h_A, lda);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0)
    {
        if (opts.verbose)
        {
            printf("Result matrix (global, column-major):\n");
            for (int64_t i = 0; i < lda; i++)
            {
                for (int64_t j = 0; j < colsA; j++)
                {
                    printf("%8.2f ", h_A[i + j * lda]);
                }
                printf("\n");
            }
            printf("\n");
        }

        /* Verify: check each element of the M-by-N submatrix at (IA, JA).
         * With FULL mode: diagonal == beta, off-diagonal == alpha.
         * Elements outside the submatrix should remain sentinel. */
        int errors = 0;

        for (int64_t j = 0; j < colsA; j++)
        {
            for (int64_t i = 0; i < lda; i++)
            {
                double val = h_A[i + j * lda];

                /* Convert to 0-based submatrix coordinates */
                int64_t si = i - (ia - 1);
                int64_t sj = j - (ja - 1);

                /* Check if (i,j) is inside the M-by-N submatrix */
                int in_submatrix = (si >= 0 && si < m && sj >= 0 && sj < n);

                double expected;
                if (!in_submatrix)
                {
                    expected = sentinel;
                }
                else if (si == sj)
                {
                    expected = beta; /* diagonal */
                }
                else
                {
                    expected = alpha; /* off-diagonal */
                }

                if (val != expected)
                {
                    if (errors < 10)
                    {
                        printf("MISMATCH at (%ld, %ld): got %.4f, expected %.4f\n",
                               (long)(i + 1),
                               (long)(j + 1),
                               val,
                               expected);
                    }
                    errors++;
                }
            }
        }

        if (errors > 0)
        {
            printf("VERIFICATION FAILED: %d mismatches\n", errors);
            sample_ok = 0;
        }
        else
        {
            printf("Verification passed: all %ld elements correct (%ld x %ld submatrix + %ld sentinel)\n",
                   (long)(lda * colsA),
                   (long)m,
                   (long)n,
                   (long)(lda * colsA - m * n));
        }
    }

    /* =========================================== */
    /*                  CLEAN UP                   */
    /* =========================================== */

    if (rank == 0)
    {
        free(h_A);
    }

    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyGrid(gridA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    if (d_A)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }

    if (d_info)
    {
        cudaStat = cudaFree(d_info);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }

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

    /* Finalize MPI environment */
    MPI_Finalize();

    if (rank == 0)
    {
        printf("%s\n", sample_ok ? "[SUCCEEDED]" : "[FAILED]");
    }

    return sample_ok ? 0 : 1;
}
