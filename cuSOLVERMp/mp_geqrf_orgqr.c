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
 * cuSOLVERMp QR factorization + explicit Q formation sample
 *
 * Demonstrates the full distributed QR workflow:
 *   1. cusolverMpGeqrf - computes the QR factorization A = Q * R
 *   2. cusolverMpOrgqr - forms the explicit orthogonal matrix Q
 *
 * Verification (on rank 0 after gathering):
 *   - Factorization:  ||A - Q*R|| / (||A|| * sqrt(max(m,n)))
 *   - Orthogonality:  ||Q^T*Q - I|| / (n * eps)
 *
 * Given an m x n matrix A (m >= n), the thin QR factorization produces:
 *   Q  (m x n)  with orthonormal columns
 *   R  (n x n)  upper triangular
 *
 * Usage:
 *   mpirun -n 2 ./mp_geqrf_orgqr
 *   mpirun -n 4 ./mp_geqrf_orgqr -p 2 -q 2 -m 100 -n 50 -mbA 32 -nbA 32
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* Generate a deterministic random m x n matrix */
static void generate_random_matrix(int64_t m, int64_t n, double* A, int64_t lda)
{
    srand(42);
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < m; i++)
        {
            A[i + j * lda] = (double)rand() / RAND_MAX;
        }
    }
}

/* Print matrix (only for small sizes to avoid flooding the terminal) */
static void print_host_matrix(int64_t m, int64_t n, const double* A, int64_t lda, const char* msg)
{
    if (m * n > 2000) return;
    printf("print_host_matrix : %s\n", msg);
    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
        {
            printf("%10.4f  ", A[i + j * lda]);
        }
        printf("\n");
    }
}

/* Compute Frobenius norm of an m x n matrix */
static double frobenius_norm(int64_t m, int64_t n, const double* A, int64_t lda)
{
    double norm = 0.0;
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < m; i++)
        {
            double val = A[i + j * lda];
            norm += val * val;
        }
    }
    return sqrt(norm);
}

int main(int argc, char* argv[])
{
    Options opts = { .m           = 20,
                     .n           = 10,
                     .nrhs        = 1,
                     .mbA         = 4,
                     .nbA         = 4,
                     .mbB         = 4,
                     .nbB         = 4,
                     .mbQ         = 4,
                     .nbQ         = 4,
                     .mbZ         = 4,
                     .nbZ         = 4,
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

    /* Initialize MPI */
    MPI_Init(NULL, NULL);

    int rank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    if (rank == 0) print(&opts);

    /* Problem dimensions - ORGQR requires m >= n */
    const int64_t m = opts.m;
    const int64_t n = opts.n;
    const int64_t k = (m < n) ? m : n; /* number of Householder reflectors */

    if (m < n)
    {
        if (rank == 0) fprintf(stderr, "Error: this sample requires m >= n (tall or square matrix)\n");
        MPI_Finalize();
        return 1;
    }

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;

    /* Base-1 offsets into the distributed matrix */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;

    /* Process grid */
    const int nprow = opts.p;
    const int npcol = opts.q;

    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t RSRCA = 0;
    const uint32_t CSRCA = 0;

    assert((nprow * npcol) <= commSize);

    /* =========================================== */
    /*             CUDA / NCCL SETUP               */
    /* =========================================== */

    const int localDeviceId = getLocalRank();

    cudaError_t cudaStat = cudaSetDevice(localDeviceId);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    assert(cudaStat == cudaSuccess);

    /* Create NCCL communicator */
    ncclUniqueId ncclId;
    if (rank == 0)
    {
        ncclGetUniqueId(&ncclId);
    }
    MPI_Bcast((void*)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t ncclComm;
    ncclResult_t ncclStat = ncclCommInitRank(&ncclComm, commSize, ncclId, rank);
    assert(ncclStat == ncclSuccess);

    /* Create CUDA stream */
    cudaStream_t stream = NULL;
    cudaStat = cudaStreamCreate(&stream);
    assert(cudaStat == cudaSuccess);

    /* Initialize cusolverMp library handle */
    cusolverMpHandle_t handle = NULL;
    cusolverStatus_t cusolverStat = cusolverMpCreate(&handle, localDeviceId, stream);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*       GRID AND MATRIX DESCRIPTORS           */
    /* =========================================== */

    cusolverMpGrid_t grid = NULL;
    cusolverStat = cusolverMpCreateDeviceGrid(handle, &grid, ncclComm, nprow, npcol, gridLayout);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Global matrix dimensions (including base-1 offset padding) */
    const int64_t m_global = (ia - 1) + m;
    const int64_t n_global = (ja - 1) + n;

    /* Compute process grid coordinates for this rank */
    int myprow, mypcol;
    if (gridLayout == CUSOLVERMP_GRID_MAPPING_COL_MAJOR)
    {
        myprow = rank % nprow;
        mypcol = rank / nprow;
    }
    else
    {
        myprow = rank / npcol;
        mypcol = rank % npcol;
    }

    /* Local matrix dimensions (2D block-cyclic distribution) */
    const int64_t m_local = cusolverMpNUMROC(m_global, mbA, myprow, RSRCA, nprow);
    const int64_t n_local = cusolverMpNUMROC(n_global, nbA, mypcol, CSRCA, npcol);

    /* Create matrix descriptor for A */
    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverStat = cusolverMpCreateMatrixDesc(
            &descA, grid, CUDA_R_64F, m_global, n_global, mbA, nbA, RSRCA, CSRCA, m_local);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*        ALLOCATE DISTRIBUTED BUFFERS         */
    /* =========================================== */

    /* Local portion of the distributed matrix A */
    void* d_A = NULL;
    cudaStat = cudaMalloc((void**)&d_A, m_local * n_local * sizeof(double));
    assert(cudaStat == cudaSuccess);

    /* Tau buffer - one element per local column of the descriptor.
     * The library requires a non-null d_tau pointer on ALL ranks when k > 0
     * (even if a rank owns zero local columns), so allocate at least 1 element. */
    void* d_tau = NULL;
    cudaStat = cudaMalloc((void**)&d_tau, ((n_local > 0) ? n_local : 1) * sizeof(double));
    assert(cudaStat == cudaSuccess);

    /* Device-side info flag */
    int* d_info = NULL;
    cudaStat = cudaMalloc((void**)&d_info, sizeof(int));
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    assert(cudaStat == cudaSuccess);

    /* =========================================== */
    /*       GENERATE INPUT ON RANK 0              */
    /* =========================================== */

    double* h_A_orig = NULL; /* original A, kept for verification */
    double* h_A      = NULL; /* scratch used by scatter             */
    double* h_QR     = NULL; /* will hold factored A (to extract R) */
    double* h_Q      = NULL; /* will hold explicit Q                */

    if (rank == 0)
    {
        h_A_orig = (double*)malloc(m_global * n_global * sizeof(double));
        h_A      = (double*)malloc(m_global * n_global * sizeof(double));
        h_QR     = (double*)malloc(m_global * n_global * sizeof(double));
        h_Q      = (double*)malloc(m_global * n_global * sizeof(double));

        memset(h_A, 0, m_global * n_global * sizeof(double));

        /* Fill the submatrix A[ia:ia+m-1, ja:ja+n-1] with random values */
        double* ptr_A = &h_A[(ia - 1) + (ja - 1) * m_global];
        generate_random_matrix(m, n, ptr_A, m_global);

        /* Save original for verification */
        memcpy(h_A_orig, h_A, m_global * n_global * sizeof(double));

        if (opts.verbose)
        {
            print_host_matrix(m, n, ptr_A, m_global, "Input matrix A");
        }
    }

    /* =========================================== */
    /*       SCATTER A TO DISTRIBUTED LAYOUT       */
    /* =========================================== */

    cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                              m_global,
                                              n_global,
                                              (void*)d_A,
                                              1,
                                              1,
                                              descA,
                                              0, /* root rank */
                                              (void*)h_A,
                                              m_global);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    assert(cudaStat == cudaSuccess);

    /* =========================================== */
    /*      QUERY WORKSPACE FOR GEQRF AND ORGQR   */
    /* =========================================== */

    size_t geqrf_d_bytes = 0, geqrf_h_bytes = 0;
    size_t orgqr_d_bytes = 0, orgqr_h_bytes = 0;

    cusolverStat = cusolverMpGeqrf_bufferSize(handle,
                                              m,
                                              n,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              CUDA_R_64F,
                                              &geqrf_d_bytes,
                                              &geqrf_h_bytes);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpOrgqr_bufferSize(handle,
                                              m,
                                              n,
                                              k,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              d_tau,
                                              CUDA_R_64F,
                                              &orgqr_d_bytes,
                                              &orgqr_h_bytes);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Allocate workspace large enough for both routines.
     * We pass each routine its own queried size (not the max) so the internal
     * workspace-adequacy check stays precise. */
    const size_t d_work_bytes = (geqrf_d_bytes > orgqr_d_bytes) ? geqrf_d_bytes : orgqr_d_bytes;
    const size_t h_work_bytes = (geqrf_h_bytes > orgqr_h_bytes) ? geqrf_h_bytes : orgqr_h_bytes;

    void* d_work = NULL;
    if (d_work_bytes > 0)
    {
        cudaStat = cudaMalloc((void**)&d_work, d_work_bytes);
        assert(cudaStat == cudaSuccess);
    }

    void* h_work = NULL;
    if (h_work_bytes > 0)
    {
        h_work = malloc(h_work_bytes);
        assert(h_work != NULL);
    }

    /* =========================================== */
    /*  STEP 1: GEQRF - DISTRIBUTED QR FACTORIZE   */
    /* =========================================== */

    if (rank == 0) printf("\nStep 1: Distributed QR factorization (cusolverMpGeqrf)...\n");

    cusolverStat = cusolverMpGeqrf(handle,
                                   m,
                                   n,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_tau,
                                   CUDA_R_64F,
                                   d_work,
                                   geqrf_d_bytes,
                                   h_work,
                                   geqrf_h_bytes,
                                   d_info);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    assert(cudaStat == cudaSuccess);

    /* Verify GEQRF completed successfully */
    int h_info = 0;
    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    assert(cudaStat == cudaSuccess);
    assert(h_info == 0);

    /* =========================================== */
    /*  GATHER FACTORED A (needed to extract R)    */
    /* =========================================== */

    /* ORGQR will overwrite d_A with Q, so gather the factored form now
     * to extract the upper-triangular R on rank 0 for verification. */
    cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                             m_global,
                                             n_global,
                                             (void*)d_A,
                                             1,
                                             1,
                                             descA,
                                             0, /* destination: rank 0 */
                                             (void*)h_QR,
                                             m_global);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    assert(cudaStat == cudaSuccess);

    /* =========================================== */
    /*  STEP 2: ORGQR - FORM EXPLICIT Q            */
    /* =========================================== */

    if (rank == 0) printf("Step 2: Forming orthogonal matrix Q (cusolverMpOrgqr)...\n");

    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    assert(cudaStat == cudaSuccess);

    cusolverStat = cusolverMpOrgqr(handle,
                                   m,
                                   n,
                                   k,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_tau,
                                   CUDA_R_64F,
                                   d_work,
                                   orgqr_d_bytes,
                                   h_work,
                                   orgqr_h_bytes,
                                   d_info);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    assert(cudaStat == cudaSuccess);

    /* Verify ORGQR completed successfully */
    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    assert(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    assert(cudaStat == cudaSuccess);
    assert(h_info == 0);

    /* =========================================== */
    /*          GATHER Q TO RANK 0                 */
    /* =========================================== */

    cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                             m_global,
                                             n_global,
                                             (void*)d_A,
                                             1,
                                             1,
                                             descA,
                                             0, /* destination: rank 0 */
                                             (void*)h_Q,
                                             m_global);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    assert(cudaStat == cudaSuccess);

    /* =========================================== */
    /*      VERIFY FACTORIZATION ON RANK 0         */
    /* =========================================== */

    if (rank == 0)
    {
        printf("\nVerification:\n");

        /* Pointers to the m x n submatrices at offset (ia, ja) */
        const double* A_orig = &h_A_orig[(ia - 1) + (ja - 1) * m_global];
        const double* QR_fac = &h_QR[(ia - 1) + (ja - 1) * m_global];
        const double* Q      = &h_Q[(ia - 1) + (ja - 1) * m_global];

        if (opts.verbose)
        {
            print_host_matrix(m, n, QR_fac, m_global, "Factored matrix (R in upper triangle, V below)");
            print_host_matrix(m, n, Q, m_global, "Orthogonal matrix Q");
        }

        /* Extract R: the upper-triangular n x n factor from the GEQRF output */
        double* h_R = (double*)calloc(n * n, sizeof(double));
        for (int64_t j = 0; j < n; j++)
        {
            for (int64_t i = 0; i <= j; i++)
            {
                h_R[i + j * n] = QR_fac[i + j * m_global];
            }
        }

        if (opts.verbose)
        {
            print_host_matrix(n, n, h_R, n, "Upper triangular R");
        }

        /* ---- Check 1: Factorization ||A - Q*R|| / (||A|| * sqrt(max(m,n))) ---- */

        /* Compute residual: A_orig - Q * R
         * Q is m x n, R is n x n upper triangular, product is m x n */
        double* h_resid = (double*)malloc(m * n * sizeof(double));
        for (int64_t j = 0; j < n; j++)
        {
            for (int64_t i = 0; i < m; i++)
            {
                double qr_ij = 0.0;
                /* R is upper triangular: R[l,j] = 0 for l > j */
                for (int64_t l = 0; l <= j; l++)
                {
                    qr_ij += Q[i + l * m_global] * h_R[l + j * n];
                }
                h_resid[i + j * m] = A_orig[i + j * m_global] - qr_ij;
            }
        }

        const double norm_A     = frobenius_norm(m, n, A_orig, m_global);
        const double norm_resid = frobenius_norm(m, n, h_resid, m);
        const double max_mn     = (double)((m > n) ? m : n);
        const double fact_err   = norm_resid / (norm_A * sqrt(max_mn));

        printf("  Factorization:  ||A - Q*R|| / (||A|| * sqrt(max(m,n))) = %E\n", fact_err);

        /* ---- Check 2: Orthogonality ||Q^T*Q - I|| / (n * eps) ---- */

        /* Compute Q^T * Q (n x n) - exploit symmetry, compute upper triangle */
        double* h_QtQ = (double*)calloc(n * n, sizeof(double));
        for (int64_t j = 0; j < n; j++)
        {
            for (int64_t i = 0; i <= j; i++)
            {
                double sum = 0.0;
                for (int64_t l = 0; l < m; l++)
                {
                    sum += Q[l + i * m_global] * Q[l + j * m_global];
                }
                h_QtQ[i + j * n] = sum;
                h_QtQ[j + i * n] = sum;
            }
        }

        /* Subtract identity */
        for (int64_t i = 0; i < n; i++)
        {
            h_QtQ[i + i * n] -= 1.0;
        }

        const double norm_ortho = frobenius_norm(n, n, h_QtQ, n);
        const double eps        = 2.220446049250313e-16; /* double precision machine epsilon */
        const double orth_err   = norm_ortho / ((double)n * eps);

        printf("  Orthogonality:  ||Q^T*Q - I|| / (n * eps)             = %E\n", orth_err);

        /* Evaluate pass/fail */
        const double fact_tol = 1.0e-12;
        const double orth_tol = 100.0;
        const int    fact_ok  = (fact_err < fact_tol);
        const int    orth_ok  = (orth_err < orth_tol);

        printf("\n  Factorization check: %s  (threshold: %E)\n", fact_ok ? "PASS" : "FAIL", fact_tol);
        printf("  Orthogonality check: %s  (threshold: %.0f)\n", orth_ok ? "PASS" : "FAIL", orth_tol);

        free(h_R);
        free(h_resid);
        free(h_QtQ);
    }

    /* =========================================== */
    /*        CLEAN UP HOST WORKSPACE              */
    /* =========================================== */

    if (rank == 0)
    {
        if (h_A_orig)
        {
            free(h_A_orig);
            h_A_orig = NULL;
        }
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
        if (h_Q)
        {
            free(h_Q);
            h_Q = NULL;
        }
    }

    /* =========================================== */
    /*           DESTROY MATRIX DESCRIPTORS        */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             DESTROY MATRIX GRIDS            */
    /* =========================================== */

    cusolverStat = cusolverMpDestroyGrid(grid);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*          DEALLOCATE DEVICE WORKSPACE        */
    /* =========================================== */

    if (d_A != NULL)
    {
        cudaStat = cudaFree(d_A);
        assert(cudaStat == cudaSuccess);
        d_A = NULL;
    }

    if (d_tau != NULL)
    {
        cudaStat = cudaFree(d_tau);
        assert(cudaStat == cudaSuccess);
        d_tau = NULL;
    }

    if (d_work != NULL)
    {
        cudaStat = cudaFree(d_work);
        assert(cudaStat == cudaSuccess);
        d_work = NULL;
    }

    if (d_info != NULL)
    {
        cudaStat = cudaFree(d_info);
        assert(cudaStat == cudaSuccess);
        d_info = NULL;
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
    /*                  CLEANUP                    */
    /* =========================================== */

    cusolverStat = cusolverMpDestroy(handle);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    assert(cudaStat == cudaSuccess);

    ncclStat = ncclCommDestroy(ncclComm);
    assert(ncclStat == ncclSuccess);

    cudaStat = cudaStreamDestroy(stream);
    assert(cudaStat == cudaSuccess);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    if (rank == 0)
    {
        printf("\n[SUCCEEDED]\n");
    }

    return 0;
}
