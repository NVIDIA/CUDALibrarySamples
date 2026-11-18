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
 * cuSOLVERMp QR factorization + least-squares solve sample
 *
 * Demonstrates the full distributed QR workflow:
 *   1. cusolverMpGeqrf - computes the QR factorization A = Q * R
 *   2. cusolverMpGeqrs - solves the least-squares system A * X = B using the
 *                        QR factors, reused across several right-hand sides
 *
 * The factorization is computed once; cusolverMpGeqrs then takes sub(A) and
 * d_tau as const inputs, so the same factors solve every right-hand-side instance
 * with no refactorization in between.
 *
 * Verification (on rank 0 after gathering):
 *   - Solution:  ||B - A*X|| / (max(m,n) * ||A|| * ||X||)
 *
 * Given an m x n matrix A (m >= n) and an m x nrhs right-hand side B, the
 * least-squares solve produces X (n x nrhs) that minimizes ||A*X - B||.
 *
 * Usage:
 *   mpirun -n 2 ./mp_geqrf_geqrs
 *   mpirun -n 4 ./mp_geqrf_geqrs -p 2 -q 2 -m 128 -n 64 -nrhs 64 -ia 1 -ib 1 -mbA 32 -nbA 32 -mbB 32 -nbB 32
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

/* Number of right-hand-side instances solved against the single factorization */
#define NUM_SOLVES 2

/* Print matrix */
static void print_host_matrix(const int64_t m, const int64_t n, const double* A, const int64_t lda, const char* msg)
{
    printf("print_host_matrix : %s\n", msg);

    for (int64_t i = 0; i < m; i++)
    {
        for (int64_t j = 0; j < n; j++)
            printf("%.6e  ", A[i + j * lda]);
        printf("\n");
    }
}

/* Generate a random m x n matrix */
static void generate_random_matrix(const int64_t m, const int64_t n, double* A, const int64_t lda)
{
    for (int64_t j = 0; j < n; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            A[i + j * lda] = (double)(rand()) / RAND_MAX;
        }
    }
}

/* Compute B = A*X  (A is m x n, X is n x nrhs, B is m x nrhs) */
static void compute_rhs(const int64_t m,
                        const int64_t n,
                        const int64_t nrhs,
                        const double* A,
                        const int64_t lda,
                        const double* X,
                        const int64_t ldx,
                        double*       B,
                        const int64_t ldb)
{
    for (int64_t j = 0; j < nrhs; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            double tmp = 0;
            for (int64_t k = 0; k < n; ++k)
            {
                tmp += A[i + k * lda] * X[k + j * ldx];
            }
            B[i + j * ldb] = tmp;
        }
    }
}

/* Compute Frobenius norm of an m x n matrix */
static double compute_norm2(const int64_t m, const int64_t n, const double* A, const int64_t lda)
{
    double norm_A = 0;
    for (int64_t j = 0; j < n; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            const double val = A[i + j * lda];
            norm_A += val * val;
        }
    }
    return sqrt(norm_A);
}

/* Compute the relative residual for A*X = B  (A is m x n, X is n x nrhs, B is m x nrhs) */
static double compute_error(const int64_t m,
                            const int64_t n,
                            const int64_t nrhs,
                            const double* A,
                            const int64_t lda,
                            const double* X,
                            const int64_t ldx,
                            double*       B,
                            const int64_t ldb)
{
    /* Relative residual: ||A*X - B|| / (max(m,n) * ||A|| * ||X||) */
    const double norm_A = compute_norm2(m, n, A, lda);
    const double norm_X = compute_norm2(n, nrhs, X, ldx);

    for (int64_t j = 0; j < nrhs; ++j)
    {
        for (int64_t i = 0; i < m; ++i)
        {
            double tmp = 0;
            for (int64_t k = 0; k < n; ++k)
            {
                tmp += A[i + k * lda] * X[k + j * ldx];
            }
            B[i + j * ldb] -= tmp;
        }
    }
    const double norm_R = compute_norm2(m, nrhs, B, ldb);

    /* A norm of X that overflows to Inf can make the error 0: return NaN, so that the check fails. */
    if (!isfinite(norm_X))
    {
        return NAN;
    }
    return (norm_R / fmax(m, n) / norm_A / norm_X);
}

/* Decode a cusolverStatus_t into a short human-readable string so a failing
 * call can print an actionable message before aborting. */
static const char* geqrs_status_string(cusolverStatus_t status)
{
    switch (status)
    {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
        default: return "CUSOLVER_STATUS_<unknown>";
    }
}

int main(int argc, char* argv[])
{
    /* =========================================== */
    /*          PARSE / VALIDATE OPTIONS           */
    /* =========================================== */

    Options opts = { .m           = 20,
                     .n           = 8,
                     .nrhs        = 8,
                     .mbA         = 2,
                     .nbA         = 2,
                     .mbB         = 2,
                     .nbB         = 2,
                     .ia          = 3,
                     .ja          = 1,
                     .ib          = 3,
                     .jb          = 1,
                     .p           = 1,
                     .q           = 2,
                     .grid_layout = 'C',
                     .verbose     = false };

    parse(&opts, argc, argv);
    validate(&opts);

    int sample_ok = 1;

    /* =========================================== */
    /*               INITIALIZE MPI                */
    /* =========================================== */

    MPI_Init(NULL, NULL);

    /* Use MPI_COMM_WORLD for process coordination */
    MPI_Comm mpi_global_comm = MPI_COMM_WORLD;

    /* Query this rank's id and the total communicator size */
    int commSize, rank;
    MPI_Comm_size(mpi_global_comm, &commSize);
    MPI_Comm_rank(mpi_global_comm, &rank);

    if (rank == 0) print(&opts);

    /* =========================================== */
    /*        PROBLEM DIMENSIONS AND LAYOUT        */
    /* =========================================== */

    /* Problem dimensions */
    const int64_t m    = opts.m;
    const int64_t n    = opts.n;
    const int64_t nrhs = opts.nrhs;

    /* This sample solves an over-determined system without transpose */
    SAMPLE_ASSERT(m >= n && "this sample requires m >= n for the least-squares solve");

    /* Because the residual check divides by norm(A) and norm(X), this sample does not accept an empty
     * problem. The library accepts N = 0 or NRHS = 0 as an empty problem. */
    SAMPLE_ASSERT(n >= 1 && nrhs >= 1 && "this sample requires n >= 1 and nrhs >= 1");

    /* Base-1 offsets into the distributed A and B matrices */
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;

    /* Each base-1 offset must be 1 or more, because the host-buffer indices use offset - 1. */
    SAMPLE_ASSERT(ia >= 1 && ja >= 1 && ib >= 1 && jb >= 1 && "this sample requires ia, ja, ib and jb >= 1");

    /* Tile sizes */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbB = opts.mbB;
    const int64_t nbB = opts.nbB;

    /* The local-size calculations divide by the tile sizes, so each tile size must be 1 or more. */
    SAMPLE_ASSERT(mbA >= 1 && nbA >= 1 && mbB >= 1 && nbB >= 1 && "this sample requires mbA, nbA, mbB and nbB >= 1");

    /* Process grid */
    const int nprow = opts.p;
    const int npcol = opts.q;

    /* Convert grid layout to cuSOLVERMp grid mapping */
    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    /* Current implementation only allows RSRC,CSRC=(0,0) */
    const uint32_t rsrca = 0;
    const uint32_t csrca = 0;
    const uint32_t rsrcb = 0;
    const uint32_t csrcb = 0;

    /* =========================================== */
    /*             CUDA / NCCL SETUP               */
    /* =========================================== */

    /* Bind this process to its local GPU */
    const int   localRank = getLocalRank();
    cudaError_t cudaStat  = cudaSetDevice(localRank);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Status/return codes */
    cusolverStatus_t cusolverStat = CUSOLVER_STATUS_SUCCESS;
    ncclResult_t     ncclStat     = ncclSuccess;

    /* Create NCCL communicator */
    ncclComm_t ncclComm = createNcclComm(commSize, rank);

    /* Create CUDA stream */
    cudaStream_t stream = NULL;
    cudaStat            = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*          CREATE CUSOLVERMP HANDLE           */
    /* =========================================== */

    /* Initialize cuSOLVERMp library handle */
    cusolverMpHandle_t handle = NULL;
    cusolverStat              = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*         DECLARE DISTRIBUTED OBJECTS         */
    /* =========================================== */

    /* The handles and buffers below are declared here and created by the
     * phases further down; grouping the declarations keeps each later phase
     * focused on a single cuSOLVERMp step. */

    /* cuSOLVERMp process grids (one per matrix) */
    cusolverMpGrid_t gridA = NULL;
    cusolverMpGrid_t gridB = NULL;

    /* cuSOLVERMp matrix descriptors. descB describes the layout only, so all
     * NUM_SOLVES right-hand-side instances share it. */
    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverMpMatrixDescriptor_t descB = NULL;

    /* Distributed matrices for A*X = B, plus the reflector scalars from the factorization */
    void* d_A   = NULL;
    void* d_tau = NULL;

    /* Single flat device buffer holding all NUM_SOLVES right-hand-side instances contiguously */
    double* d_B = NULL;

    /* Distributed device and host workspace, sized for the larger of the two routines */
    void* d_work = NULL;
    void* h_work = NULL;

    size_t geqrfWorkspaceInBytesOnDevice = 0;
    size_t geqrfWorkspaceInBytesOnHost   = 0;
    size_t geqrsWorkspaceInBytesOnDevice = 0;
    size_t geqrsWorkspaceInBytesOnHost   = 0;

    /* Device-side info flag */
    int* d_info = NULL;

    /* Host copy of the info flag */
    int h_info = 0;

    /* Require exactly one process per grid cell */
    SAMPLE_ASSERT((nprow * npcol) == commSize);

    /* =========================================== */
    /*          GENERATE INPUT ON RANK 0           */
    /* =========================================== */

    /* Global matrix dimensions (including base-1 offset padding) */
    const int64_t m_global_A = (ia - 1) + m;
    const int64_t n_global_A = (ja - 1) + n;
    const int64_t m_global_B = (ib - 1) + m;
    const int64_t n_global_B = (jb - 1) + nrhs;

    /* Element count of one right-hand-side instance in the global B layout */
    const int64_t h_ld_instance = m_global_B * n_global_B;

    double* h_A = NULL; /* input matrix A                                       */
    double* h_B = NULL; /* flat buffer: all NUM_SOLVES right-hand-side instances   */
    double* h_X = NULL; /* exact X, later reused to gather solution             */

    if (rank == 0)
    {
        /* Allocate host buffers. h_B is a single flat buffer holding all
         * NUM_SOLVES right-hand-side instances contiguously. */
        h_A = (double*)malloc(m_global_A * n_global_A * sizeof(double));
        h_B = (double*)malloc(NUM_SOLVES * h_ld_instance * sizeof(double));
        h_X = (double*)malloc(m_global_B * n_global_B * sizeof(double));
        SAMPLE_ASSERT(h_A != NULL && h_B != NULL && h_X != NULL);

        /* Fill with a poison pattern (0xFF) to expose any unscattered entries */
        memset(h_A, 0xFF, m_global_A * n_global_A * sizeof(double));
        memset(h_X, 0xFF, m_global_B * n_global_B * sizeof(double));

        /* Fill the submatrix A[ia:ia+m-1, ja:ja+n-1] with random values */
        double* ptr_A = &h_A[(ia - 1) + (ja - 1) * m_global_A];
        generate_random_matrix(m, n, ptr_A, m_global_A);

        /* One A, several unrelated right-hand sides. Each is generated from its
         * own random exact solution so a solve that reused a stale result would
         * be caught by the residual check below. */
        for (int s = 0; s < NUM_SOLVES; ++s)
        {
            double* h_B_s = h_B + s * h_ld_instance;
            memset(h_B_s, 0xFF, h_ld_instance * sizeof(double));

            double* ptr_B = &h_B_s[(ib - 1) + (jb - 1) * m_global_B];
            double* ptr_X = &h_X[(ib - 1) + (jb - 1) * m_global_B];

            generate_random_matrix(n, nrhs, ptr_X, m_global_B);
            compute_rhs(m, n, nrhs, ptr_A, m_global_A, ptr_X, m_global_B, ptr_B, m_global_B);
        }

        if (opts.verbose)
        {
            print_host_matrix(m_global_A, n_global_A, h_A, m_global_A, "Input matrix A");
            for (int s = 0; s < NUM_SOLVES; ++s)
            {
                print_host_matrix(m_global_B, n_global_B, h_B + s * h_ld_instance, m_global_B, "Input matrix B");
            }
        }
    }

    /* =========================================== */
    /*          COMPUTE LOCAL DIMENSIONS           */
    /* =========================================== */

    /* Compute process grid coordinates for this rank */
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

    /* Local matrix dimensions for A and B (2D block-cyclic distribution) */
    const int64_t m_local_A = cusolverMpNUMROC(m_global_A, mbA, myRowRank, rsrca, nprow);
    const int64_t n_local_A = cusolverMpNUMROC(n_global_A, nbA, myColRank, csrca, npcol);

    const int64_t m_local_B = cusolverMpNUMROC(m_global_B, mbB, myRowRank, rsrcb, nprow);
    const int64_t n_local_B = cusolverMpNUMROC(n_global_B, nbB, myColRank, csrcb, npcol);

    /* =========================================== */
    /*           ALLOCATE DEVICE BUFFERS           */
    /* =========================================== */

    /* Leading dimensions and allocation sizes are padded to at least one so that
     * empty-owner ranks still pass a valid lld and a non-NULL, non-zero-sized
     * pointer to the collective calls. */
    const int64_t lld_A = (m_local_A > 0) ? m_local_A : 1;
    cudaStat = cudaMalloc((void**)&d_A, ((lld_A * n_local_A) > 0 ? (lld_A * n_local_A) : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /*
     * d_tau has dimension LOCc(JA+N-1) in A's column distribution, i.e. one
     * element per local column of descA -- not LOCc(N). It must be non-null on
     * every rank, including ranks that own no local columns, so allocate at
     * least one element. This buffer is the caller's to keep: cusolverMpGeqrs
     * reads it and leaves it untouched, which is what makes the factorization
     * reusable.
     */
    cudaStat = cudaMalloc((void**)&d_tau, ((n_local_A > 0) ? n_local_A : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    const int64_t lld_B = (m_local_B > 0) ? m_local_B : 1;

    /* Single flat device buffer holding all NUM_SOLVES right-hand-side instances
     * contiguously; instance s starts at d_B + s * instance_stride.
     * instance_stride is the per-RHS-instance element stride of the flat buffer
     * (lld_B * n_local_B), NOT a leading dimension. It is derived from the same
     * lld_B passed to the descriptor so a padded lld cannot make instances overlap. */
    const int64_t instance_stride = lld_B * n_local_B;
    cudaStat = cudaMalloc((void**)&d_B, NUM_SOLVES * ((instance_stride > 0) ? instance_stride : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*            CREATE PROCESS GRIDS             */
    /* =========================================== */
    cusolverStat = cusolverMpCreateDeviceGrid(handle, &gridA, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpCreateDeviceGrid(handle, &gridB, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*          CREATE MATRIX DESCRIPTORS          */
    /* =========================================== */
    cusolverStat = cusolverMpCreateMatrixDesc(
            &descA, gridA, CUDA_R_64F, m_global_A, n_global_A, mbA, nbA, rsrca, csrca, lld_A);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpCreateMatrixDesc(
            &descB, gridB, CUDA_R_64F, m_global_B, n_global_B, mbB, nbB, rsrcb, csrcb, lld_B);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             ALLOCATE D_INFO                 */
    /* =========================================== */

    cudaStat = cudaMalloc((void**)&d_info, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*      QUERY WORKSPACE FOR GEQRF AND GEQRS    */
    /* =========================================== */

    cusolverStat = cusolverMpGeqrf_bufferSize(
            handle, m, n, d_A, ia, ja, descA, CUDA_R_64F, &geqrfWorkspaceInBytesOnDevice, &geqrfWorkspaceInBytesOnHost);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "cusolverMpGeqrf_bufferSize failed: %s (status=%d). If you changed the sample flags, check them "
                "against the cusolverMpGeqrf requirements in "
                "https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.\n",
                geqrs_status_string(cusolverStat),
                (int)cusolverStat);
    }
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpGeqrs_bufferSize(handle,
                                              CUBLAS_OP_N,
                                              m,
                                              n,
                                              nrhs,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              d_tau,
                                              d_B,
                                              ib,
                                              jb,
                                              descB,
                                              CUDA_R_64F,
                                              &geqrsWorkspaceInBytesOnDevice,
                                              &geqrsWorkspaceInBytesOnHost);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "cusolverMpGeqrs_bufferSize failed: %s (status=%d). If you changed the sample flags, check them "
                "against the cusolverMpGeqrs requirements in "
                "https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.\n",
                geqrs_status_string(cusolverStat),
                (int)cusolverStat);
    }
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* =========================================== */
    /*             ALLOCATE WORKSPACE              */
    /* =========================================== */

    /* Allocate workspace large enough for both routines. We pass each routine
     * its own queried size (not the max) so the internal workspace-adequacy
     * check stays precise. */
    const size_t workspaceInBytesOnDevice = (geqrfWorkspaceInBytesOnDevice > geqrsWorkspaceInBytesOnDevice)
                                                    ? geqrfWorkspaceInBytesOnDevice
                                                    : geqrsWorkspaceInBytesOnDevice;
    const size_t workspaceInBytesOnHost   = (geqrfWorkspaceInBytesOnHost > geqrsWorkspaceInBytesOnHost)
                                                    ? geqrfWorkspaceInBytesOnHost
                                                    : geqrsWorkspaceInBytesOnHost;

    cudaStat = cudaMalloc((void**)&d_work, workspaceInBytesOnDevice > 0 ? workspaceInBytesOnDevice : 1);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    h_work = (void*)malloc(workspaceInBytesOnHost > 0 ? workspaceInBytesOnHost : 1);
    SAMPLE_ASSERT(h_work != NULL);

    /* =========================================== */
    /*     SCATTER A AND B TO DISTRIBUTED LAYOUT   */
    /* =========================================== */
    cusolverStat = cusolverMpMatrixScatterH2D(
            handle, m_global_A, n_global_A, (void*)d_A, 1, 1, descA, 0 /* root rank */, (void*)h_A, m_global_A);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    for (int s = 0; s < NUM_SOLVES; ++s)
    {
        cusolverStat = cusolverMpMatrixScatterH2D(handle,
                                                  m_global_B,
                                                  n_global_B,
                                                  (void*)(d_B + s * instance_stride),
                                                  1,
                                                  1,
                                                  descB,
                                                  0 /* root rank */,
                                                  (void*)(h_B ? h_B + s * h_ld_instance : NULL),
                                                  m_global_B);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    }

    /* Wait for the scattered data to reach the device */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* =========================================== */
    /*  STEP 1: GEQRF - DISTRIBUTED QR FACTORIZE   */
    /* =========================================== */

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
                                   geqrfWorkspaceInBytesOnDevice,
                                   h_work,
                                   geqrfWorkspaceInBytesOnHost,
                                   d_info);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS)
    {
        fprintf(stderr,
                "cusolverMpGeqrf failed: %s (status=%d). If you changed the sample flags, check them against the "
                "cusolverMpGeqrf requirements in https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.\n",
                geqrs_status_string(cusolverStat),
                (int)cusolverStat);
    }
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Verify GEQRF completed successfully */
    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    if (rank == 0) printf("\nFactored once with cusolverMpGeqrf.\n");

    /* =========================================== */
    /*  STEP 2: GEQRS - SOLVE A*X = B REPEATEDLY   */
    /* =========================================== */

    for (int s = 0; s < NUM_SOLVES; ++s)
    {
        /* d_A and d_tau are const here: the same factorization serves every
         * pass, and no cusolverMpGeqrf call appears inside this loop. */
        cusolverStat = cusolverMpGeqrs(handle,
                                       CUBLAS_OP_N,
                                       m,
                                       n,
                                       nrhs,
                                       d_A,
                                       ia,
                                       ja,
                                       descA,
                                       d_tau,
                                       d_B + s * instance_stride,
                                       ib,
                                       jb,
                                       descB,
                                       CUDA_R_64F,
                                       d_work,
                                       geqrsWorkspaceInBytesOnDevice,
                                       h_work,
                                       geqrsWorkspaceInBytesOnHost,
                                       d_info);
        if (cusolverStat != CUSOLVER_STATUS_SUCCESS)
        {
            fprintf(stderr,
                    "cusolverMpGeqrs failed: %s (status=%d). If you changed the sample flags, check them against the "
                    "cusolverMpGeqrs requirements in https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html.\n",
                    geqrs_status_string(cusolverStat),
                    (int)cusolverStat);
        }
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cudaStat = cudaStreamSynchronize(stream);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);

        /* Verify the solve completed successfully */
        cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        cudaStat = cudaStreamSynchronize(stream);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        SAMPLE_ASSERT(h_info == 0);
    }

    if (rank == 0) printf("Solved %d right-hand-side instances with cusolverMpGeqrs.\n\n", NUM_SOLVES);

    /* =========================================== */
    /*     GATHER SOLUTIONS AND VERIFY ON RANK 0   */
    /* =========================================== */

    for (int s = 0; s < NUM_SOLVES; ++s)
    {
        cusolverStat = cusolverMpMatrixGatherD2H(handle,
                                                 m_global_B,
                                                 n_global_B,
                                                 (void*)(d_B + s * instance_stride),
                                                 1,
                                                 1,
                                                 descB,
                                                 0 /* master rank, destination */,
                                                 (void*)h_X,
                                                 m_global_B);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

        cudaStat = cudaStreamSynchronize(stream);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);

        if (rank == 0)
        {
            if (opts.verbose)
            {
                print_host_matrix(m_global_B, n_global_B, h_X, m_global_B, "Output matrix X");
            }

            double* h_B_s = h_B + s * h_ld_instance;
            double* ptr_A = &h_A[(ia - 1) + (ja - 1) * m_global_A];
            double* ptr_B = &h_B_s[(ib - 1) + (jb - 1) * m_global_B];
            double* ptr_X = &h_X[(ib - 1) + (jb - 1) * m_global_B];

            /* Only the leading n rows of the gathered block are the solution X; the
             * trailing m-n rows hold the Q^H*B residual block, so we read just n rows. */
            /* Compute the relative residual; B is overwritten in place with B - A*X */
            const double rel_err = compute_error(m, n, nrhs, ptr_A, m_global_A, ptr_X, m_global_B, ptr_B, m_global_B);
            const double tol     = 1.0e-10;
            const int    ok      = (rel_err < tol);

            printf("  solve %d: |b - A*x|/(max(m,n)*|A|*|x|) = %E  ->  %s  (threshold: %E)\n",
                   s,
                   rel_err,
                   ok ? "PASS" : "FAIL",
                   tol);
            sample_ok = sample_ok && ok;
        }
    }

    /* =========================================== */
    /*        CLEAN UP HOST WORKSPACE              */
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

        if (h_X)
        {
            free(h_X);
            h_X = NULL;
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

    if (d_tau)
    {
        cudaStat = cudaFree(d_tau);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_tau = NULL;
    }

    if (d_B)
    {
        cudaStat = cudaFree(d_B);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_B = NULL;
    }

    if (d_work)
    {
        cudaStat = cudaFree(d_work);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_work = NULL;
    }

    if (d_info)
    {
        cudaStat = cudaFree(d_info);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
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
    /*                      CLEANUP                */
    /* =========================================== */

    /* Destroy cuSOLVERMp handle */
    cusolverStat = cusolverMpDestroy(handle);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* Synchronize before destroying the NCCL communicator */
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* Destroy the NCCL communicator */
    ncclStat = ncclCommDestroy(ncclComm);
    SAMPLE_ASSERT(ncclStat == ncclSuccess);

    /* Destroy the CUDA stream */
    cudaStat = cudaStreamDestroy(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* MPI barrier before MPI_Finalize */
    MPI_Barrier(mpi_global_comm);
    sample_ok = sample_all_ranks_succeeded(sample_ok);

    /* Finalize MPI environment */
    MPI_Finalize();

    if (rank == 0)
    {
        printf("%s\n", sample_ok ? "[SUCCEEDED]" : "[FAILED]");
    }

    return sample_ok ? 0 : 1;
}
