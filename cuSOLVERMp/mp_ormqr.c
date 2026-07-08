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
 * cuSOLVERMp ORMQR sample
 *
 * Demonstrates applying the Q factor represented by distributed GEQRF
 * Householder reflectors:
 *   1. cusolverMpGeqrf computes reflectors for A.
 *   2. cusolverMpOrmqr applies Q to C.
 *   3. cusolverMpOrmqr applies Q^T to recover the original C.
 *
 * Usage:
 *   mpirun -n 2 ./mp_ormqr -p 1 -q 2
 *   mpirun -n 4 ./mp_ormqr -p 2 -q 2 -m 128 -n 64 -nrhs 64 -mbA 32 -nbA 32 -mbB 32 -nbB 32
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

static void generate_random_matrix(int64_t m, int64_t n, double* A, int64_t lda, unsigned int seed)
{
    srand(seed);
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < m; i++)
        {
            A[i + j * lda] = (double)rand() / RAND_MAX;
        }
    }
}

static double frobenius_norm(int64_t m, int64_t n, const double* A, int64_t lda)
{
    double norm = 0.0;
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < m; i++)
        {
            const double val = A[i + j * lda];
            norm += val * val;
        }
    }
    return sqrt(norm);
}

static int64_t count_spanned_tiles(int64_t first_index, int64_t extent, int64_t tile_size)
{
    if (extent == 0)
    {
        return 0;
    }

    const int64_t first_tile = (first_index - 1) / tile_size;
    const int64_t last_tile  = (first_index - 1 + extent - 1) / tile_size;
    return last_tile - first_tile + 1;
}

int main(int argc, char* argv[])
{
    Options opts = { .m           = 64,
                     .n           = 32,
                     .nrhs        = 32,
                     .mbA         = 16,
                     .nbA         = 16,
                     .mbB         = 16,
                     .nbB         = 16,
                     .mbQ         = 16,
                     .nbQ         = 16,
                     .mbZ         = 16,
                     .nbZ         = 16,
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

    MPI_Init(NULL, NULL);

    int rank, commSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    if (rank == 0) print(&opts);

    const int64_t m     = opts.m;
    const int64_t k     = opts.n;
    const int64_t n     = opts.nrhs;
    const int     nprow = opts.p;
    const int     npcol = opts.q;

    SAMPLE_ASSERT(commSize == nprow * npcol && "MPI rank count must match p*q");
    SAMPLE_ASSERT(m >= k && "this sample requires m >= n for the GEQRF input A");
    SAMPLE_ASSERT(opts.mbA == opts.mbB && "this sample configuration requires mbA == mbB");
    SAMPLE_ASSERT((commSize <= 1 || opts.ia == opts.ib) &&
                  "this sample configuration requires ia == ib for multi-rank ORMQR");

    if (commSize > 1 && npcol > 1)
    {
        const int64_t reflector_tiles = count_spanned_tiles(opts.ja, k, opts.nbA);
        const int64_t c_tiles         = count_spanned_tiles(opts.jb, n, opts.nbB);
        SAMPLE_ASSERT(reflector_tiles <= c_tiles &&
                      "this sample configuration requires sub(C) to span at least as many column tiles as "
                      "the reflector panel");
    }

    const cusolverMpGridMapping_t gridLayout =
            (opts.grid_layout == 'C' || opts.grid_layout == 'c' ? CUSOLVERMP_GRID_MAPPING_COL_MAJOR
                                                                : CUSOLVERMP_GRID_MAPPING_ROW_MAJOR);

    const uint32_t rsrc = 0;
    const uint32_t csrc = 0;

    const int localDeviceId = getLocalRank();

    cudaError_t cudaStat = cudaSetDevice(localDeviceId);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    ncclComm_t   ncclComm = createNcclComm(commSize, rank);
    ncclResult_t ncclStat = ncclSuccess;

    cudaStream_t stream = NULL;
    cudaStat            = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cusolverMpHandle_t handle       = NULL;
    cusolverStatus_t   cusolverStat = cusolverMpCreate(&handle, localDeviceId, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverMpGrid_t grid = NULL;
    cusolverStat          = cusolverMpCreateDeviceGrid(handle, &grid, ncclComm, nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

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

    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ic = opts.ib;
    const int64_t jc = opts.jb;

    const int64_t mA_global = (ia - 1) + m;
    const int64_t nA_global = (ja - 1) + k;
    const int64_t mC_global = (ic - 1) + m;
    const int64_t nC_global = (jc - 1) + n;

    const int64_t mA_local = cusolverMpNUMROC(mA_global, opts.mbA, myprow, rsrc, nprow);
    const int64_t nA_local = cusolverMpNUMROC(nA_global, opts.nbA, mypcol, csrc, npcol);
    const int64_t mC_local = cusolverMpNUMROC(mC_global, opts.mbB, myprow, rsrc, nprow);
    const int64_t nC_local = cusolverMpNUMROC(nC_global, opts.nbB, mypcol, csrc, npcol);

    const int64_t lldA = (mA_local > 0) ? mA_local : 1;
    const int64_t lldC = (mC_local > 0) ? mC_local : 1;

    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverStat                       = cusolverMpCreateMatrixDesc(
            &descA, grid, CUDA_R_64F, mA_global, nA_global, opts.mbA, opts.nbA, rsrc, csrc, lldA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverMpMatrixDescriptor_t descC = NULL;
    cusolverStat                       = cusolverMpCreateMatrixDesc(
            &descC, grid, CUDA_R_64F, mC_global, nC_global, opts.mbB, opts.nbB, rsrc, csrc, lldC);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    void* d_A = NULL;
    cudaStat  = cudaMalloc((void**)&d_A, ((mA_local * nA_local) > 0 ? (mA_local * nA_local) : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    void* d_C = NULL;
    cudaStat  = cudaMalloc((void**)&d_C, ((mC_local * nC_local) > 0 ? (mC_local * nC_local) : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    void* d_tau = NULL;
    cudaStat    = cudaMalloc((void**)&d_tau, ((nA_local > 0) ? nA_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    int* d_info = NULL;
    cudaStat    = cudaMalloc((void**)&d_info, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    double* h_A      = NULL;
    double* h_C      = NULL;
    double* h_C_orig = NULL;
    if (rank == 0)
    {
        h_A      = (double*)calloc(mA_global * nA_global, sizeof(double));
        h_C      = (double*)calloc(mC_global * nC_global, sizeof(double));
        h_C_orig = (double*)calloc(mC_global * nC_global, sizeof(double));
        SAMPLE_ASSERT(h_A != NULL && h_C != NULL && h_C_orig != NULL);

        generate_random_matrix(m, k, &h_A[(ia - 1) + (ja - 1) * mA_global], mA_global, 42);
        generate_random_matrix(m, n, &h_C[(ic - 1) + (jc - 1) * mC_global], mC_global, 777);
        memcpy(h_C_orig, h_C, mC_global * nC_global * sizeof(double));
    }

    cusolverStat = cusolverMpMatrixScatterH2D(handle, mA_global, nA_global, d_A, 1, 1, descA, 0, h_A, mA_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpMatrixScatterH2D(handle, mC_global, nC_global, d_C, 1, 1, descC, 0, h_C, mC_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    size_t geqrf_d_bytes = 0, geqrf_h_bytes = 0;
    size_t ormqr_d_bytes_n = 0, ormqr_h_bytes_n = 0;
    size_t ormqr_d_bytes_t = 0, ormqr_h_bytes_t = 0;

    cusolverStat =
            cusolverMpGeqrf_bufferSize(handle, m, k, d_A, ia, ja, descA, CUDA_R_64F, &geqrf_d_bytes, &geqrf_h_bytes);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpOrmqr_bufferSize(handle,
                                              CUBLAS_SIDE_LEFT,
                                              CUBLAS_OP_N,
                                              m,
                                              n,
                                              k,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              d_tau,
                                              d_C,
                                              ic,
                                              jc,
                                              descC,
                                              CUDA_R_64F,
                                              &ormqr_d_bytes_n,
                                              &ormqr_h_bytes_n);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpOrmqr_bufferSize(handle,
                                              CUBLAS_SIDE_LEFT,
                                              CUBLAS_OP_T,
                                              m,
                                              n,
                                              k,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              d_tau,
                                              d_C,
                                              ic,
                                              jc,
                                              descC,
                                              CUDA_R_64F,
                                              &ormqr_d_bytes_t,
                                              &ormqr_h_bytes_t);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    size_t d_work_bytes = geqrf_d_bytes;
    if (ormqr_d_bytes_n > d_work_bytes) d_work_bytes = ormqr_d_bytes_n;
    if (ormqr_d_bytes_t > d_work_bytes) d_work_bytes = ormqr_d_bytes_t;

    size_t h_work_bytes = geqrf_h_bytes;
    if (ormqr_h_bytes_n > h_work_bytes) h_work_bytes = ormqr_h_bytes_n;
    if (ormqr_h_bytes_t > h_work_bytes) h_work_bytes = ormqr_h_bytes_t;

    void* d_work = NULL;
    cudaStat     = cudaMalloc((void**)&d_work, (d_work_bytes > 0 ? d_work_bytes : 1));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    void* h_work = malloc(h_work_bytes > 0 ? h_work_bytes : 1);
    SAMPLE_ASSERT(h_work != NULL);

    if (rank == 0) printf("\nStep 1: GEQRF factorization of A...\n");
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cusolverStat = cusolverMpGeqrf(
            handle, m, k, d_A, ia, ja, descA, d_tau, CUDA_R_64F, d_work, geqrf_d_bytes, h_work, geqrf_h_bytes, d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    int h_info = 0;
    cudaStat   = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    if (rank == 0) printf("Step 2: Apply Q to C with ORMQR...\n");
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cusolverStat = cusolverMpOrmqr(handle,
                                   CUBLAS_SIDE_LEFT,
                                   CUBLAS_OP_N,
                                   m,
                                   n,
                                   k,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_tau,
                                   d_C,
                                   ic,
                                   jc,
                                   descC,
                                   CUDA_R_64F,
                                   d_work,
                                   ormqr_d_bytes_n,
                                   h_work,
                                   ormqr_h_bytes_n,
                                   d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    if (rank == 0) printf("Step 3: Apply Q^T to recover C...\n");
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cusolverStat = cusolverMpOrmqr(handle,
                                   CUBLAS_SIDE_LEFT,
                                   CUBLAS_OP_T,
                                   m,
                                   n,
                                   k,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_tau,
                                   d_C,
                                   ic,
                                   jc,
                                   descC,
                                   CUDA_R_64F,
                                   d_work,
                                   ormqr_d_bytes_t,
                                   h_work,
                                   ormqr_h_bytes_t,
                                   d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    cusolverStat = cusolverMpMatrixGatherD2H(handle, mC_global, nC_global, d_C, 1, 1, descC, 0, h_C, mC_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0)
    {
        double* h_diff = (double*)calloc(m * n, sizeof(double));
        SAMPLE_ASSERT(h_diff != NULL);
        for (int64_t j = 0; j < n; j++)
        {
            for (int64_t i = 0; i < m; i++)
            {
                const int64_t src_idx = (ic - 1 + i) + (jc - 1 + j) * mC_global;
                h_diff[i + j * m]     = h_C[src_idx] - h_C_orig[src_idx];
            }
        }

        const double norm_orig = frobenius_norm(m, n, &h_C_orig[(ic - 1) + (jc - 1) * mC_global], mC_global);
        const double norm_diff = frobenius_norm(m, n, h_diff, m);
        const double rel_err   = norm_diff / (norm_orig > 1.0 ? norm_orig : 1.0);

        printf("\nVerification:\n");
        printf("  ||Q^T*(Q*C) - C|| / max(||C||, 1) = %E\n", rel_err);
        sample_ok = sample_ok && (rel_err < 1.0e-12);
        printf("  Round-trip check: %s\n", sample_ok ? "PASS" : "FAIL");

        free(h_diff);
    }

    if (rank == 0)
    {
        free(h_A);
        free(h_C);
        free(h_C_orig);
    }

    if (d_A != NULL)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_A = NULL;
    }
    if (d_C != NULL)
    {
        cudaStat = cudaFree(d_C);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_C = NULL;
    }
    if (d_tau != NULL)
    {
        cudaStat = cudaFree(d_tau);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_tau = NULL;
    }
    if (d_info != NULL)
    {
        cudaStat = cudaFree(d_info);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_info = NULL;
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

    cusolverStat = cusolverMpDestroyMatrixDesc(descC);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpDestroyGrid(grid);
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
