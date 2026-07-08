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
 * cuSOLVERMp SYGST sample
 *
 * Demonstrates reducing a generalized symmetric eigenproblem to standard form:
 *   C = inv(L) * A * inv(L^T), where L is the lower triangular factor in B.
 *
 * Usage:
 *   mpirun -n 2 ./mp_sygst -p 1 -q 2
 *   mpirun -n 4 ./mp_sygst -p 2 -q 2 -n 128 -mbA 32 -nbA 32 -mbB 32 -nbB 32
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#include <cusolverMp.h>

#include "helpers.h"

static void generate_symmetric_matrix(int64_t n, double* A, int64_t lda)
{
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < n; i++)
        {
            A[i + j * lda] = 0.0;
        }
    }

    for (int64_t j = 0; j < n; j++)
    {
        double col_sum = 0.0;
        for (int64_t i = j; i < n; i++)
        {
            const double value = (i == j) ? 0.0 : 0.01 * (double)(1 + ((i + 2) * (j + 3)) % 11);
            A[i + j * lda]     = value;
            A[j + i * lda]     = value;
            col_sum += fabs(value);
        }
        A[j + j * lda] = (double)(n + j + 1) + col_sum;
    }
}

static void generate_lower_factor(int64_t n, double* L, int64_t ldl)
{
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < n; i++)
        {
            L[i + j * ldl] = 0.0;
        }
    }

    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = j; i < n; i++)
        {
            L[i + j * ldl] = (i == j) ? 2.0 + 0.01 * (double)(i + 1) : 0.002 * (double)(1 + ((i + 5) * (j + 7)) % 13);
        }
    }
}

static double sygst_residual(int64_t n, const double* A, const double* L, const double* C)
{
    double norm_A = 0.0;
    double norm_R = 0.0;

    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < n; i++)
        {
            double recon = 0.0;
            for (int64_t k = 0; k < n; k++)
            {
                const double Lik = (i >= k) ? L[i + k * n] : 0.0;
                for (int64_t l = 0; l < n; l++)
                {
                    const double Ljl = (j >= l) ? L[j + l * n] : 0.0;
                    recon += Lik * C[k + l * n] * Ljl;
                }
            }

            const double diff = A[i + j * n] - recon;
            norm_R += diff * diff;
            norm_A += A[i + j * n] * A[i + j * n];
        }
    }

    return sqrt(norm_R) / fmax(sqrt(norm_A), 1.0);
}

static void hermitianize_lower(int64_t n, double* A)
{
    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < j; i++)
        {
            A[i + j * n] = A[j + i * n];
        }
    }
}

int main(int argc, char* argv[])
{
    Options opts = { .m           = 1,
                     .n           = 64,
                     .nrhs        = 1,
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

    const int64_t n = opts.n;
    SAMPLE_ASSERT(opts.ia == 1 && opts.ja == 1 && opts.ib == 1 && opts.jb == 1 && "this sample requires ia=ja=ib=jb=1");
    SAMPLE_ASSERT(opts.mbA == opts.nbA && opts.mbB == opts.nbB && opts.mbA == opts.mbB &&
                  "SYGST sample requires mbA=nbA=mbB=nbB");

    const int nprow = opts.p;
    const int npcol = opts.q;
    SAMPLE_ASSERT(commSize == nprow * npcol && "MPI rank count must match p*q");

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
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;

    const int64_t nA_global = n + ia - 1;
    const int64_t nB_global = n + ib - 1;
    const int64_t mA_local  = cusolverMpNUMROC(nA_global, opts.mbA, myprow, rsrc, nprow);
    const int64_t nA_local  = cusolverMpNUMROC(nA_global, opts.nbA, mypcol, csrc, npcol);
    const int64_t mB_local  = cusolverMpNUMROC(nB_global, opts.mbB, myprow, rsrc, nprow);
    const int64_t nB_local  = cusolverMpNUMROC(nB_global, opts.nbB, mypcol, csrc, npcol);

    const int64_t lldA = (mA_local > 0) ? mA_local : 1;
    const int64_t lldB = (mB_local > 0) ? mB_local : 1;

    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverStat                       = cusolverMpCreateMatrixDesc(
            &descA, grid, CUDA_R_64F, nA_global, nA_global, opts.mbA, opts.nbA, rsrc, csrc, lldA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverMpMatrixDescriptor_t descB = NULL;
    cusolverStat                       = cusolverMpCreateMatrixDesc(
            &descB, grid, CUDA_R_64F, nB_global, nB_global, opts.mbB, opts.nbB, rsrc, csrc, lldB);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    void* d_A    = NULL;
    void* d_B    = NULL;
    int*  d_info = NULL;
    int   h_info = 0;

    cudaStat = cudaMalloc(&d_A, (mA_local * nA_local > 0 ? mA_local * nA_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc(&d_B, (mB_local * nB_local > 0 ? mB_local * nB_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void**)&d_info, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    double* h_A      = NULL;
    double* h_A_orig = NULL;
    double* h_B      = NULL;
    if (rank == 0)
    {
        h_A      = (double*)calloc(nA_global * nA_global, sizeof(double));
        h_A_orig = (double*)calloc(nA_global * nA_global, sizeof(double));
        h_B      = (double*)calloc(nB_global * nB_global, sizeof(double));
        SAMPLE_ASSERT(h_A != NULL && h_A_orig != NULL && h_B != NULL);

        generate_symmetric_matrix(n, &h_A[(ia - 1) + (ja - 1) * nA_global], nA_global);
        generate_lower_factor(n, &h_B[(ib - 1) + (jb - 1) * nB_global], nB_global);
        for (int64_t j = 0; j < nA_global; j++)
        {
            for (int64_t i = 0; i < nA_global; i++)
            {
                h_A_orig[i + j * nA_global] = h_A[i + j * nA_global];
            }
        }
    }

    cusolverStat = cusolverMpMatrixScatterH2D(handle, nA_global, nA_global, d_A, 1, 1, descA, 0, h_A, nA_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpMatrixScatterH2D(handle, nB_global, nB_global, d_B, 1, 1, descB, 0, h_B, nB_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    size_t d_work_bytes = 0;
    size_t h_work_bytes = 0;
    cusolverStat        = cusolverMpSygst_bufferSize(handle,
                                              CUSOLVER_EIG_TYPE_1,
                                              CUBLAS_FILL_MODE_LOWER,
                                              n,
                                              ia,
                                              ja,
                                              descA,
                                              ib,
                                              jb,
                                              descB,
                                              CUDA_R_64F,
                                              &d_work_bytes,
                                              &h_work_bytes);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    void* d_work = NULL;
    cudaStat     = cudaMalloc(&d_work, d_work_bytes > 0 ? d_work_bytes : 1);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    void* h_work = malloc(h_work_bytes > 0 ? h_work_bytes : 1);
    SAMPLE_ASSERT(h_work != NULL);

    if (rank == 0) printf("\nRunning SYGST with itype=1 and uplo=lower...\n");
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cusolverStat = cusolverMpSygst(handle,
                                   CUSOLVER_EIG_TYPE_1,
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_B,
                                   ib,
                                   jb,
                                   descB,
                                   CUDA_R_64F,
                                   d_work,
                                   d_work_bytes,
                                   h_work,
                                   h_work_bytes,
                                   d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    cusolverStat = cusolverMpMatrixGatherD2H(handle, nA_global, nA_global, d_A, 1, 1, descA, 0, h_A, nA_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0)
    {
        hermitianize_lower(n, &h_A[(ia - 1) + (ja - 1) * nA_global]);
        const double residual = sygst_residual(n,
                                               &h_A_orig[(ia - 1) + (ja - 1) * nA_global],
                                               &h_B[(ib - 1) + (jb - 1) * nB_global],
                                               &h_A[(ia - 1) + (ja - 1) * nA_global]);

        printf("\nVerification:\n");
        printf("  ||A - L*C*L^T|| / max(||A||, 1) = %E\n", residual);
        sample_ok = sample_ok && (residual < 1.0e-11);
        printf("  SYGST check: %s\n", sample_ok ? "PASS" : "FAIL");
    }

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

    if (rank == 0)
    {
        free(h_A);
        free(h_A_orig);
        free(h_B);
    }

    cusolverStat = cusolverMpDestroyMatrixDesc(descB);
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
