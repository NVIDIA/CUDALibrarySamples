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
 * cuSOLVERMp SYTRD + STEDC + ORMTR sample
 *
 * Demonstrates the eigensolver workflow from the lower triangle of a
 * distributed symmetric matrix:
 *   1. cusolverMpSytrd reduces A to tridiagonal T and stores Householder data.
 *   2. cusolverMpStedc computes eigenpairs of T.
 *   3. cusolverMpOrmtr applies the SYTRD reflectors to produce eigenvectors of A.
 *
 * Usage:
 *   mpirun -n 2 ./mp_sytrd_stedc_ormtr -p 1 -q 2
 *   mpirun -n 4 ./mp_sytrd_stedc_ormtr -p 2 -q 2 -n 64 -mbA 16 -nbA 16 -mbQ 16 -nbQ 16
 */

#include <stdbool.h>
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
        for (int64_t i = j; i < n; i++)
        {
            const double value = (i == j) ? (double)(n + i + 1) : 0.25 + 0.01 * (double)(((i + 3) * (j + 5)) % 17);
            A[i + j * lda]     = value;
            A[j + i * lda]     = value;
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
            const double value = A[i + j * lda];
            norm += value * value;
        }
    }
    return sqrt(norm);
}

static double eigen_residual(int64_t n, const double* A, const double* Z, const double* W)
{
    double norm_A = 0.0;
    double norm_R = 0.0;

    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < n; i++)
        {
            double az = 0.0;
            for (int64_t k = 0; k < n; k++)
            {
                az += A[i + k * n] * Z[k + j * n];
            }

            const double zw   = Z[i + j * n] * W[j];
            const double diff = az - zw;
            norm_R += diff * diff;
            norm_A += A[i + j * n] * A[i + j * n];
        }
    }

    return sqrt(norm_R) / fmax(sqrt(norm_A), 1.0);
}

static double orthogonality_error(int64_t n, const double* Z)
{
    double norm_R = 0.0;

    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < n; i++)
        {
            double dot = 0.0;
            for (int64_t k = 0; k < n; k++)
            {
                dot += Z[k + i * n] * Z[k + j * n];
            }

            const double expected = (i == j) ? 1.0 : 0.0;
            const double diff     = dot - expected;
            norm_R += diff * diff;
        }
    }

    return sqrt(norm_R) / (double)n;
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
    SAMPLE_ASSERT(opts.ia == 1 && opts.ja == 1 && opts.iq == 1 && opts.jq == 1 &&
                  "this workflow sample requires ia=ja=iq=jq=1");
    SAMPLE_ASSERT(opts.mbA == opts.nbA && opts.mbQ == opts.nbQ && opts.mbA == opts.mbQ &&
                  "this workflow sample requires mbA=nbA=mbQ=nbQ");

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
    const int64_t iq = opts.iq;
    const int64_t jq = opts.jq;

    const int64_t nA_global = n + ia - 1;
    const int64_t nQ_global = n + iq - 1;

    const int64_t mA_local = cusolverMpNUMROC(nA_global, opts.mbA, myprow, rsrc, nprow);
    const int64_t nA_local = cusolverMpNUMROC(nA_global, opts.nbA, mypcol, csrc, npcol);
    const int64_t mQ_local = cusolverMpNUMROC(nQ_global, opts.mbQ, myprow, rsrc, nprow);
    const int64_t nQ_local = cusolverMpNUMROC(nQ_global, opts.nbQ, mypcol, csrc, npcol);
    const int64_t nD_local = cusolverMpNUMROC(n, opts.nbA, mypcol, csrc, npcol);
    const int64_t lldA     = mA_local > 0 ? mA_local : 1;
    const int64_t lldQ     = mQ_local > 0 ? mQ_local : 1;

    cusolverMpMatrixDescriptor_t descA = NULL;
    cusolverStat                       = cusolverMpCreateMatrixDesc(
            &descA, grid, CUDA_R_64F, nA_global, nA_global, opts.mbA, opts.nbA, rsrc, csrc, lldA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverMpMatrixDescriptor_t descQ = NULL;
    cusolverStat                       = cusolverMpCreateMatrixDesc(
            &descQ, grid, CUDA_R_64F, nQ_global, nQ_global, opts.mbQ, opts.nbQ, rsrc, csrc, lldQ);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverMpMatrixDescriptor_t descDiag = NULL;
    cusolverStat = cusolverMpCreateMatrixDesc(&descDiag, grid, CUDA_R_64F, 1, n, 1, opts.nbA, rsrc, csrc, 1);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    void* d_A       = NULL;
    void* d_Q       = NULL;
    void* d_D_sytrd = NULL;
    void* d_E_sytrd = NULL;
    void* d_tau     = NULL;
    void* d_D_stedc = NULL;
    void* d_E_stedc = NULL;
    int*  d_info    = NULL;

    cudaStat = cudaMalloc(&d_A, (mA_local * nA_local > 0 ? mA_local * nA_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc(&d_Q, (mQ_local * nQ_local > 0 ? mQ_local * nQ_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc(&d_D_sytrd, (nD_local > 0 ? nD_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc(&d_E_sytrd, (nD_local > 0 ? nD_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc(&d_tau, (nD_local > 0 ? nD_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc(&d_D_stedc, n * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc(&d_E_stedc, n * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void**)&d_info, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cudaStat = cudaMemset(d_D_sytrd, 0, (nD_local > 0 ? nD_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMemset(d_E_sytrd, 0, (nD_local > 0 ? nD_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMemset(d_tau, 0, (nD_local > 0 ? nD_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMemset(d_Q, 0, (mQ_local * nQ_local > 0 ? mQ_local * nQ_local : 1) * sizeof(double));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    double* h_A = NULL;
    double* h_Z = NULL;
    double* h_D = (double*)calloc(n, sizeof(double));
    double* h_E = (double*)calloc(n, sizeof(double));
    SAMPLE_ASSERT(h_D != NULL && h_E != NULL);

    if (rank == 0)
    {
        h_A = (double*)calloc(nA_global * nA_global, sizeof(double));
        h_Z = (double*)calloc(nQ_global * nQ_global, sizeof(double));
        SAMPLE_ASSERT(h_A != NULL && h_Z != NULL);
        generate_symmetric_matrix(n, &h_A[(ia - 1) + (ja - 1) * nA_global], nA_global);
    }

    cusolverStat = cusolverMpMatrixScatterH2D(handle, nA_global, nA_global, d_A, 1, 1, descA, 0, h_A, nA_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    size_t        sytrd_d_bytes = 0, sytrd_h_bytes = 0;
    size_t        stedc_d_bytes = 0, stedc_h_bytes = 0;
    size_t        ormtr_d_bytes = 0, ormtr_h_bytes = 0;
    const int64_t iwork_len = n > (int64_t)(2 * commSize + 1) ? n : (int64_t)(2 * commSize + 1);
    int*          h_iwork   = (int*)calloc(iwork_len, sizeof(int));
    SAMPLE_ASSERT(h_iwork != NULL);
    char compz = 'I';

    cusolverStat = cusolverMpSytrd_bufferSize(handle,
                                              CUBLAS_FILL_MODE_LOWER,
                                              n,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              d_D_sytrd,
                                              d_E_sytrd,
                                              d_tau,
                                              CUDA_R_64F,
                                              &sytrd_d_bytes,
                                              &sytrd_h_bytes);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpStedc_bufferSize(handle,
                                              &compz,
                                              n,
                                              d_D_stedc,
                                              d_E_stedc,
                                              d_Q,
                                              iq,
                                              jq,
                                              descQ,
                                              CUDA_R_64F,
                                              &stedc_d_bytes,
                                              &stedc_h_bytes,
                                              h_iwork);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpOrmtr_bufferSize(handle,
                                              CUBLAS_SIDE_LEFT,
                                              CUBLAS_FILL_MODE_LOWER,
                                              CUBLAS_OP_N,
                                              n,
                                              n,
                                              d_A,
                                              ia,
                                              ja,
                                              descA,
                                              d_tau,
                                              d_Q,
                                              iq,
                                              jq,
                                              descQ,
                                              CUDA_R_64F,
                                              &ormtr_d_bytes,
                                              &ormtr_h_bytes);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    size_t d_work_bytes = sytrd_d_bytes;
    if (stedc_d_bytes > d_work_bytes) d_work_bytes = stedc_d_bytes;
    if (ormtr_d_bytes > d_work_bytes) d_work_bytes = ormtr_d_bytes;

    size_t h_work_bytes = sytrd_h_bytes;
    if (stedc_h_bytes > h_work_bytes) h_work_bytes = stedc_h_bytes;
    if (ormtr_h_bytes > h_work_bytes) h_work_bytes = ormtr_h_bytes;

    void* d_work = NULL;
    cudaStat     = cudaMalloc(&d_work, d_work_bytes > 0 ? d_work_bytes : 1);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    void* h_work = malloc(h_work_bytes > 0 ? h_work_bytes : 1);
    SAMPLE_ASSERT(h_work != NULL);
    int h_info = 0;

    if (rank == 0) printf("\nStep 1: SYTRD reduction of A...\n");
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cusolverStat = cusolverMpSytrd(handle,
                                   CUBLAS_FILL_MODE_LOWER,
                                   n,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_D_sytrd,
                                   d_E_sytrd,
                                   d_tau,
                                   CUDA_R_64F,
                                   d_work,
                                   sytrd_d_bytes,
                                   h_work,
                                   sytrd_h_bytes,
                                   d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    cusolverStat = cusolverMpMatrixGatherD2H(handle, 1, n, d_D_sytrd, 1, 1, descDiag, 0, h_D, 1);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpMatrixGatherD2H(handle, 1, n, d_E_sytrd, 1, 1, descDiag, 0, h_E, 1);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    MPI_Bcast(h_D, (int)n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_E, (int)n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cudaStat = cudaMemcpy(d_D_stedc, h_D, n * sizeof(double), cudaMemcpyHostToDevice);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMemcpy(d_E_stedc, h_E, n * sizeof(double), cudaMemcpyHostToDevice);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0) printf("Step 2: STEDC solve of tridiagonal T...\n");
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cusolverStat = cusolverMpStedc(handle,
                                   &compz,
                                   n,
                                   d_D_stedc,
                                   d_E_stedc,
                                   d_Q,
                                   iq,
                                   jq,
                                   descQ,
                                   CUDA_R_64F,
                                   d_work,
                                   stedc_d_bytes,
                                   h_work,
                                   stedc_h_bytes,
                                   d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    cudaStat = cudaMemcpy(h_D, d_D_stedc, n * sizeof(double), cudaMemcpyDeviceToHost);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0) printf("Step 3: ORMTR apply of SYTRD reflectors...\n");
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cusolverStat = cusolverMpOrmtr(handle,
                                   CUBLAS_SIDE_LEFT,
                                   CUBLAS_FILL_MODE_LOWER,
                                   CUBLAS_OP_N,
                                   n,
                                   n,
                                   d_A,
                                   ia,
                                   ja,
                                   descA,
                                   d_tau,
                                   d_Q,
                                   iq,
                                   jq,
                                   descQ,
                                   CUDA_R_64F,
                                   d_work,
                                   ormtr_d_bytes,
                                   h_work,
                                   ormtr_h_bytes,
                                   d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaMemcpyAsync(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    SAMPLE_ASSERT(h_info == 0);

    cusolverStat = cusolverMpMatrixGatherD2H(handle, nQ_global, nQ_global, d_Q, 1, 1, descQ, 0, h_Z, nQ_global);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0)
    {
        const double residual = eigen_residual(n, &h_A[(ia - 1) + (ja - 1) * nA_global], h_Z, h_D);
        const double ortho    = orthogonality_error(n, h_Z);
        const double norm_A   = frobenius_norm(n, n, &h_A[(ia - 1) + (ja - 1) * nA_global], nA_global);

        printf("\nVerification:\n");
        printf("  ||A*Z - Z*D|| / max(||A||, 1) = %E\n", residual);
        printf("  ||I - Z^T*Z|| / N = %E\n", ortho);
        printf("  ||A||_F = %E\n", norm_A);
        sample_ok = sample_ok && (residual < 1.0e-10 && ortho < 1.0e-10);
        printf("  Eigenpair check: %s\n", sample_ok ? "PASS" : "FAIL");
    }

    if (d_A != NULL)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_A = NULL;
    }
    if (d_Q != NULL)
    {
        cudaStat = cudaFree(d_Q);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_Q = NULL;
    }
    if (d_D_sytrd != NULL)
    {
        cudaStat = cudaFree(d_D_sytrd);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_D_sytrd = NULL;
    }
    if (d_E_sytrd != NULL)
    {
        cudaStat = cudaFree(d_E_sytrd);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_E_sytrd = NULL;
    }
    if (d_tau != NULL)
    {
        cudaStat = cudaFree(d_tau);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_tau = NULL;
    }
    if (d_D_stedc != NULL)
    {
        cudaStat = cudaFree(d_D_stedc);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_D_stedc = NULL;
    }
    if (d_E_stedc != NULL)
    {
        cudaStat = cudaFree(d_E_stedc);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_E_stedc = NULL;
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
    if (h_iwork != NULL)
    {
        free(h_iwork);
        h_iwork = NULL;
    }
    if (h_D != NULL)
    {
        free(h_D);
        h_D = NULL;
    }
    if (h_E != NULL)
    {
        free(h_E);
        h_E = NULL;
    }
    if (rank == 0)
    {
        free(h_A);
        free(h_Z);
    }

    cusolverStat = cusolverMpDestroyMatrixDesc(descDiag);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpDestroyMatrixDesc(descQ);
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
