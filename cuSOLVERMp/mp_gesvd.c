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
 * cusolverMpGesvd sample: distributed singular value decomposition
 *
 * Computes the THIN SVD A = U * diag(S) * V^T where:
 *   - A is M-by-N (overwritten with intermediate Up; not preserved)
 *   - U is M-by-K   (left singular vectors, K = min(M, N))
 *   - S is length K (singular values, descending order, replicated on every rank)
 *   - V^T is K-by-N (right singular vectors, transposed)
 *
 * Implementation: QDWH polar (A = Up * H) followed by an internal
 * Hermitian eigensolver on H. Exact (non-perturbed) SVD; numerical failure
 * before singular values are complete reports d_info = +1.
 *
 * Usage:
 *   mpirun -np 2 mp_gesvd
 *   mpirun -np 4 mp_gesvd -m 256 -n 128 -mbA 32 -nbA 32 -p 2 -q 2
 *   mpirun -np 4 mp_gesvd -m 128 -n 256 -mbA 32 -nbA 32 -p 2 -q 2     # m<n wide path
 *
 * Parameters:
 *   -m, -n     : matrix dimensions
 *   -mbA, -nbA : tile sizes (must be equal: mbA == nbA)
 *   -p, -q     : process grid dimensions
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

/* Generate a random M-by-N matrix with a diagonal boost so the singular
 * values are well separated from zero (helps QDWH converge quickly). */
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
    int64_t minmn = (m < n) ? m : n;
    for (int64_t i = 0; i < minmn; i++)
    {
        A[i + i * lda] += (double)minmn;
    }
}

int main(int argc, char* argv[])
{
    /* ================================================================
     * 1. Parse command-line options
     * ================================================================ */
    Options opts = {
        .m = 128, .n = 128, .nrhs = 1,
        .mbA = 32, .nbA = 32, .mbB = 32, .nbB = 32,
        .mbQ = 32, .nbQ = 32, .mbZ = 32, .nbZ = 32,
        .ia = 1, .ja = 1, .ib = 1, .jb = 1,
        .iq = 1, .jq = 1, .iz = 1, .jz = 1,
        .p = 2, .q = 1,
        .grid_layout = 'C',
        .verbose = false
    };
    parse(&opts, argc, argv);
    validate(&opts);
    int sample_ok = 1;

    const int64_t m   = opts.m;
    const int64_t n   = opts.n;
    const int64_t k   = (m < n) ? m : n;   /* min(m, n) */
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int nprow = opts.p;
    const int npcol = opts.q;

    /* ================================================================
     * 2. MPI initialization
     * ================================================================ */
    MPI_Init(NULL, NULL);

    int rank, commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Sample-configuration checks run after MPI_Init so a failure routes
     * through MPI_Abort (via SAMPLE_ASSERT) rather than a bare abort(). */
    SAMPLE_ASSERT(mbA == nbA && "tile sizes must be equal (mbA == nbA)");
    SAMPLE_ASSERT(commSize == nprow * npcol);

    if (k == 0)
    {
        sample_ok = sample_all_ranks_succeeded(sample_ok);
        if (rank == 0)
        {
            printf("cusolverMpGesvd sample: A (%ld x %ld) has empty spectrum (K=0)\n",
                   (long)m, (long)n);
            printf("%s\n", sample_ok ? "[SUCCEEDED]" : "[FAILED]");
        }
        MPI_Finalize();
        return sample_ok ? 0 : 1;
    }

    if (rank == 0)
    {
        printf("cusolverMpGesvd sample: A (%ld x %ld) = U * diag(S) * V^T\n",
               (long)m, (long)n);
        printf("  thin SVD: U (%ld x %ld), S (%ld), V^T (%ld x %ld)\n",
               (long)m, (long)k, (long)k, (long)k, (long)n);
        printf("  tile size: %ld x %ld, grid: %d x %d, ranks: %d\n\n",
               (long)mbA, (long)nbA, nprow, npcol, commSize);
    }

    /* ================================================================
     * 3. CUDA / NCCL / cuSOLVERMp setup
     * ================================================================ */
    int localRank = getLocalRank();
    cudaError_t cudaStat = cudaSetDevice(localRank);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaFree(0);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    ncclComm_t   ncclComm = createNcclComm(commSize, rank);
    ncclResult_t ncclStat = ncclSuccess;

    cudaStream_t stream = NULL;
    cudaStat = cudaStreamCreate(&stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    cusolverMpHandle_t handle = NULL;
    cusolverStatus_t cusolverStat = cusolverMpCreate(&handle, localRank, stream);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* ================================================================
     * 4. Process grid
     * ================================================================ */
    cusolverMpGridMapping_t gridLayout =
        (opts.grid_layout == 'R') ? CUSOLVERMP_GRID_MAPPING_ROW_MAJOR
                                  : CUSOLVERMP_GRID_MAPPING_COL_MAJOR;

    cusolverMpGrid_t grid = NULL;
    cusolverStat = cusolverMpCreateDeviceGrid(handle, &grid, ncclComm,
                                               nprow, npcol, gridLayout);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* ================================================================
     * 5. Compute local dimensions (ScaLAPACK 2DBC layout)
     * ================================================================ */
    const int rsrc = 0, csrc = 0;

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

    /* A is M-by-N, U is M-by-K, V^T is K-by-N */
    const int64_t localRowsA = cusolverMpNUMROC(m, mbA, myRowRank, rsrc, nprow);
    const int64_t localColsA = cusolverMpNUMROC(n, nbA, myColRank, csrc, npcol);
    const int64_t lldA       = (localRowsA > 0) ? localRowsA : 1;

    const int64_t localRowsU = cusolverMpNUMROC(m, mbA, myRowRank, rsrc, nprow);
    const int64_t localColsU = cusolverMpNUMROC(k, nbA, myColRank, csrc, npcol);
    const int64_t lldU       = (localRowsU > 0) ? localRowsU : 1;

    const int64_t localRowsVT = cusolverMpNUMROC(k, mbA, myRowRank, rsrc, nprow);
    const int64_t localColsVT = cusolverMpNUMROC(n, nbA, myColRank, csrc, npcol);
    const int64_t lldVT       = (localRowsVT > 0) ? localRowsVT : 1;

    /* ================================================================
     * 6. Allocate device memory
     *
     * d_S is REPLICATED: every rank holds a full length-k array of
     * singular values (real type matching computeType). No descriptor.
     * ================================================================ */
    double* d_A    = NULL;
    double* d_U    = NULL;
    double* d_VT   = NULL;
    double* d_S    = NULL;
    int*    d_info = NULL;

    const size_t elemsA  = (size_t)(lldA  * localColsA);
    const size_t elemsU  = (size_t)(lldU  * localColsU);
    const size_t elemsVT = (size_t)(lldVT * localColsVT);
    const size_t bytesA  = ((elemsA  > 0) ? elemsA  : 1) * sizeof(double);
    const size_t bytesU  = ((elemsU  > 0) ? elemsU  : 1) * sizeof(double);
    const size_t bytesVT = ((elemsVT > 0) ? elemsVT : 1) * sizeof(double);

    /* GESVD uses collective cuBLASMp operations, so active distributed
     * matrix pointers must be non-NULL on every rank. Empty-owner ranks use
     * a one-element dummy allocation. */
    cudaStat = cudaMalloc((void**)&d_A, bytesA);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void**)&d_U, bytesU);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void**)&d_VT, bytesVT);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void**)&d_S,    k     * sizeof(double));   /* replicated */
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMalloc((void**)&d_info, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);
    cudaStat = cudaMemset(d_info, 0, sizeof(int));
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* ================================================================
     * 7. Create matrix descriptors for A, U, V^T
     * ================================================================ */
    cusolverMpMatrixDescriptor_t descA  = NULL;
    cusolverMpMatrixDescriptor_t descU  = NULL;
    cusolverMpMatrixDescriptor_t descVT = NULL;

    cusolverStat = cusolverMpCreateMatrixDesc(&descA, grid, CUDA_R_64F,
                                               m, n, mbA, nbA, rsrc, csrc, lldA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpCreateMatrixDesc(&descU, grid, CUDA_R_64F,
                                               m, k, mbA, nbA, rsrc, csrc, lldU);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpCreateMatrixDesc(&descVT, grid, CUDA_R_64F,
                                               k, n, mbA, nbA, rsrc, csrc, lldVT);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* ================================================================
     * 8. Generate test matrix on rank 0 and scatter
     * ================================================================ */
    double* h_A    = NULL;  /* full input (rank 0 only) */
    double* h_Aref = NULL;  /* reference copy for verification */

    if (rank == 0)
    {
        h_A    = (double*)malloc(m * n * sizeof(double));
        h_Aref = (double*)malloc(m * n * sizeof(double));
        SAMPLE_ASSERT(h_A != NULL && h_Aref != NULL);

        generate_random_matrix(m, n, h_A, m);
        memcpy(h_Aref, h_A, m * n * sizeof(double));
    }

    cusolverStat = cusolverMpMatrixScatterH2D(handle, m, n,
                                               (void*)d_A, 1, 1, descA,
                                               0, /* root */
                                               (void*)h_A, m);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    /* ================================================================
     * 9. Configure GESVD (optional descriptor for diagnostics)
     *
     * Passing gesvdDesc=NULL is fine — it uses defaults and the
     * diagnostic output attributes are simply not retrievable. Here we
     * create one to demonstrate readback and enable the optional
     * residual diagnostic.
     * ================================================================ */
    cusolverEigMode_t jobu  = CUSOLVER_EIG_MODE_VECTOR;
    cusolverEigMode_t jobvt = CUSOLVER_EIG_MODE_VECTOR;
    cudaDataType_t    computeType = CUDA_R_64F;

    cusolverMpGesvdDescriptor_t gesvdDesc = NULL;
    cusolverStat = cusolverMpGesvdDescriptorCreate(&gesvdDesc);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    int compute_residual = 1;
    cusolverStat = cusolverMpGesvdDescriptorSetAttribute(
        gesvdDesc,
        CUSOLVERMP_GESVD_DESCRIPTOR_ATTRIBUTE_COMPUTE_RESIDUAL,
        &compute_residual,
        sizeof(compute_residual));
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    /* ================================================================
     * 10. Workspace query
     * ================================================================ */
    size_t d_workSize = 0, h_workSize = 0;
    cusolverStat = cusolverMpGesvd_bufferSize(
        handle, gesvdDesc, jobu, jobvt, m, n,
        (const void*)d_A,  1, 1, descA,
        (const void*)d_S,
        (const void*)d_U,  1, 1, descU,
        (const void*)d_VT, 1, 1, descVT,
        computeType, &d_workSize, &h_workSize);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    if (rank == 0)
    {
        printf("Workspace: device=%zu bytes, host=%zu bytes\n", d_workSize, h_workSize);
    }

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

    /* ================================================================
     * 11. Execute SVD: A = U * diag(S) * V^T
     *
     * NOTE: d_A is overwritten with intermediate state (Up from polar)
     * and is NOT recoverable afterwards.
     * ================================================================ */
    cusolverStat = cusolverMpGesvd(
        handle, gesvdDesc, jobu, jobvt, m, n,
        (void*)d_A,  1, 1, descA,
        (void*)d_S,
        (void*)d_U,  1, 1, descU,
        (void*)d_VT, 1, 1, descVT,
        computeType,
        d_work, d_workSize, h_work, h_workSize,
        d_info);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    int h_info = 0;
    cudaStat = cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0)
    {
        /* d_info decode mirrors the backend-independent cusolverMpGesvd
         * contract documented in cusolverMp.h. */
        const char* msg = "success";
        if      (h_info <  0) msg = "invalid argument";
        else if (h_info == 1) msg = "singular-value computation did not complete";
        else if (h_info == 2) msg = "singular-vector computation did not complete";
        printf("\nResults:\n");
        printf("  d_info = %d (%s)\n", h_info, msg);
    }
    SAMPLE_ASSERT(h_info == 0);

    /* ================================================================
     * 12. Read descriptor diagnostics (rank 0)
     * ================================================================ */
    if (rank == 0)
    {
        double a_nrmF = 0, rcond = 0, resid = 0;
        int64_t num_singular = 0;
        size_t written = 0;
        cusolverStat = cusolverMpGesvdDescriptorGetAttribute(
            gesvdDesc, CUSOLVERMP_GESVD_DESCRIPTOR_ATTRIBUTE_A_NORM_FROBENIUS,
            &a_nrmF, sizeof(a_nrmF), &written);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        SAMPLE_ASSERT(written == sizeof(a_nrmF));
        cusolverStat = cusolverMpGesvdDescriptorGetAttribute(
            gesvdDesc, CUSOLVERMP_GESVD_DESCRIPTOR_ATTRIBUTE_RCOND,
            &rcond, sizeof(rcond), &written);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        SAMPLE_ASSERT(written == sizeof(rcond));
        cusolverStat = cusolverMpGesvdDescriptorGetAttribute(
            gesvdDesc, CUSOLVERMP_GESVD_DESCRIPTOR_ATTRIBUTE_RESIDUAL_FROBENIUS_ESTIMATE,
            &resid, sizeof(resid), &written);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        SAMPLE_ASSERT(written == sizeof(resid));
        cusolverStat = cusolverMpGesvdDescriptorGetAttribute(
            gesvdDesc, CUSOLVERMP_GESVD_DESCRIPTOR_ATTRIBUTE_NUM_SINGULAR_FOUND,
            &num_singular, sizeof(num_singular), &written);
        SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        SAMPLE_ASSERT(written == sizeof(num_singular));

        printf("  ||A||_F             = %.6e\n", a_nrmF);
        printf("  rcond               = %.6e\n", rcond);
        printf("  num_singular_found  = %ld\n", (long)num_singular);
        /* resid is finite when the optional residual diagnostic is enabled. */
        if (resid == resid) printf("  residual_F_estimate = %.6e\n", resid);
    }

    /* ================================================================
     * 13. Gather U, V^T, S to rank 0 and verify
     *
     * d_S is replicated, so a plain D2H copy from any rank works.
     * U and V^T are distributed and need a gather.
     * ================================================================ */
    double* h_U  = NULL;
    double* h_VT = NULL;
    double* h_S  = NULL;

    if (rank == 0)
    {
        h_U  = (double*)malloc(m * k * sizeof(double));
        h_VT = (double*)malloc(k * n * sizeof(double));
        h_S  = (double*)malloc(k * sizeof(double));
        SAMPLE_ASSERT(h_U != NULL && h_VT != NULL && h_S != NULL);
    }

    cusolverStat = cusolverMpMatrixGatherD2H(handle, m, k,
                                              (void*)d_U, 1, 1, descU,
                                              0, (void*)h_U, m);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpMatrixGatherD2H(handle, k, n,
                                              (void*)d_VT, 1, 1, descVT,
                                              0, (void*)h_VT, k);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    if (rank == 0)
    {
        cudaStat = cudaMemcpy(h_S, d_S, k * sizeof(double), cudaMemcpyDeviceToHost);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }

    cudaStat = cudaStreamSynchronize(stream);
    SAMPLE_ASSERT(cudaStat == cudaSuccess);

    if (rank == 0)
    {
        /* Reconstruction error: ||A - U * diag(S) * V^T||_F / ||A||_F */
        double a_norm2 = 0.0, err2 = 0.0;
        for (int64_t j = 0; j < n; j++)
        {
            for (int64_t i = 0; i < m; i++)
            {
                double recon = 0.0;
                for (int64_t l = 0; l < k; l++)
                {
                    recon += h_U[i + l * m] * h_S[l] * h_VT[l + j * k];
                }
                double aij = h_Aref[i + j * m];
                double diff = aij - recon;
                a_norm2 += aij * aij;
                err2    += diff * diff;
            }
        }
        const double a_norm = sqrt(a_norm2);
        const double err    = sqrt(err2);
        const double rel    = (a_norm > 0) ? err / a_norm : err;

        /* Singular values monotone non-increasing? */
        int monotone = 1;
        for (int64_t l = 1; l < k; l++)
            if (h_S[l] > h_S[l - 1] + 1e-14) { monotone = 0; break; }

        printf("\nVerification:\n");
        printf("  S[0]   = %.6e\n", h_S[0]);
        printf("  S[k-1] = %.6e\n", h_S[k - 1]);
        printf("  S monotone non-increasing: %s\n", monotone ? "YES" : "NO");
        printf("  ||A - U*diag(S)*V^T||_F / ||A||_F = %.6e\n", rel);
        const double tol = 100.0 * (double)((m > n) ? m : n) * 2.22e-16;
        sample_ok = sample_ok && (rel < tol && monotone);
        printf("  Pass (tol=%.2e): %s\n", tol, sample_ok ? "YES" : "NO");
    }

    /* ================================================================
     * 14. Cleanup
     * ================================================================ */
    if (rank == 0)
    {
        if (h_A)    free(h_A);
        if (h_Aref) free(h_Aref);
        if (h_U)    free(h_U);
        if (h_VT)   free(h_VT);
        if (h_S)    free(h_S);
    }

    if (d_work)
    {
        cudaStat = cudaFree(d_work);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
        d_work = NULL;
    }
    if (h_work) free(h_work);

    cusolverStat = cusolverMpGesvdDescriptorDestroy(gesvdDesc);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyMatrixDesc(descA);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpDestroyMatrixDesc(descU);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);
    cusolverStat = cusolverMpDestroyMatrixDesc(descVT);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    cusolverStat = cusolverMpDestroyGrid(grid);
    SAMPLE_ASSERT(cusolverStat == CUSOLVER_STATUS_SUCCESS);

    if (d_A)
    {
        cudaStat = cudaFree(d_A);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }
    if (d_U)
    {
        cudaStat = cudaFree(d_U);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }
    if (d_VT)
    {
        cudaStat = cudaFree(d_VT);
        SAMPLE_ASSERT(cudaStat == cudaSuccess);
    }
    if (d_S)
    {
        cudaStat = cudaFree(d_S);
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

    MPI_Barrier(MPI_COMM_WORLD);
    sample_ok = sample_all_ranks_succeeded(sample_ok);
    MPI_Finalize();

    if (rank == 0)
    {
        printf("%s\n", sample_ok ? "[SUCCEEDED]" : "[FAILED]");
    }

    return sample_ok ? 0 : 1;
}
