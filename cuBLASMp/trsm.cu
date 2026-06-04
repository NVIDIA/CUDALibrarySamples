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

#include <assert.h>
#include <cublasmp.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#include "helpers.h"
#include "matrix_generator.hxx"

template <typename T>
static bool check_trsm_result(
    cublasMpHandle_t mp_handle,
    ncclComm_t comm,
    cudaStream_t stream,
    int rank,
    cublasMpGrid_t grid,
    int nprow,
    int npcol,
    int myprow,
    int mypcol,
    cublasSideMode_t side,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    cublasDiagType_t diag,
    int64_t m,
    int64_t n,
    const T* alpha,
    T* d_A,
    int64_t ia,
    int64_t ja,
    cublasMpMatrixDescriptor_t descA,
    T* d_B_ref,
    T* d_B,
    int64_t ib,
    int64_t jb,
    cublasMpMatrixDescriptor_t descB)
{
    static_assert(std::is_same_v<T, double>, "trsm sample reference check supports double only");

    const int64_t a_size = side == CUBLAS_SIDE_LEFT ? m : n;
    T* full_A = nullptr;
    T* full_B_ref = nullptr;
    T* full_B_result = nullptr;
    int64_t full_A_lld = 0;
    int64_t full_B_ref_lld = 0;
    int64_t full_B_result_lld = 0;

    gather_matrix(
        mp_handle,
        comm,
        stream,
        a_size,
        a_size,
        d_A,
        ia,
        ja,
        descA,
        grid,
        nprow,
        npcol,
        myprow,
        mypcol,
        &full_A,
        &full_A_lld);
    gather_matrix(
        mp_handle,
        comm,
        stream,
        m,
        n,
        d_B_ref,
        ib,
        jb,
        descB,
        grid,
        nprow,
        npcol,
        myprow,
        mypcol,
        &full_B_ref,
        &full_B_ref_lld);
    gather_matrix(
        mp_handle,
        comm,
        stream,
        m,
        n,
        d_B,
        ib,
        jb,
        descB,
        grid,
        nprow,
        npcol,
        myprow,
        mypcol,
        &full_B_result,
        &full_B_result_lld);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool passed = true;
    if (rank == 0)
    {
        cublasHandle_t cublas_handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

        if constexpr (std::is_same_v<T, double>)
        {
            CUBLAS_CHECK(cublasDtrsm(
                cublas_handle,
                side,
                uplo,
                trans,
                diag,
                static_cast<int>(m),
                static_cast<int>(n),
                alpha,
                full_A,
                static_cast<int>(full_A_lld),
                full_B_ref,
                static_cast<int>(full_B_ref_lld)));
        }

        passed = allclose_device("trsm", full_B_result, full_B_result_lld, full_B_ref, full_B_ref_lld, m, n, stream);
        CUBLAS_CHECK(cublasDestroy(cublas_handle));
    }

    CUDA_CHECK(cudaFree(full_A));
    CUDA_CHECK(cudaFree(full_B_ref));
    CUDA_CHECK(cudaFree(full_B_result));

    int passed_int = passed ? 1 : 0;
    MPI_CHECK(MPI_Bcast(&passed_int, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return passed_int != 0;
}

int main(int argc, char* argv[])
{
    Options opts = { .m = 10,
                     .n = 10,
                     .k = 10,
                     .mbA = 2,
                     .nbA = 2,
                     .mbB = 2,
                     .nbB = 2,
                     .mbC = 2,
                     .nbC = 2,
                     .ia = 3,
                     .ja = 3,
                     .ib = 3,
                     .jb = 1,
                     .ic = 1,
                     .jc = 1,
                     .p = 2,
                     .q = 1,
                     .grid_layout = 'c',
                     .verbose = false,
                     .cycles = 10,
                     .warmup = 5 };

    opts.parse(argc, argv);
    opts.validate();

    MPI_Init(nullptr, nullptr);

    const int64_t m = opts.m;
    const int64_t n = opts.n;
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbB = opts.mbB;
    const int64_t nbB = opts.nbB;

    const int nprow = opts.p;
    const int npcol = opts.q;

    int rank, nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int myprow = (opts.grid_layout == 'c' ? rank % nprow : rank / npcol);
    const int mypcol = (opts.grid_layout == 'c' ? rank / nprow : rank % npcol);

    const int local_device = getLocalDevice();
    CUDA_CHECK(cudaSetDevice(local_device));
    CUDA_CHECK(cudaFree(nullptr));

    ncclUniqueId id;

    if (rank == 0)
    {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }

    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasMpHandle_t handle = nullptr;
    CUBLASMP_CHECK(cublasMpCreate(&handle, stream));

    cublasMpGrid_t grid = nullptr;

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_B_ref = nullptr;

    double* d_work = nullptr;

    double alpha = 1.0;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    const int64_t global_m_a = (ia - 1) + m;
    const int64_t global_n_a = (ja - 1) + m;
    const int64_t global_m_b = (ib - 1) + m;
    const int64_t global_n_b = (jb - 1) + n;

    const int64_t llda = cublasMpNumroc(global_m_a, mbA, myprow, 0, nprow);
    const int64_t loc_n_a = cublasMpNumroc(global_n_a, nbA, mypcol, 0, npcol);

    const int64_t lldb = cublasMpNumroc(global_m_b, mbB, myprow, 0, nprow);
    const int64_t loc_n_b = cublasMpNumroc(global_n_b, nbB, mypcol, 0, npcol);

    std::vector<double> h_A(llda * loc_n_a, 0);
    std::vector<double> h_B(lldb * loc_n_b, 0);

    generate_diag_matrix(m, m, h_A.data(), mbA, nbA, ia, ja, llda, nprow, npcol, myprow, mypcol);
    generate_random_matrix(m, n, h_B.data(), mbB, nbB, ib, jb, lldb, nprow, npcol, myprow, mypcol);

    CUDA_CHECK(cudaMalloc(&d_A, llda * loc_n_a * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, lldb * loc_n_b * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B_ref, lldb * loc_n_b * sizeof(double)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), llda * loc_n_a * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), lldb * loc_n_b * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_ref, h_B.data(), lldb * loc_n_b * sizeof(double), cudaMemcpyHostToDevice, stream));

    CUBLASMP_CHECK(cublasMpGridCreate(
        nprow,
        npcol,
        opts.grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        comm,
        &grid));

    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(global_m_a, global_n_a, mbA, nbA, 0, 0, llda, CUDA_R_64F, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, CUDA_R_64F, grid, &descB));

    cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_64F;

#if CUBLAS_VERSION >= 130002
    cublasMpEmulationStrategy_t emulationStrategy = string_to_emulation_strategy(opts.emulationStrategy);
    if (emulationStrategy != cublasMpEmulationStrategy_t(-1))
    {
        CUBLASMP_CHECK(cublasMpSetEmulationStrategy(handle, emulationStrategy));
        cublas_compute_type = CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
    }
#endif

    CUBLASMP_CHECK(cublasMpTrsm_bufferSize(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        m,
        n,
        &alpha,
        d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        cublas_compute_type,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(&d_work, workspaceInBytesOnDevice));

    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < opts.warmup + opts.cycles; i++)
    {
        if (i == opts.warmup)
        {
            CUDA_CHECK(cudaEventRecord(start, stream));
        }

        CUBLASMP_CHECK(cublasMpTrsm(
            handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            m,
            n,
            &alpha,
            d_A,
            ia,
            ja,
            descA,
            d_B,
            ib,
            jb,
            descB,
            cublas_compute_type,
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double elapsed = (elapsed_ms / 1000.0) / opts.cycles;

    if (rank == 0)
    {
        printf(
            "Duration: %lf GFlops: %lf\n",
            elapsed,
            ((((0.5 * m * (m - 1)) + ((0.5 * m * (m + 1)))) * n) * 1e-9) / elapsed);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Reset B to the clean right-hand side before the single verification solve; TRSM mutates B in place.
    CUDA_CHECK(cudaMemcpyAsync(d_B, d_B_ref, lldb * loc_n_b * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUBLASMP_CHECK(cublasMpTrsm(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        m,
        n,
        &alpha,
        d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        cublas_compute_type,
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost));

    const bool passed = !opts.check_result || check_trsm_result(
                                                  handle,
                                                  comm,
                                                  stream,
                                                  rank,
                                                  grid,
                                                  nprow,
                                                  npcol,
                                                  myprow,
                                                  mypcol,
                                                  CUBLAS_SIDE_LEFT,
                                                  CUBLAS_FILL_MODE_LOWER,
                                                  CUBLAS_OP_N,
                                                  CUBLAS_DIAG_NON_UNIT,
                                                  m,
                                                  n,
                                                  &alpha,
                                                  d_A,
                                                  ia,
                                                  ja,
                                                  descA,
                                                  d_B_ref,
                                                  d_B,
                                                  ib,
                                                  jb,
                                                  descB);

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));

    CUBLASMP_CHECK(cublasMpGridDestroy(grid));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_B_ref));
    CUDA_CHECK(cudaFree(d_work));

    NCCL_CHECK(ncclCommFinalize(comm));
    NCCL_CHECK(ncclCommDestroy(comm));

    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0)
    {
        printf(passed ? "[SUCCEEDED]\n" : "[FAILED]\n");
    }

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
};
