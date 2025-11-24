/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                     .verbose = false };

    opts.parse(argc, argv);
    opts.validate();
    opts.print();

    MPI_Init(nullptr, nullptr);

    const int64_t n = opts.n;
    const int64_t k = opts.k;
    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ic = opts.ic;
    const int64_t jc = opts.jc;
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbC = opts.mbC;
    const int64_t nbC = opts.nbC;

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
    cublasMpMatrixDescriptor_t descC = nullptr;

    double* d_A = nullptr;
    double* d_C = nullptr;

    double* d_work = nullptr;

    double alpha = 1.0;
    double beta = 1.0;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    const int64_t global_m_a = (ia - 1) + n;
    const int64_t global_n_a = (ja - 1) + k;
    const int64_t global_m_c = (ic - 1) + n;
    const int64_t global_n_c = (jc - 1) + n;

    const int64_t llda = cublasMpNumroc(global_m_a, mbA, myprow, 0, nprow);
    const int64_t loc_n_a = cublasMpNumroc(global_n_a, nbA, mypcol, 0, npcol);

    const int64_t lldc = cublasMpNumroc(global_m_c, mbC, myprow, 0, nprow);
    const int64_t loc_n_c = cublasMpNumroc(global_n_c, nbC, mypcol, 0, npcol);

    std::vector<double> h_A(llda * loc_n_a, 0);
    std::vector<double> h_C(lldc * loc_n_c, 0);

    generate_random_matrix(n, k, h_A.data(), mbA, nbA, ia, ja, llda, nprow, npcol, myprow, mypcol);
    generate_random_matrix(n, n, h_C.data(), mbC, nbC, ic, jc, lldc, nprow, npcol, myprow, mypcol);

    CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c * sizeof(double), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), llda * loc_n_a * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, h_C.data(), lldc * loc_n_c * sizeof(double), cudaMemcpyHostToDevice, stream));

    CUBLASMP_CHECK(cublasMpGridCreate(
        nprow,
        npcol,
        opts.grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        comm,
        &grid));

    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(global_m_a, global_n_a, mbA, nbA, 0, 0, llda, CUDA_R_64F, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, CUDA_R_64F, grid, &descC));

    CUBLASMP_CHECK(cublasMpSyrk_bufferSize(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        n,
        k,
        &alpha,
        d_A,
        ia,
        ja,
        descA,
        &beta,
        d_C,
        ic,
        jc,
        descC,
        CUBLAS_COMPUTE_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));

    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double begin = MPI_Wtime();

    CUBLASMP_CHECK(cublasMpSyrk(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        n,
        k,
        &alpha,
        d_A,
        ia,
        ja,
        descA,
        &beta,
        d_C,
        ic,
        jc,
        descC,
        CUBLAS_COMPUTE_64F,
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Duration: %lf GFlops: %lf\n", end - begin, (n * n * k * 1e-9) / (end - begin));
    }

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descC));

    CUBLASMP_CHECK(cublasMpGridDestroy(grid));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));
    CUDA_CHECK(cudaFreeAsync(d_work, stream));

    NCCL_CHECK(ncclCommFinalize(comm));
    NCCL_CHECK(ncclCommDestroy(comm));

    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0)
    {
        printf("[SUCCEEDED]\n");
    }

    return 0;
};