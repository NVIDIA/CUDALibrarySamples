/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#ifdef USE_CAL_MPI
#include <cal_mpi.h>
#endif

#include <cublasmp.h>

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

    cal_comm_t cal_comm = nullptr;
#ifdef USE_CAL_MPI
    CAL_CHECK(cal_comm_create_mpi(MPI_COMM_WORLD, rank, nranks, local_device, &cal_comm));
#else
    cal_comm_create_params_t params;
    params.allgather = allgather;
    params.req_test = request_test;
    params.req_free = request_free;
    params.data = (void*)(MPI_COMM_WORLD);
    params.rank = rank;
    params.nranks = nranks;
    params.local_device = local_device;
    CAL_CHECK(cal_comm_create(params, &cal_comm));
#endif

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasMpHandle_t handle = nullptr;
    CUBLASMP_CHECK(cublasMpCreate(&handle, stream));

    cublasMpGrid_t grid = nullptr;

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;

    double* d_A = nullptr;
    double* d_B = nullptr;

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

    CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b * sizeof(double), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), llda * loc_n_a * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), lldb * loc_n_b * sizeof(double), cudaMemcpyHostToDevice, stream));

    CUBLASMP_CHECK(cublasMpGridCreate(
        nprow,
        npcol,
        opts.grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        cal_comm,
        &grid));

    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(global_m_a, global_n_a, mbA, nbA, 0, 0, llda, CUDA_R_64F, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, CUDA_R_64F, grid, &descB));

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
        CUBLAS_COMPUTE_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));

    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CAL_CHECK(cal_stream_sync(cal_comm, stream));
    CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    const double begin = MPI_Wtime();

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
        CUBLAS_COMPUTE_64F,
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost));

    CAL_CHECK(cal_stream_sync(cal_comm, stream));
    CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    const double end = MPI_Wtime();

    if (rank == 0)
    {
        printf(
            "Duration: %lf GFlops: %lf\n",
            end - begin,
            ((((0.5 * m * (m - 1)) + ((0.5 * m * (m + 1)))) * n) * 1e-9) / (end - begin));
    }

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));

    CUBLASMP_CHECK(cublasMpGridDestroy(grid));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_work, stream));

    CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    CAL_CHECK(cal_comm_destroy(cal_comm));

    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    if (rank == 0)
    {
        printf("[SUCCEEDED]\n");
    }

    return 0;
};