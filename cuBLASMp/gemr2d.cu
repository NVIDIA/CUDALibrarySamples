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

#include <cublasmp.h>
#include <mpi.h>
#include <stdio.h>

#include <vector>

#include "helpers.h"
#include "matrix_generator.hxx"

template <typename T>
int run_gemr2d(const Options& opts)
{
    int rank, nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    cublasMpGrid_t grid_pq = nullptr;
    cublasMpGrid_t grid_qp = nullptr;

    const cublasMpGridLayout_t grid_layout =
        (opts.grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR);

    CUBLASMP_CHECK(cublasMpGridCreate(opts.p, opts.q, grid_layout, comm, &grid_pq));
    CUBLASMP_CHECK(cublasMpGridCreate(opts.q, opts.p, grid_layout, comm, &grid_qp));

    const int64_t m = opts.m;
    const int64_t n = opts.n;

    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbB = opts.mbB;
    const int64_t nbB = opts.nbB;

    const int64_t ia = opts.ia;
    const int64_t ja = opts.ja;
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;

    const int nprowA = opts.p;
    const int npcolA = opts.q;
    const int nprowB = opts.q;
    const int npcolB = opts.p;

    const int64_t rsrcA = 0;
    const int64_t csrcA = 0;
    const int64_t rsrcB = 0;
    const int64_t csrcB = 0;

    const int myprowA = (opts.grid_layout == 'c' ? rank % nprowA : rank / npcolA);
    const int mypcolA = (opts.grid_layout == 'c' ? rank / nprowA : rank % npcolA);
    const int myprowB = (opts.grid_layout == 'c' ? rank % nprowB : rank / npcolB);
    const int mypcolB = (opts.grid_layout == 'c' ? rank / nprowB : rank % npcolB);

    const int64_t loc_a_m = cublasMpNumroc(m, mbA, myprowA, rsrcA, nprowA);
    const int64_t loc_a_n = cublasMpNumroc(n, nbA, mypcolA, csrcA, npcolA);
    const int64_t loc_b_m = cublasMpNumroc(m, mbB, myprowB, rsrcB, nprowB);
    const int64_t loc_b_n = cublasMpNumroc(n, nbB, mypcolB, csrcB, npcolB);

    std::vector<T> h_A(loc_a_m * loc_a_n, T(0));
    std::vector<T> h_B(loc_b_m * loc_b_n, T(0));

    generate_random_matrix(m, n, h_A.data(), mbA, nbA, ia, ja, loc_a_m, nprowA, npcolA, myprowA, mypcolA);
    generate_random_matrix(m, n, h_B.data(), mbB, nbB, ib, jb, loc_b_m, nprowB, npcolB, myprowB, mypcolB);

    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_work = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_A, loc_a_m * loc_a_n * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, loc_b_m * loc_b_n * sizeof(T)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), loc_a_m * loc_a_n * sizeof(T), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), loc_b_m * loc_b_n * sizeof(T), cudaMemcpyHostToDevice, stream));

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;

    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        m, n, mbA, nbA, rsrcA, csrcA, loc_a_m, CudaTypeTraits<T>::typeEnum, grid_pq, &descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        m, n, mbB, nbB, rsrcB, csrcB, loc_b_m, CudaTypeTraits<T>::typeEnum, grid_qp, &descB));

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    CUBLASMP_CHECK(cublasMpGemr2D_bufferSize(
        handle,
        m,
        n,
        d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost,
        comm));

    CUDA_CHECK(cudaMalloc((void**)&d_work, workspaceInBytesOnDevice));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CUBLASMP_CHECK(cublasMpGemr2D(
        handle,
        m,
        n,
        d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost,
        comm));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_work));

    CUBLASMP_CHECK(cublasMpGridDestroy(grid_pq));
    CUBLASMP_CHECK(cublasMpGridDestroy(grid_qp));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

    NCCL_CHECK(ncclCommFinalize(comm));
    NCCL_CHECK(ncclCommDestroy(comm));

    CUDA_CHECK(cudaStreamDestroy(stream));

    if (rank == 0)
    {
        printf("[SUCCEEDED]\n");
    }

    return 0;
};

int main(int argc, char** argv)
{
    Options opts {
        .m = 1024,
        .n = 1024,
        .mbA = 64,
        .nbA = 64,
        .mbB = 128,
        .nbB = 128,
        .ia = 1,
        .ja = 1,
        .ib = 1,
        .jb = 1,
        .p = 2,
        .q = 1,
        .grid_layout = 'c',
        .typeA = CUDA_R_32F,
        .typeB = CUDA_R_32F,
    };

    opts.parse(argc, argv);

    MPI_Init(&argc, &argv);

    if (opts.typeA == CUDA_R_32F && opts.typeB == CUDA_R_32F)
    {
        run_gemr2d<float>(opts);
    }
    else if (opts.typeA == CUDA_R_64F && opts.typeB == CUDA_R_64F)
    {
        run_gemr2d<double>(opts);
    }
    else
    {
        throw std::runtime_error("The gemr2d sample doesn't support the given datatype combination");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}