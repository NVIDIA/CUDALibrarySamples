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
#include <cublasmp.h>
#include <cuda_fp8.h>
#include <math.h>
#include <mpi.h>
#include <nvshmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#include "helpers.h"
#include "matrix_generator.hxx"

int main(int argc, char* argv[])
{
    using input_t = __half;
    using output_t = __half;
    using compute_t = float;
    const cudaDataType_t cuda_input_type = CUDA_R_16F;
    const cudaDataType_t cuda_output_type = CUDA_R_16F;
    const cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F;
    const cublasOperation_t transA = CUBLAS_OP_T;
    const cublasOperation_t transB = CUBLAS_OP_N;

    MPI_Init(nullptr, nullptr);

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

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    cublasMpMatrixDescriptor_t descC = nullptr;

    cublasMpMatmulDescriptor_t matmulDesc = nullptr;

    input_t* d_A = nullptr;
    input_t* d_B = nullptr;
    output_t* d_C = nullptr;

    void* d_work = nullptr;

    compute_t alpha = 1.0;
    compute_t beta = 0.0;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    cublasMpGrid_t grid_col_major = nullptr;
    cublasMpGrid_t grid_row_major = nullptr;

    CUBLASMP_CHECK(cublasMpGridCreate(nranks, 1, CUBLASMP_GRID_LAYOUT_COL_MAJOR, comm, &grid_col_major));
    CUBLASMP_CHECK(cublasMpGridCreate(1, nranks, CUBLASMP_GRID_LAYOUT_ROW_MAJOR, comm, &grid_row_major));

    const int64_t m = 64;
    const int64_t n = 64 * nranks;
    const int64_t k = 64 * nranks;

    const int64_t llda = k / nranks;
    const int64_t lldb = k / nranks;
    const int64_t lldc = m;

    std::vector<input_t> h_A(llda * m, input_t(0));
    std::vector<input_t> h_B(lldb * n, input_t(0));
    std::vector<output_t> h_C(lldc * n, output_t(0));

    generate_random_matrix(k, m, h_A.data(), llda, m, 1, 1, llda, nranks, 1, rank, 1);
    generate_random_matrix(k, n, h_B.data(), lldb, n, 1, 1, lldb, nranks, 1, rank, 1);
    generate_random_matrix(m, n * nranks, h_C.data(), m, n, 1, 1, lldc, 1, nranks, 1, rank);

    CUDA_CHECK(cudaMalloc((void**)&d_A, llda * m * sizeof(input_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, lldb * n * sizeof(input_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, lldc * n * sizeof(output_t)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), llda * m * sizeof(input_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), lldb * n * sizeof(input_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, h_C.data(), lldc * n * sizeof(output_t), cudaMemcpyHostToDevice, stream));

    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(k, m, llda, m, 0, 0, llda, cuda_input_type, grid_col_major, &descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(k, n, lldb, n, 0, 0, lldb, cuda_input_type, grid_col_major, &descB));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(m, n * nranks, lldc, n, 0, 0, lldc, cuda_output_type, grid_row_major, &descC));

    const cublasMpMatmulAlgoType_t algoType = CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_MULTICAST;
    const cublasMpMatmulEpilogue_t epilogue = CUBLASMP_MATMUL_EPILOGUE_ALLREDUCE;

    CUBLASMP_CHECK(cublasMpMatmulDescriptorCreate(&matmulDesc, cublas_compute_type));
    CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
        matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &transA, sizeof(transA)));
    CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
        matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &transB, sizeof(transB)));
    CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
        matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE, &algoType, sizeof(algoType)));
    CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
        matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_EPILOGUE, &epilogue, sizeof(epilogue)));

    CUBLASMP_CHECK(cublasMpMatmul_bufferSize(
        handle,
        matmulDesc,
        m,
        n,
        k,
        &alpha,
        d_A,
        1,
        1,
        descA,
        d_B,
        1,
        1,
        descB,
        &beta,
        nullptr,
        1,
        1,
        descC,
        d_C,
        1,
        1,
        descC,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    // NVSHMEM is initialized as part of cublasMpGridCreate.
    d_work = nvshmem_malloc(workspaceInBytesOnDevice);

    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double begin = MPI_Wtime();

    CUBLASMP_CHECK(cublasMpMatmul(
        handle,
        matmulDesc,
        m,
        n,
        k,
        &alpha,
        d_A,
        1,
        1,
        descA,
        d_B,
        1,
        1,
        descB,
        &beta,
        nullptr,
        1,
        1,
        descC,
        d_C,
        1,
        1,
        descC,
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    const double end = MPI_Wtime();

    if (rank == 0)
    {
        printf(
            "Matmul + Allreduce: %lf (s) %lf (GFlops)\n",
            end - begin,
            (((2 * m * n * k) + (nranks * m * n)) * 1e-9) / (end - begin));
    }

    CUBLASMP_CHECK(cublasMpMatmulDescriptorDestroy(matmulDesc));

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descC));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    nvshmem_free(d_work);

    CUBLASMP_CHECK(cublasMpGridDestroy(grid_col_major));
    CUBLASMP_CHECK(cublasMpGridDestroy(grid_row_major));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

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