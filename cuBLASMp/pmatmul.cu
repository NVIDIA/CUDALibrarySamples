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
#include <cuda_fp8.h>
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
#include <nvshmem.h>

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

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    cublasMpMatrixDescriptor_t descC = nullptr;

    cublasMpMatmulDescriptor_t matmulDesc = nullptr;

    output_t* d_X1 = nullptr;
    output_t* d_X2 = nullptr;

    void* d_work = nullptr;

    compute_t alpha = 1.0;
    compute_t beta = 0.0;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    cublasMpGrid_t grid_col_major = nullptr;
    cublasMpGrid_t grid_row_major = nullptr;

    CUBLASMP_CHECK(cublasMpGridCreate(nranks, 1, CUBLASMP_GRID_LAYOUT_COL_MAJOR, cal_comm, &grid_col_major));
    CUBLASMP_CHECK(cublasMpGridCreate(1, nranks, CUBLASMP_GRID_LAYOUT_ROW_MAJOR, cal_comm, &grid_row_major));

    const bool ta = (transA != CUBLAS_OP_N);

    // AG + Matmul
    {
        const int64_t m = 64 * nranks;
        const int64_t n = 64 * nranks;
        const int64_t k = 64;

        const int64_t loc_a_m = ta ? k : m / nranks;
        const int64_t loc_a_n = ta ? m / nranks : k;
        const int64_t loc_b_m = k;
        const int64_t loc_b_n = n / nranks;
        const int64_t loc_c_m = m / nranks;
        const int64_t loc_c_n = n / nranks;

        std::vector<input_t> h_X0(loc_a_m * loc_a_n, input_t(0));
        std::vector<input_t> h_W0(loc_b_m * loc_b_n, input_t(0));
        std::vector<output_t> h_X1(loc_c_m * loc_c_n * nranks, output_t(0));

        generate_random_matrix(
            ta ? k : m,
            ta ? m : k,
            h_X0.data(),
            loc_a_m,
            loc_a_n,
            1,
            1,
            loc_a_m,
            ta ? 1 : nranks,
            ta ? nranks : 1,
            ta ? 1 : rank,
            ta ? rank : 1);
        generate_random_matrix(k, n, h_W0.data(), loc_b_m, loc_b_n, 1, 1, loc_b_m, 1, nranks, 1, rank);
        generate_random_matrix(m, n, h_X1.data(), loc_c_m, loc_c_n, 1, 1, loc_c_m, nranks, 1, rank, 1);

        input_t* d_X0 = nullptr;
        input_t* d_W0 = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_X0, loc_a_m * loc_a_n * sizeof(input_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_W0, loc_b_m * loc_b_n * sizeof(input_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_X1, loc_c_m * loc_c_n * nranks * sizeof(output_t)));

        CUDA_CHECK(
            cudaMemcpyAsync(d_X0, h_X0.data(), loc_a_m * loc_a_n * sizeof(input_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(d_W0, h_W0.data(), loc_b_m * loc_b_n * sizeof(input_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_X1, h_X1.data(), loc_c_m * loc_c_n * nranks * sizeof(output_t), cudaMemcpyHostToDevice, stream));

        CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
            ta ? k : m,
            ta ? m : k,
            loc_a_m,
            loc_a_n,
            0,
            0,
            loc_a_m,
            cuda_input_type,
            ta ? grid_row_major : grid_col_major,
            &descA));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
            k, n, loc_b_m, loc_b_n, 0, 0, loc_b_m, cuda_input_type, grid_row_major, &descB));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
            m, n, loc_c_m, loc_c_n, 0, 0, loc_c_m, cuda_output_type, grid_col_major, &descC));

        const cublasMpMatmulAlgoType_t algoType = CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_P2P;

        CUBLASMP_CHECK(cublasMpMatmulDescriptorCreate(&matmulDesc, cublas_compute_type));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &transA, sizeof(transA)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &transB, sizeof(transB)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE, &algoType, sizeof(algoType)));

        CUBLASMP_CHECK(cublasMpMatmul_bufferSize(
            handle,
            matmulDesc,
            m,
            n,
            k,
            &alpha,
            d_X0,
            1,
            1,
            descA,
            d_W0,
            1,
            1,
            descB,
            &beta,
            nullptr,
            1,
            1,
            descC,
            d_X1,
            1,
            1,
            descC,
            &workspaceInBytesOnDevice,
            &workspaceInBytesOnHost));

        // NVSHMEM is initialized as part of cublasMpGridCreate.
        d_work = nvshmem_malloc(workspaceInBytesOnDevice);

        std::vector<int8_t> h_work(workspaceInBytesOnHost);

        CAL_CHECK(cal_stream_sync(cal_comm, stream));
        CAL_CHECK(cal_comm_barrier(cal_comm, stream));

        const double begin = MPI_Wtime();

        CUBLASMP_CHECK(cublasMpMatmul(
            handle,
            matmulDesc,
            m,
            n,
            k,
            &alpha,
            d_X0,
            1,
            1,
            descA,
            d_W0,
            1,
            1,
            descB,
            &beta,
            nullptr,
            1,
            1,
            descC,
            d_X1,
            1,
            1,
            descC,
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost));

        CAL_CHECK(cal_stream_sync(cal_comm, stream));
        CAL_CHECK(cal_comm_barrier(cal_comm, stream));

        const double end = MPI_Wtime();

        if (rank == 0)
        {
            printf("AG + Matmul: %lf (s) %lf (GFlops)\n", end - begin, (2 * m * n * k * 1e-9) / (end - begin));
        }

        CUBLASMP_CHECK(cublasMpMatmulDescriptorDestroy(matmulDesc));

        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descC));

        CUDA_CHECK(cudaFree(d_X0));
        CUDA_CHECK(cudaFree(d_W0));
        nvshmem_free(d_work);
    }

    // Matmul + RS
    {
        const int64_t m = 64;
        const int64_t n = 64 * nranks;
        const int64_t k = 64 * nranks;

        const int64_t loc_a_m = ta ? k / nranks : m;
        const int64_t loc_a_n = ta ? m : k / nranks;
        const int64_t loc_b_m = k / nranks;
        const int64_t loc_b_n = n / nranks;
        const int64_t loc_c_m = m;
        const int64_t loc_c_n = n / nranks;

        std::vector<input_t> h_W1(loc_a_m * loc_a_n, input_t(0));
        std::vector<output_t> h_X2(loc_c_m * loc_c_n, output_t(0));

        generate_random_matrix(
            ta ? k : m,
            ta ? m : k,
            h_W1.data(),
            loc_a_m,
            loc_a_n,
            1,
            1,
            loc_a_m,
            ta ? nranks : 1,
            ta ? 1 : nranks,
            ta ? rank : 1,
            ta ? 1 : rank);
        generate_random_matrix(m, n, h_X2.data(), loc_c_m, loc_c_n, 1, 1, loc_c_m, 1, nranks, 1, rank);

        input_t* d_W1 = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_W1, loc_a_m * loc_a_n * sizeof(input_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_X2, loc_c_m * loc_c_n * sizeof(output_t)));

        CUDA_CHECK(
            cudaMemcpyAsync(d_W1, h_W1.data(), loc_a_m * loc_a_n * sizeof(input_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(d_X2, h_X2.data(), loc_c_m * loc_c_n * sizeof(output_t), cudaMemcpyHostToDevice, stream));

        CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
            ta ? k : m,
            ta ? m : k,
            loc_a_m,
            loc_a_n,
            0,
            0,
            loc_a_m,
            cuda_input_type,
            ta ? grid_col_major : grid_row_major,
            &descA));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
            k, n, loc_b_m, loc_b_n, 0, 0, loc_b_m, cuda_input_type, grid_col_major, &descB));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
            m, n, loc_c_m, loc_c_n, 0, 0, loc_c_m, cuda_output_type, grid_row_major, &descC));

        const cublasMpMatmulAlgoType_t algoType = CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_P2P;

        CUBLASMP_CHECK(cublasMpMatmulDescriptorCreate(&matmulDesc, cublas_compute_type));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &transA, sizeof(transA)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &transB, sizeof(transB)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorAttributeSet(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE, &algoType, sizeof(algoType)));

        CUBLASMP_CHECK(cublasMpMatmul_bufferSize(
            handle,
            matmulDesc,
            m,
            n,
            k,
            &alpha,
            d_W1,
            1,
            1,
            descA,
            d_X1,
            1,
            1,
            descB,
            &beta,
            nullptr,
            1,
            1,
            descC,
            d_X2,
            1,
            1,
            descC,
            &workspaceInBytesOnDevice,
            &workspaceInBytesOnHost));

        d_work = nvshmem_malloc(workspaceInBytesOnDevice);

        std::vector<int8_t> h_work(workspaceInBytesOnHost);

        CAL_CHECK(cal_stream_sync(cal_comm, stream));
        CAL_CHECK(cal_comm_barrier(cal_comm, stream));

        const double begin = MPI_Wtime();

        CUBLASMP_CHECK(cublasMpMatmul(
            handle,
            matmulDesc,
            m,
            n,
            k,
            &alpha,
            d_W1,
            1,
            1,
            descA,
            d_X1,
            1,
            1,
            descB,
            &beta,
            nullptr,
            1,
            1,
            descC,
            d_X2,
            1,
            1,
            descC,
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost));

        CAL_CHECK(cal_stream_sync(cal_comm, stream));
        CAL_CHECK(cal_comm_barrier(cal_comm, stream));

        const double end = MPI_Wtime();

        if (rank == 0)
        {
            printf("Matmul + RS: %lf (s) %lf (GFlops)\n", end - begin, (2 * m * n * k * 1e-9) / (end - begin));
        }

        CUBLASMP_CHECK(cublasMpMatmulDescriptorDestroy(matmulDesc));

        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descC));

        CUDA_CHECK(cudaFree(d_X1));
        CUDA_CHECK(cudaFree(d_W1));
        CUDA_CHECK(cudaFree(d_X2));
        nvshmem_free(d_work);
    }

    CUBLASMP_CHECK(cublasMpGridDestroy(grid_col_major));
    CUBLASMP_CHECK(cublasMpGridDestroy(grid_row_major));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

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