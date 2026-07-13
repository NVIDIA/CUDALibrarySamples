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
#include <cuda_fp8.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#include "helpers.h"
#include "matrix_generator.hxx"

static Result run_tp_matmul(const Options& opts, ncclComm_t comm)
{
    Result result;
    const int rank = get_nccl_rank(comm);
    using input_t = __half;
    using output_t = __half;
    using compute_t = float;
    const cudaDataType_t cuda_input_type = CUDA_R_16F;
    const cudaDataType_t cuda_output_type = CUDA_R_16F;
    const cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F;
    const cublasOperation_t transA = CUBLAS_OP_T;
    const cublasOperation_t transB = CUBLAS_OP_N;

    const int nranks = opts.p * opts.q;

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

    CUBLASMP_CHECK(cublasMpGridCreate(nranks, 1, CUBLASMP_GRID_LAYOUT_COL_MAJOR, comm, &grid_col_major));
    CUBLASMP_CHECK(cublasMpGridCreate(1, nranks, CUBLASMP_GRID_LAYOUT_ROW_MAJOR, comm, &grid_row_major));

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
            ta ? 0 : rank,
            ta ? rank : 0,
            rank);
        generate_random_matrix(k, n, h_W0.data(), loc_b_m, loc_b_n, 1, 1, loc_b_m, 1, nranks, 0, rank, rank);
        generate_random_matrix(m, n, h_X1.data(), loc_c_m, loc_c_n, 1, 1, loc_c_m, nranks, 1, rank, 0, rank);

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

        const cublasMpMatmulAlgoType_t algoType = CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT;

        CUBLASMP_CHECK(cublasMpMatmulDescriptorCreate(&matmulDesc, cublas_compute_type));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &transA, sizeof(transA)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &transB, sizeof(transB)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
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

        NCCL_CHECK(ncclMemAlloc(&d_work, workspaceInBytesOnDevice));

        cublasMpStatus_t register_status = cublasMpBufferRegister(grid_col_major, d_work, workspaceInBytesOnDevice);
        if (register_status != CUBLASMP_STATUS_SUCCESS && rank == 0)
        {
            fprintf(
                stderr,
                "Warning: failed to register workspace memory with cuBLASMp (%s); continuing without workspace "
                "registration. The implementation will fall back to the NO_OVERLAP algorithm.\n",
                cublasMpGetStatusString(register_status));
        }

        std::vector<int8_t> h_work(workspaceInBytesOnHost);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int warmup = opts.warmup;
        const int cycles = opts.cycles;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < warmup + cycles; i++)
        {
            if (i == warmup)
            {
                CUDA_CHECK(cudaEventRecord(start, stream));
            }

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
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        const double elapsed = (elapsed_ms / 1000.0) / cycles;
        result.elapsed = std::max(result.elapsed, elapsed);

        if (rank == 0)
        {
            printf("AG + Matmul: %lf (s) %lf (GFlops)\n", elapsed, (2 * m * n * k * 1e-9) / elapsed);
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUBLASMP_CHECK(cublasMpMatmulDescriptorDestroy(matmulDesc));

        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descC));

        CUDA_CHECK(cudaFree(d_X0));
        CUDA_CHECK(cudaFree(d_W0));
        if (register_status == CUBLASMP_STATUS_SUCCESS)
        {
            CUBLASMP_CHECK(cublasMpBufferDeregister(grid_col_major, d_work));
        }
        NCCL_CHECK(ncclMemFree(d_work));
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
            ta ? rank : 0,
            ta ? 0 : rank,
            rank);
        generate_random_matrix(m, n, h_X2.data(), loc_c_m, loc_c_n, 1, 1, loc_c_m, 1, nranks, 0, rank, rank);

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

        const cublasMpMatmulAlgoType_t algoType = CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT;

        CUBLASMP_CHECK(cublasMpMatmulDescriptorCreate(&matmulDesc, cublas_compute_type));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &transA, sizeof(transA)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &transB, sizeof(transB)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
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

        NCCL_CHECK(ncclMemAlloc(&d_work, workspaceInBytesOnDevice));

        cublasMpStatus_t register_status = cublasMpBufferRegister(grid_row_major, d_work, workspaceInBytesOnDevice);
        if (register_status != CUBLASMP_STATUS_SUCCESS && rank == 0)
        {
            fprintf(
                stderr,
                "Warning: failed to register workspace memory with cuBLASMp (%s); continuing without workspace "
                "registration. The implementation will fall back to the NO_OVERLAP algorithm.\n",
                cublasMpGetStatusString(register_status));
        }

        std::vector<int8_t> h_work(workspaceInBytesOnHost);

        CUDA_CHECK(cudaStreamSynchronize(stream));

        const int warmup = opts.warmup;
        const int cycles = opts.cycles;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < warmup + cycles; i++)
        {
            if (i == warmup)
            {
                CUDA_CHECK(cudaEventRecord(start, stream));
            }

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
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        const double elapsed = (elapsed_ms / 1000.0) / cycles;
        result.elapsed = std::max(result.elapsed, elapsed);

        if (rank == 0)
        {
            printf("Matmul + RS: %lf (s) %lf (GFlops)\n", elapsed, (2 * m * n * k * 1e-9) / elapsed);
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUBLASMP_CHECK(cublasMpMatmulDescriptorDestroy(matmulDesc));

        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));
        CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descC));

        CUDA_CHECK(cudaFree(d_X1));
        CUDA_CHECK(cudaFree(d_W1));
        CUDA_CHECK(cudaFree(d_X2));
        if (register_status == CUBLASMP_STATUS_SUCCESS)
        {
            CUBLASMP_CHECK(cublasMpBufferDeregister(grid_row_major, d_work));
        }
        NCCL_CHECK(ncclMemFree(d_work));
    }

    CUBLASMP_CHECK(cublasMpGridDestroy(grid_col_major));
    CUBLASMP_CHECK(cublasMpGridDestroy(grid_row_major));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

    CUDA_CHECK(cudaStreamDestroy(stream));

    return Result {};
}

int main(int argc, char* argv[])
{
    Options opts = { .m = 0,
                     .n = 0,
                     .k = 0,
                     .mbA = 1,
                     .nbA = 1,
                     .mbB = 1,
                     .nbB = 1,
                     .mbC = 1,
                     .nbC = 1,
                     .ia = 1,
                     .ja = 1,
                     .ib = 1,
                     .jb = 1,
                     .ic = 1,
                     .jc = 1,
                     .p = 1,
                     .q = 2,
                     .grid_layout = 'c',
                     .verbose = false,
                     .cycles = 10,
                     .warmup = 5 };

    opts.parse(argc, argv);
    opts.validate();

    if (opts.cycles <= 0)
    {
        fprintf(stderr, "Error: -cycles expects a positive integer\n");
        return EXIT_FAILURE;
    }

    const int nranks = opts.p * opts.q;
    Comm comm(nranks, opts.gpus_per_process);
    const Result result = comm.collective_launch([&](ncclComm_t nccl_comm) { return run_tp_matmul(opts, nccl_comm); });

    if (comm.is_root())
    {
        printf(status_ok(result.status) ? "[SUCCEEDED]\n" : "[FAILED]\n");
    }

    return status_ok(result.status) ? EXIT_SUCCESS : EXIT_FAILURE;
}
