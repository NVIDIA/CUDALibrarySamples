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

#include <cublasmp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include "helpers.h"
#include "matmul.h"
#include "matrix_generator.hxx"

template <typename TypeA, typename TypeB, typename TypeD>
int run_matmul_rs(const Options& opts)
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

    float alpha = 1.0;
    float beta = 0.0;

    cublasMpGrid_t grid_col_major = nullptr;
    cublasMpGrid_t grid_row_major = nullptr;

    CUBLASMP_CHECK(cublasMpGridCreate(nranks, 1, CUBLASMP_GRID_LAYOUT_COL_MAJOR, comm, &grid_col_major));
    CUBLASMP_CHECK(cublasMpGridCreate(1, nranks, CUBLASMP_GRID_LAYOUT_ROW_MAJOR, comm, &grid_row_major));

    const cublasOperation_t transA = opts.transA;
    const cublasOperation_t transB = opts.transB;

    const bool ta = (transA != CUBLAS_OP_N);
    const bool tb = (transB != CUBLAS_OP_N);

    const int64_t m = opts.m;
    const int64_t n = opts.n;
    const int64_t k = opts.k;

    const int64_t loc_a_m = ta ? k / nranks : m;
    const int64_t loc_a_n = ta ? m : k / nranks;
    const int64_t mb_b = tb ? n / nranks : k / nranks;
    const int64_t nb_b = tb ? k / nranks : n / nranks;
    const int64_t loc_b_m = tb ? n : k / nranks;
    const int64_t loc_b_n = tb ? k / nranks : n;
    const int64_t loc_d_m = m;
    const int64_t loc_d_n = n / nranks;

    std::vector<TypeA> h_A(loc_a_m * loc_a_n, TypeA(0));
    std::vector<TypeB> h_B(loc_b_m * loc_b_n, TypeB(0));
    std::vector<TypeD> h_D(loc_d_m * loc_d_n, TypeD(0));

    generate_random_matrix(
        ta ? k : m,
        ta ? m : k,
        h_A.data(),
        loc_a_m,
        loc_a_n,
        1,
        1,
        loc_a_m,
        ta ? nranks : 1,
        ta ? 1 : nranks,
        ta ? rank : 0,
        ta ? 0 : rank);
    generate_random_matrix(
        tb ? n : k,
        tb ? k : n,
        h_B.data(),
        mb_b,
        nb_b,
        1,
        1,
        loc_b_m,
        tb ? 1 : nranks,
        tb ? nranks : 1,
        tb ? 0 : rank,
        tb ? rank : 0);
    generate_random_matrix(m, n, h_D.data(), loc_d_m, loc_d_n, 1, 1, loc_d_m, 1, nranks, 0, rank);

    TypeA* d_A = nullptr;
    TypeB* d_B = nullptr;
    TypeD* d_D = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_A, loc_a_m * loc_a_n * sizeof(TypeA)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, loc_b_m * loc_b_n * sizeof(TypeB)));
    CUDA_CHECK(cudaMalloc((void**)&d_D, loc_d_m * loc_d_n * sizeof(TypeD)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), loc_a_m * loc_a_n * sizeof(TypeA), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), loc_b_m * loc_b_n * sizeof(TypeB), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_D, h_D.data(), loc_d_m * loc_d_n * sizeof(TypeD), cudaMemcpyHostToDevice, stream));

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    cublasMpMatrixDescriptor_t descD = nullptr;

    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        ta ? k : m,
        ta ? m : k,
        loc_a_m,
        loc_a_n,
        0,
        0,
        loc_a_m,
        CudaTypeTraits<TypeA>::typeEnum,
        ta ? grid_col_major : grid_row_major,
        &descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        tb ? n : k,
        tb ? k : n,
        mb_b,
        nb_b,
        0,
        0,
        loc_b_m,
        CudaTypeTraits<TypeB>::typeEnum,
        tb ? grid_row_major : grid_col_major,
        &descB));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        m, n, loc_d_m, loc_d_n, 0, 0, loc_d_m, CudaTypeTraits<TypeD>::typeEnum, grid_row_major, &descD));

    const cublasMpMatmulAlgoType_t algoType = CUBLASMP_MATMUL_ALGO_TYPE_DEFAULT;

    cublasMpMatmulDescriptor_t matmulDesc = nullptr;

    CUBLASMP_CHECK(cublasMpMatmulDescriptorCreate(&matmulDesc, CUBLAS_COMPUTE_32F));
    CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
        matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &transA, sizeof(transA)));
    CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
        matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &transB, sizeof(transB)));
    CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
        matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE, &algoType, sizeof(algoType)));

    // Allocate and set scaling factors if provided
    cublasMpMatmulMatrixScale_t a_scale_mode = string_to_scale_type(opts.scaleA);
    cublasMpMatmulMatrixScale_t b_scale_mode = string_to_scale_type(opts.scaleB);
    cublasMpMatmulMatrixScale_t d_scale_mode = string_to_scale_type(opts.scaleD);
    cublasMpMatmulMatrixScale_t d_out_scale_mode = string_to_scale_type(opts.scaleDOut);
    const int64_t a_scale_rows = ta ? k : m;
    const int64_t a_scale_cols = ta ? m : k;
    const int64_t b_scale_rows = tb ? n : k;
    const int64_t b_scale_cols = tb ? k : n;

    void* d_a_scale = nullptr;
    void* d_b_scale = nullptr;
    void* d_d_scale = nullptr;
    void* d_d_out_scale = nullptr;
    void* d_work = nullptr;
    cublasMpStatus_t register_status = CUBLASMP_STATUS_INVALID_VALUE;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    auto cleanup = [&]() {
        if (start) CUDA_CHECK(cudaEventDestroy(start));
        if (stop) CUDA_CHECK(cudaEventDestroy(stop));

        if (matmulDesc) CUBLASMP_CHECK(cublasMpMatmulDescriptorDestroy(matmulDesc));
        if (descA) CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
        if (descB) CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descB));
        if (descD) CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descD));

        if (d_A) CUDA_CHECK(cudaFree(d_A));
        if (d_B) CUDA_CHECK(cudaFree(d_B));
        if (d_D) CUDA_CHECK(cudaFree(d_D));

        if (d_a_scale) CUDA_CHECK(cudaFree(d_a_scale));
        if (d_b_scale) CUDA_CHECK(cudaFree(d_b_scale));
        if (d_d_scale) CUDA_CHECK(cudaFree(d_d_scale));
        if (d_d_out_scale) CUDA_CHECK(cudaFree(d_d_out_scale));

        if (d_work)
        {
            if (register_status == CUBLASMP_STATUS_SUCCESS)
            {
                CUBLASMP_CHECK(cublasMpBufferDeregister(grid_row_major, d_work));
            }
            NCCL_CHECK(ncclMemFree(d_work));
        }

        if (grid_col_major) CUBLASMP_CHECK(cublasMpGridDestroy(grid_col_major));
        if (grid_row_major) CUBLASMP_CHECK(cublasMpGridDestroy(grid_row_major));
        if (handle) CUBLASMP_CHECK(cublasMpDestroy(handle));

        NCCL_CHECK(ncclCommFinalize(comm));
        NCCL_CHECK(ncclCommDestroy(comm));

        if (stream) CUDA_CHECK(cudaStreamDestroy(stream));
    };

    if (opts.scaleA)
    {
        d_a_scale = allocate_and_init_scaling_factors(a_scale_rows, a_scale_cols, a_scale_mode);
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_MODE, &a_scale_mode, sizeof(a_scale_mode)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale)));
    }

    if (opts.scaleB)
    {
        d_b_scale = allocate_and_init_scaling_factors(b_scale_rows, b_scale_cols, b_scale_mode);
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_MODE, &b_scale_mode, sizeof(b_scale_mode)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale)));
    }

    if (opts.scaleD)
    {
        d_d_scale = allocate_and_init_scaling_factors(m, n, d_scale_mode);
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_MODE, &d_scale_mode, sizeof(d_scale_mode)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc, CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_SCALE_POINTER, &d_d_scale, sizeof(d_d_scale)));
    }

    if (opts.scaleDOut)
    {
        d_d_out_scale = allocate_and_init_scaling_factors(m, n, d_out_scale_mode);
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc,
            CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_OUT_SCALE_MODE,
            &d_out_scale_mode,
            sizeof(d_out_scale_mode)));
        CUBLASMP_CHECK(cublasMpMatmulDescriptorSetAttribute(
            matmulDesc,
            CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_D_OUT_SCALE_POINTER,
            &d_d_out_scale,
            sizeof(d_d_out_scale)));
    }

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    cublasMpStatus_t status = cublasMpMatmul_bufferSize(
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
        nullptr,
        d_D,
        1,
        1,
        descD,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost);
    if (skip_if_cublasmp_not_supported("matmul_rs", "cublasMpMatmul_bufferSize", status, rank))
    {
        cleanup();
        return EXIT_SUCCESS;
    }
    CUBLASMP_CHECK_STATUS(status);

    NCCL_CHECK(ncclMemAlloc(&d_work, workspaceInBytesOnDevice));

    register_status = cublasMpBufferRegister(grid_row_major, d_work, workspaceInBytesOnDevice);
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

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < opts.warmup + opts.cycles; i++)
    {
        if (i == opts.warmup)
        {
            CUDA_CHECK(cudaEventRecord(start, stream));
        }

        status = cublasMpMatmul(
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
            nullptr,
            d_D,
            1,
            1,
            descD,
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost);
        if (skip_if_cublasmp_not_supported("matmul_rs", "cublasMpMatmul", status, rank))
        {
            cleanup();
            return EXIT_SUCCESS;
        }
        CUBLASMP_CHECK_STATUS(status);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));

    if (rank == 0)
    {
        printf(
            "Matmul + RS: %lf (s) %lf (GFlops)\n", elapsed_time / 1000, (2 * m * n * k * 1e-9) / (elapsed_time / 1000));
    }

    const bool passed = !opts.check_result || check_matmul_result(
                                                  "matmul_rs",
                                                  handle,
                                                  comm,
                                                  stream,
                                                  rank,
                                                  ta ? grid_col_major : grid_row_major,
                                                  ta ? nranks : 1,
                                                  ta ? 1 : nranks,
                                                  ta ? rank : 0,
                                                  ta ? 0 : rank,
                                                  tb ? grid_row_major : grid_col_major,
                                                  tb ? 1 : nranks,
                                                  tb ? nranks : 1,
                                                  tb ? 0 : rank,
                                                  tb ? rank : 0,
                                                  grid_row_major,
                                                  1,
                                                  nranks,
                                                  0,
                                                  rank,
                                                  transA,
                                                  transB,
                                                  m,
                                                  n,
                                                  k,
                                                  &alpha,
                                                  d_A,
                                                  descA,
                                                  d_B,
                                                  descB,
                                                  &beta,
                                                  d_D,
                                                  descD,
                                                  CUBLAS_COMPUTE_32F,
                                                  d_a_scale,
                                                  a_scale_mode,
                                                  loc_a_m,
                                                  loc_a_n,
                                                  d_b_scale,
                                                  b_scale_mode,
                                                  loc_b_m,
                                                  loc_b_n,
                                                  d_d_scale,
                                                  d_scale_mode,
                                                  loc_d_m,
                                                  loc_d_n,
                                                  d_d_out_scale,
                                                  d_out_scale_mode,
                                                  loc_d_m,
                                                  loc_d_n,
                                                  /*gather_d_scales=*/true);

    cleanup();

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
};

int main(int argc, char** argv)
{
    Options opts {
        .m = 1024,
        .n = 1024,
        .k = 1024,
        .typeA = CUDA_R_16F,
        .typeB = CUDA_R_16F,
        .typeD = CUDA_R_16F,
        .transA = CUBLAS_OP_T,
        .transB = CUBLAS_OP_N,
        .cycles = 10,
        .warmup = 5,
    };

    opts.parse(argc, argv);

    MPI_Init(&argc, &argv);

    int rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    const bool needs_fp8 = is_fp8(opts.typeA) || is_fp8(opts.typeB) || is_fp8(opts.typeD);
    const bool needs_fp4 = is_fp4(opts.typeA) || is_fp4(opts.typeB) || is_fp4(opts.typeD);

    int status = EXIT_SUCCESS;
    if (needs_fp8 && !deviceSupportsFp8())
    {
        if (rank == 0) fprintf(stderr, "matmul_rs: FP8 not supported on this device, skipping\n");
    }
    else if (needs_fp4 && !deviceSupportsFp4())
    {
        if (rank == 0) fprintf(stderr, "matmul_rs: FP4 not supported on this device, skipping\n");
    }
    else if (opts.typeA == CUDA_R_16F && opts.typeB == CUDA_R_16F && opts.typeD == CUDA_R_16F)
    {
        status = run_matmul_rs<__half, __half, __half>(opts);
    }
    else if (opts.typeA == CUDA_R_16BF && opts.typeB == CUDA_R_16BF && opts.typeD == CUDA_R_16BF)
    {
        status = run_matmul_rs<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_8F_E4M3)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_fp8_e4m3>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_16F)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e4m3, __half>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_16BF)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_32F)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e4m3, float>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E5M2 && opts.typeD == CUDA_R_8F_E4M3)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e5m2, __nv_fp8_e4m3>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E5M2 && opts.typeD == CUDA_R_8F_E5M2)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e5m2, __nv_fp8_e5m2>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E5M2 && opts.typeD == CUDA_R_16F)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e5m2, __half>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E5M2 && opts.typeD == CUDA_R_16BF)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e5m2, __nv_bfloat16>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E4M3 && opts.typeB == CUDA_R_8F_E5M2 && opts.typeD == CUDA_R_32F)
    {
        status = run_matmul_rs<__nv_fp8_e4m3, __nv_fp8_e5m2, float>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E5M2 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_8F_E4M3)
    {
        status = run_matmul_rs<__nv_fp8_e5m2, __nv_fp8_e4m3, __nv_fp8_e4m3>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E5M2 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_8F_E5M2)
    {
        status = run_matmul_rs<__nv_fp8_e5m2, __nv_fp8_e4m3, __nv_fp8_e5m2>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E5M2 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_16F)
    {
        status = run_matmul_rs<__nv_fp8_e5m2, __nv_fp8_e4m3, __half>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E5M2 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_16BF)
    {
        status = run_matmul_rs<__nv_fp8_e5m2, __nv_fp8_e4m3, __nv_bfloat16>(opts);
    }
    else if (opts.typeA == CUDA_R_8F_E5M2 && opts.typeB == CUDA_R_8F_E4M3 && opts.typeD == CUDA_R_32F)
    {
        status = run_matmul_rs<__nv_fp8_e5m2, __nv_fp8_e4m3, float>(opts);
    }
    else if (opts.typeA == CUDA_R_4F_E2M1 && opts.typeB == CUDA_R_4F_E2M1 && opts.typeD == CUDA_R_4F_E2M1)
    {
        status = run_matmul_rs<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_fp4_e2m1>(opts);
    }
    else if (opts.typeA == CUDA_R_4F_E2M1 && opts.typeB == CUDA_R_4F_E2M1 && opts.typeD == CUDA_R_16F)
    {
        status = run_matmul_rs<__nv_fp4_e2m1, __nv_fp4_e2m1, __half>(opts);
    }
    else if (opts.typeA == CUDA_R_4F_E2M1 && opts.typeB == CUDA_R_4F_E2M1 && opts.typeD == CUDA_R_16BF)
    {
        status = run_matmul_rs<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>(opts);
    }
    else if (opts.typeA == CUDA_R_4F_E2M1 && opts.typeB == CUDA_R_4F_E2M1 && opts.typeD == CUDA_R_32F)
    {
        status = run_matmul_rs<__nv_fp4_e2m1, __nv_fp4_e2m1, float>(opts);
    }
    else
    {
        throw std::runtime_error("The matmul_rs sample doesn't support the given datatype combination");
    }

    if (rank == 0)
    {
        printf(status == EXIT_SUCCESS ? "[SUCCEEDED]\n" : "[FAILED]\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return status;
}
