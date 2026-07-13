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
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>

#include "helpers.h"
#include "matrix_generator.hxx"

template <typename T>
static bool check_tradd_result(
    cublasMpHandle_t mp_handle,
    ncclComm_t comm,
    cudaStream_t stream,
    int rank,
    cublasMpGrid_t grid,
    int nprow,
    int npcol,
    int myprow,
    int mypcol,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int64_t m,
    int64_t n,
    const T* alpha,
    T* d_A,
    int64_t ia,
    int64_t ja,
    cublasMpMatrixDescriptor_t descA,
    const T* beta,
    T* d_C_ref,
    T* d_C,
    int64_t ic,
    int64_t jc,
    cublasMpMatrixDescriptor_t descC)
{
    static_assert(
        std::is_same_v<T, double> || std::is_same_v<T, float>,
        "tradd sample reference check supports float and double only");

    const int64_t a_rows = trans == CUBLAS_OP_N ? m : n;
    const int64_t a_cols = trans == CUBLAS_OP_N ? n : m;
    T* full_A = nullptr;
    T* full_C_ref = nullptr;
    T* full_C_result = nullptr;
    int64_t full_A_lld = 0;
    int64_t full_C_ref_lld = 0;
    int64_t full_C_result_lld = 0;

    gather_matrix(
        mp_handle,
        comm,
        stream,
        a_rows,
        a_cols,
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
        d_C_ref,
        ic,
        jc,
        descC,
        grid,
        nprow,
        npcol,
        myprow,
        mypcol,
        &full_C_ref,
        &full_C_ref_lld);
    gather_matrix(
        mp_handle,
        comm,
        stream,
        m,
        n,
        d_C,
        ic,
        jc,
        descC,
        grid,
        nprow,
        npcol,
        myprow,
        mypcol,
        &full_C_result,
        &full_C_result_lld);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool passed = true;
    if (rank == 0)
    {
        std::vector<T> h_A(full_A_lld * a_cols);
        std::vector<T> h_C(full_C_ref_lld * n);
        std::vector<T> h_result(full_C_result_lld * n);
        CUDA_CHECK(cudaMemcpyAsync(h_A.data(), full_A, h_A.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_C.data(), full_C_ref, h_C.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            h_result.data(), full_C_result, h_result.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for (int64_t col = 0; col < n; ++col)
        {
            for (int64_t row = 0; row < m; ++row)
            {
                const bool in_triangle = (uplo == CUBLAS_FILL_MODE_LOWER) ? (row >= col) : (row <= col);
                if (!in_triangle)
                {
                    continue;
                }

                const T aval = (trans == CUBLAS_OP_N) ? h_A[row + col * full_A_lld] : h_A[col + row * full_A_lld];
                h_C[row + col * full_C_ref_lld] =
                    static_cast<T>((*alpha) * to_double(aval) + (*beta) * to_double(h_C[row + col * full_C_ref_lld]));
            }
        }

        passed = allclose_host(
            "tradd",
            h_result.data(),
            full_C_result_lld,
            h_C.data(),
            full_C_ref_lld,
            m,
            n,
            default_rtol<T>(),
            default_atol<T>());
    }

    CUDA_CHECK(cudaFree(full_A));
    CUDA_CHECK(cudaFree(full_C_ref));
    CUDA_CHECK(cudaFree(full_C_result));
    return passed;
}

static Result run_tradd(const Options& opts, ncclComm_t comm)
{
    Result result;
    const int rank = get_nccl_rank(comm);
    const int64_t m = opts.m;
    const int64_t n = opts.n;
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

    const int myprow = (opts.grid_layout == 'c' ? rank % nprow : rank / npcol);
    const int mypcol = (opts.grid_layout == 'c' ? rank / nprow : rank % npcol);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasMpHandle_t handle = nullptr;
    CUBLASMP_CHECK(cublasMpCreate(&handle, stream));

    cublasMpGrid_t grid = nullptr;

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descC = nullptr;

    double* d_A = nullptr;
    double* d_C = nullptr;
    double* d_C_ref = nullptr;

    double* d_work = nullptr;

    double alpha = 1.0;
    double beta = 1.0;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    const int64_t global_m_a = (ia - 1) + m;
    const int64_t global_n_a = (ja - 1) + n;
    const int64_t global_m_c = (ic - 1) + m;
    const int64_t global_n_c = (jc - 1) + n;

    const int64_t llda = cublasMpNumroc(global_m_a, mbA, myprow, 0, nprow);
    const int64_t loc_n_a = cublasMpNumroc(global_n_a, nbA, mypcol, 0, npcol);

    const int64_t lldc = cublasMpNumroc(global_m_c, mbC, myprow, 0, nprow);
    const int64_t loc_n_c = cublasMpNumroc(global_n_c, nbC, mypcol, 0, npcol);

    std::vector<double> h_A(llda * loc_n_a, 0);
    std::vector<double> h_C(lldc * loc_n_c, 0);

    generate_diag_matrix(m, n, h_A.data(), mbA, nbA, ia, ja, llda, nprow, npcol, myprow, mypcol, rank);
    generate_diag_matrix(m, n, h_C.data(), mbC, nbC, ic, jc, lldc, nprow, npcol, myprow, mypcol, rank);

    CUDA_CHECK(cudaMalloc(&d_A, llda * loc_n_a * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, lldc * loc_n_c * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C_ref, lldc * loc_n_c * sizeof(double)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), llda * loc_n_a * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, h_C.data(), lldc * loc_n_c * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_ref, h_C.data(), lldc * loc_n_c * sizeof(double), cudaMemcpyHostToDevice, stream));

    CUBLASMP_CHECK(cublasMpGridCreate(nprow, npcol, char_to_grid_layout(opts.grid_layout), comm, &grid));

    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(global_m_a, global_n_a, mbA, nbA, 0, 0, llda, CUDA_R_64F, grid, &descA));
    CUBLASMP_CHECK(
        cublasMpMatrixDescriptorCreate(global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, CUDA_R_64F, grid, &descC));

    CUBLASMP_CHECK(cublasMpTradd_bufferSize(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        m,
        n,
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

        CUBLASMP_CHECK(cublasMpTradd(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N,
            m,
            n,
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
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    result.elapsed = (elapsed_ms / 1000.0) / opts.cycles;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Reset C to the clean input before the single verification call; the timing loop mutates C repeatedly.
    CUDA_CHECK(cudaMemcpyAsync(d_C, d_C_ref, lldc * loc_n_c * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUBLASMP_CHECK(cublasMpTradd(
        handle,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        m,
        n,
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
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost));

    const bool passed = !opts.check_result || check_tradd_result(
                                                  handle,
                                                  comm,
                                                  stream,
                                                  rank,
                                                  grid,
                                                  nprow,
                                                  npcol,
                                                  myprow,
                                                  mypcol,
                                                  CUBLAS_FILL_MODE_LOWER,
                                                  CUBLAS_OP_N,
                                                  m,
                                                  n,
                                                  &alpha,
                                                  d_A,
                                                  ia,
                                                  ja,
                                                  descA,
                                                  &beta,
                                                  d_C_ref,
                                                  d_C,
                                                  ic,
                                                  jc,
                                                  descC);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descA));
    CUBLASMP_CHECK(cublasMpMatrixDescriptorDestroy(descC));

    CUBLASMP_CHECK(cublasMpGridDestroy(grid));

    CUBLASMP_CHECK(cublasMpDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_ref));
    CUDA_CHECK(cudaFree(d_work));

    CUDA_CHECK(cudaStreamDestroy(stream));

    return make_result(passed, result.elapsed);
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

    if (opts.cycles <= 0)
    {
        fprintf(stderr, "Error: -cycles expects a positive integer\n");
        return EXIT_FAILURE;
    }

    const int nranks = opts.p * opts.q;
    Comm comm(nranks, opts.gpus_per_process);
    const Result result = comm.collective_launch([&](ncclComm_t nccl_comm) { return run_tradd(opts, nccl_comm); });

    if (comm.is_root() && result.status == CUBLASMP_STATUS_SUCCESS)
    {
        printf("Duration: %lf\n", result.elapsed);
    }

    if (comm.is_root())
    {
        printf(status_ok(result.status) ? "[SUCCEEDED]\n" : "[FAILED]\n");
    }

    return status_ok(result.status) ? EXIT_SUCCESS : EXIT_FAILURE;
}
