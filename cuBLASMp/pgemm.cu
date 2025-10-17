/*
 * Copyright 2023 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
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
    using input_t = __half;
    using output_t = __half;
    using compute_t = float;
    const cudaDataType_t cuda_input_type = CUDA_R_16F;
    const cudaDataType_t cuda_output_type = CUDA_R_16F;
    const cublasComputeType_t cublas_compute_type = CUBLAS_COMPUTE_32F;

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
    const int64_t k = opts.k;
    const int64_t ia = opts.ia;
    const int64_t ja = opts.jb;
    const int64_t ib = opts.ib;
    const int64_t jb = opts.jb;
    const int64_t ic = opts.ic;
    const int64_t jc = opts.jc;
    const int64_t mbA = opts.mbA;
    const int64_t nbA = opts.nbA;
    const int64_t mbB = opts.mbB;
    const int64_t nbB = opts.nbB;
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
    CUBLAS_CHECK(cublasMpCreate(&handle, stream));

    cublasMpGrid_t grid = nullptr;

    cublasMpMatrixDescriptor_t descA = nullptr;
    cublasMpMatrixDescriptor_t descB = nullptr;
    cublasMpMatrixDescriptor_t descC = nullptr;

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    double* d_work = nullptr;

    compute_t alpha = 1.0;
    compute_t beta = 1.0;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    const int64_t global_m_a = (ia - 1) + m;
    const int64_t global_n_a = (ja - 1) + k;
    const int64_t global_m_b = (ib - 1) + k;
    const int64_t global_n_b = (jb - 1) + n;
    const int64_t global_m_c = (ic - 1) + m;
    const int64_t global_n_c = (jc - 1) + n;

    const int64_t llda = cublasMpNumroc(global_m_a, mbA, myprow, 0, nprow);
    const int64_t loc_n_a = cublasMpNumroc(global_n_a, nbA, mypcol, 0, npcol);

    const int64_t lldb = cublasMpNumroc(global_m_b, mbB, myprow, 0, nprow);
    const int64_t loc_n_b = cublasMpNumroc(global_n_b, nbB, mypcol, 0, npcol);

    const int64_t lldc = cublasMpNumroc(global_m_c, mbC, myprow, 0, nprow);
    const int64_t loc_n_c = cublasMpNumroc(global_n_c, nbC, mypcol, 0, npcol);

    std::vector<input_t> h_A(llda * loc_n_a, 0);
    std::vector<input_t> h_B(lldb * loc_n_b, 0);
    std::vector<output_t> h_C(lldc * loc_n_c, 0);

    generate_random_matrix(m, k, h_A.data(), mbA, nbA, ia, ja, llda, nprow, npcol, myprow, mypcol);
    generate_random_matrix(k, n, h_B.data(), mbB, nbB, ib, jb, lldb, nprow, npcol, myprow, mypcol);
    generate_random_matrix(m, n, h_C.data(), mbC, nbC, ic, jc, lldc, nprow, npcol, myprow, mypcol);

    CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a * sizeof(input_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b * sizeof(input_t), stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c * sizeof(output_t), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), llda * loc_n_a * sizeof(input_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), lldb * loc_n_b * sizeof(input_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, h_C.data(), lldc * loc_n_c * sizeof(output_t), cudaMemcpyHostToDevice, stream));

    CUBLAS_CHECK(cublasMpGridCreate(
        handle,
        nprow,
        npcol,
        opts.grid_layout == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        cal_comm,
        &grid));

    CUBLAS_CHECK(cublasMpMatrixDescriptorCreate(
        handle, global_m_a, global_n_a, mbA, nbA, 0, 0, llda, cuda_input_type, grid, &descA));
    CUBLAS_CHECK(cublasMpMatrixDescriptorCreate(
        handle, global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, cuda_input_type, grid, &descB));
    CUBLAS_CHECK(cublasMpMatrixDescriptorCreate(
        handle, global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, cuda_output_type, grid, &descC));

    CUBLAS_CHECK(cublasMpGemm_bufferSize(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        &beta,
        d_C,
        ic,
        jc,
        descC,
        cublas_compute_type,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));

    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CAL_CHECK(cal_stream_sync(cal_comm, stream));
    CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    const double begin = MPI_Wtime();

    CUBLAS_CHECK(cublasMpGemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        &beta,
        d_C,
        ic,
        jc,
        descC,
        cublas_compute_type,
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost));

    CAL_CHECK(cal_stream_sync(cal_comm, stream));
    CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    const double end = MPI_Wtime();

    printf("Duration: %lf GFlops: %lf\n", end - begin, (2 * m * n * k * 1e-9) / (end - begin));

    CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descA));
    CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descB));
    CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descC));

    CUBLAS_CHECK(cublasMpGridDestroy(handle, grid));

    CUBLAS_CHECK(cublasMpDestroy(handle));

    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));
    CUDA_CHECK(cudaFreeAsync(d_work, stream));

    CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    CAL_CHECK(cal_comm_destroy(cal_comm));

    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
};