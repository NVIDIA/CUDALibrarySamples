// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <iostream>

#include <cusolverdx.hpp>
#include "common.hpp"

// This example demonstrates how to use cuSolverDx API to solve a batched linear systems with multiple right hand side after performing LU
// factorization (without pivoting) of the batched general matrix A.  The results are compared with the reference values obtained with cuSolver host API.

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ void kernel(DataType* A, unsigned int lda, DataType* B, unsigned int ldb, typename Solver::status_type* info, const unsigned int batches) {

    constexpr auto m    = Solver::m_size;
    constexpr auto n    = Solver::n_size;
    constexpr auto nrhs = Solver::nrhs;

    const auto lda_gmem              = cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major ? m : n;
    const auto ldb_gmem              = cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major ? n : nrhs;
    const auto one_batch_size_a_gmem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? lda * n: m * lda;
    const auto one_batch_size_b_gmem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? ldb * nrhs : n * ldb;

    constexpr auto lda_smem              = Solver::lda;
    constexpr auto ldb_smem              = Solver::ldb;
    constexpr auto one_batch_size_a_smem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? lda_smem * n : m * lda_smem;
    constexpr auto one_batch_size_b_smem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? ldb_smem * nrhs : n * ldb_smem;

    extern __shared__ unsigned char __align__(sizeof(DataType)) shared_mem[];

    DataType* As = reinterpret_cast<DataType*>(shared_mem);
    DataType* Bs = As + one_batch_size_a_smem * BatchesPerBlock;

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    auto Ag = A + size_t(one_batch_size_a_gmem) * batch_idx;
    auto Bg = B + size_t(one_batch_size_b_gmem) * batch_idx;

    // Load data from global memory to registers
    common::io<Solver, BatchesPerBlock>::load(Ag, lda_gmem, As, lda_smem);
    common::io<Solver, BatchesPerBlock>::load_rhs(Bg, ldb_gmem, Bs, ldb_smem);

    Solver().execute(As, lda_smem, Bs, ldb_smem, &info[batch_idx]);

    // store
    common::io<Solver, BatchesPerBlock>::store(As, lda_smem, Ag, lda_gmem);
    common::io<Solver, BatchesPerBlock>::store_rhs(Bs, ldb_smem, Bg, ldb_gmem);
}

template<int Arch>
int gesv_batched_wo_pivot() {

    using namespace cusolverdx;
    using Base   = decltype(Size<5, 5, 4>() + Precision<float>() + Type<type::real>() + Function<gesv_no_pivot>() + Arrangement<arrangement::col_major, col_major>() + LeadingDimension<5, 7>() +
                          SM<Arch>() + Block());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>() + BlockDim<65, 1, 1>());

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<Solver>;
    using cuda_data_type = typename example::a_cuda_data_type_t<Solver>;
#else
    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
#endif
    constexpr auto m            = Solver::m_size;
    constexpr auto n            = Solver::n_size;
    constexpr auto nrhs         = Solver::nrhs;
    const auto     lda          = arrangement_of_v_a<Solver> == col_major ? m : n;
    const auto     ldb          = arrangement_of_v_b<Solver> == col_major ? n : nrhs;
    const auto     input_size_a = m * n;
    const auto     input_size_b = n * nrhs;

    const auto batches        = 14;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * padded_batches);
    common::fillup_random_diagonal_dominant_matrix<data_type>(arrangement_of_v_a<Solver> == col_major, m, n, A.data(), lda, false, 2, 4, batches); // not symmetric
    std::vector<data_type> L(input_size_a * padded_batches);

    std::vector<data_type> B(input_size_b * padded_batches);
    common::fillup_random_matrix<data_type>(arrangement_of_v_b<Solver> == col_major, n, nrhs, B.data(), ldb, false, false, -1, 1, batches);
    std::vector<data_type> X(input_size_b * padded_batches);

    std::vector<int> info(padded_batches, 0);
    data_type*       d_A    = nullptr; /* device copy of A */
    data_type*       d_B    = nullptr; /* device copy of A */
    int*             d_info = nullptr; /* error info */

    // printf("A = \n");
    // common::print_matrix(m, n * batches, A.data(), lda);
    // const unsigned int b_dim_fast = (arrangement_of_v_b<Solver> == col_major) ? n : nrhs;
    // printf("B = \n");
    // common::print_matrix(b_dim_fast, B.size() / b_dim_fast, B.data(), ldb);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda, d_B, ldb, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.end(), 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }
    // printf("=====\n");
    // printf(" after cuSolverDx\n");
    // printf("L = \n");
    // common::print_matrix(m, n * batches, L.data(), lda);
    // printf("X = \n");
    // common::print_matrix(b_dim_fast, X.size() / b_dim_fast, X.data(), ldb);

    //=========================
    // cuSolver reference
    //=========================
    common::reference_cusolver_lu<data_type, cuda_data_type>(A,
                                                             B,
                                                             info.data(),
                                                             m,
                                                             n,
                                                             nrhs,
                                                             padded_batches,
                                                             false,
                                                             (arrangement_of_v_a<Solver> == arrangement::col_major),
                                                             (arrangement_of_v_b<Solver> == arrangement::col_major),
                                                             false,
                                                             true,
                                                             nullptr,
                                                             batches);

    // printf("=====\n");
    // printf(" after cuSolver API\n");
    // printf("A = \n");
    // common::print_matrix(m, n * batches, A.data(), lda);
    // printf("B = \n");
    // common::print_matrix(b_dim_fast, X.size() / b_dim_fast, B.data(), ldb);

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));

    // check result
    auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), batches * input_size_a);
    std::cout << "GETRF no pivoting: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error << std::endl;

    total_relative_error = common::check_error<data_type, data_type>(X.data(), B.data(), batches * input_size_b);
    printf("GETRS: relative error of B between cuSolverDx and cuSolver results: = %e\n", total_relative_error);
    if (common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Success compared to cuSolver getrs Result " << std::endl;
        return 0;
    } else {
        std::cout << "Failure compared to cuSolver getrs Result " << std::endl;
        return 1;
    }
    CUDA_CHECK_AND_EXIT(cudaDeviceReset());
    return 0;
}

template<int Arch>
struct gesv_batched_wo_pivot_functor {
    int operator()() { return gesv_batched_wo_pivot<Arch>(); }
};


int main() { return common::run_example_with_sm<gesv_batched_wo_pivot_functor>(); }
