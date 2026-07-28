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

#include <cusolverdx.hpp>
#include <cusolverdx_io.hpp>

#include <numeric>

#include "../common/cudart.hpp"
#include "../common/device_io.hpp"
#include "../common/error_checking.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/random.hpp"
#include "../common/cusolver_reference_hegv.hpp"
#include "../common/cublas_reference_hegv_eigenvectors.hpp"

// This example demonstrates batched Hermitian-definite generalized eigenproblems with cuSolverDx HEGV.
// Eigenvalues are compared with cuSolverDn<t>sy/he_gvj or cuSolverDn<t>sy/hegvd; eigenvectors are verified using cuBLAS GEMM operations. 

template<class Solver,
         unsigned int BatchesPerBlock,
         bool         compute_vectors = false,
         typename DataType            = typename Solver::a_data_type,
         typename PrecisionType       = typename Solver::a_precision>
__global__ void kernel(DataType*                     A,
                       const int                     lda_gmem,
                       DataType*                     B,
                       const int                     ldb_gmem,
                       PrecisionType*                lambda,
                       DataType*                     workspace,
                       typename Solver::status_type* info,
                       const unsigned                batches) {
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(Solver);

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches) {
        return;
    }

    constexpr auto m                     = Solver::m_size;
    constexpr auto one_batch_size_a_gmem = m * m;
    constexpr auto one_batch_size_b_gmem = m * m;
    constexpr auto lda_smem              = Solver::lda;
    constexpr auto ldb_smem              = Solver::ldb;
    const auto     one_batch_size_a_smem = lda_smem * m;
    const auto     one_batch_size_b_smem = ldb_smem * m;
    constexpr auto workspace_elems       = Solver::workspace_size > 0 ? Solver::workspace_size : 1;

    extern __shared__ __align__(16) cusolverdx::byte shared_mem[];
    auto [A_s, B_s, lambda_s, workspace_s] =
        cusolverdx::shared_memory::slice<DataType, DataType, PrecisionType, DataType>(shared_mem,
                                                                                      alignof(DataType),
                                                                                      one_batch_size_a_smem * BatchesPerBlock,
                                                                                      alignof(DataType),
                                                                                      one_batch_size_b_smem * BatchesPerBlock,
                                                                                      alignof(PrecisionType),
                                                                                      m * BatchesPerBlock,
                                                                                      alignof(DataType),
                                                                                      workspace_elems);

    auto A_g      = A + one_batch_size_a_gmem * batch_idx;
    auto B_g      = B + one_batch_size_b_gmem * batch_idx;
    auto lambda_g = lambda + m * batch_idx;

    common::io<Solver, BatchesPerBlock>::load_a<m, m>(A_g, lda_gmem, A_s, lda_smem);
    common::io<Solver, BatchesPerBlock>::load_b<m, m>(B_g, ldb_gmem, B_s, ldb_smem);

    Solver().execute(A_s, lda_smem, B_s, ldb_smem, lambda_s, workspace_s, &info[batch_idx]);

    cusolverdx::copy_2d<Solver, m, 1, cusolverdx::arrangement::col_major, BatchesPerBlock, PrecisionType>(
        lambda_s, m, lambda_g, m);
    if constexpr (compute_vectors) {
        common::io<Solver, BatchesPerBlock>::store_a<m, m, Solver::a_arrangement>(A_s, lda_smem, A_g, lda_gmem);
    }
}

template<int Arch>
int hegv_batched_block() {

    using namespace cusolverdx;

    using Base   = decltype(Size<32>() + Precision<float>() + Type<type::complex>() + Function<hegv>() +
                          FillMode<fill_mode::upper>() + Arrangement<arrangement::row_major, arrangement::col_major>() +
                          EigType<2>() + Job<job::overwrite_vectors>() + SM<Arch>() + Block());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
    using precision_type = typename Solver::a_precision;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;
    std::cout << "HEGV EigType = " << eig_type_of_v<Solver> << std::endl;

    constexpr auto m = Solver::m_size;

    constexpr bool is_col_maj_a    = arrangement_of_v_a<Solver> == arrangement::col_major;
    constexpr bool is_col_maj_b    = arrangement_of_v_b<Solver> == arrangement::col_major;
    constexpr bool is_lower_fill   = fill_mode_of_v<Solver> == fill_mode::lower;
    constexpr bool compute_vectors = job_of_v<Solver> != job::no_vectors;
    constexpr int  eig_type        = eig_type_of_v<Solver>;

    const auto     lda_gmem     = m;
    const auto     ldb_gmem     = m;
    constexpr auto input_size_a = m * m;

    const auto batches        = 5;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * padded_batches);
    std::vector<data_type> B(input_size_a * padded_batches);

    common::fillup_random_matrix<data_type>(
        is_col_maj_a, m, m, A.data(), lda_gmem, true /*symmetric*/, false /*diagonal dominant*/, -2, 4, batches);
    common::fillup_random_matrix<data_type>(
        is_col_maj_b, m, m, B.data(), ldb_gmem, false /*symmetric*/, true /*diagonal dominant*/, -2, 1, batches);

    auto A_input = A;
    auto B_input = B;

    std::vector<data_type> V;
    if constexpr (compute_vectors) {
        V.resize(input_size_a * padded_batches);
    }

    std::vector<precision_type> lambda(m * padded_batches, 0);
    std::vector<precision_type> lambda_ref(m * padded_batches, 0);
    std::vector<int>            info(padded_batches, 0);

    data_type*      d_A         = nullptr;
    data_type*      d_B         = nullptr;
    precision_type* d_lambda    = nullptr;
    data_type*      d_workspace = nullptr;
    int*            d_info      = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_lambda), sizeof(precision_type) * lambda.size()));
    CUDA_CHECK_AND_EXIT(
        cudaMalloc(reinterpret_cast<void**>(&d_workspace), sizeof(data_type) * std::max(Solver::workspace_size, 1u)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        kernel<Solver, bpb, compute_vectors>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    kernel<Solver, bpb, compute_vectors><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(
        d_A, lda_gmem, d_B, ldb_gmem, d_lambda, d_workspace, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(
        cudaMemcpyAsync(lambda.data(), d_lambda, sizeof(precision_type) * lambda.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));
    if constexpr (compute_vectors) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(V.data(), d_A, sizeof(data_type) * V.size(), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));
    CUDA_CHECK_AND_EXIT(cudaFree(d_lambda));
    CUDA_CHECK_AND_EXIT(cudaFree(d_workspace));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    for (unsigned int j = 0; j < batches; ++j) {
        if (info[j] != 0) {
            std::cout << "non-zero info after cuSolverDx HEGV kernel for batch " << j << " (info=" << info[j] << ")" << std::endl;
            return -1;
        }
    }

    // Compare eigenvalues with cuSolver reference
    std::vector<data_type> A_ref = A_input;
    std::vector<data_type> B_ref = B_input;
    if (!common::reference_cusolver_hegv<data_type, cuda_data_type, precision_type, false /*true:use_sygvj; false:use_sygvd*/>(
            A_ref,
            B_ref,
            lambda_ref,
            info.data(),
            m,
            eig_type,
            padded_batches,
            is_lower_fill,
            is_col_maj_a,
            is_col_maj_b,
            compute_vectors,
            batches)) {
        return -1;
    }

    const auto total_relative_error_lambda =
        common::check_error<precision_type, precision_type>(lambda.data(), lambda_ref.data(), batches * m);
    std::cout << "HEGV: relative error of lambda between cuSolverDx and cuSolver sygvj/hegvj reference: "
              << total_relative_error_lambda << std::endl;

    if (!common::is_error_acceptable<precision_type>(total_relative_error_lambda)) {
        std::cout << "Failure compared with cuSolver sygvj/hegvj reference (lambda)" << std::endl;
        return 1;
    }

    if constexpr (compute_vectors) {
        // Verify eigenvectors
        const double ev_error =
            common::cublas_reference_hegv_eigenvector_verification<eig_type,
                                                                   fill_mode_of_v<Solver>,
                                                                   arrangement_of_v_a<Solver>,
                                                                   arrangement_of_v_b<Solver>,
                                                                   data_type,
                                                                   precision_type>(A_input, B_input, V, lambda, m, batches);

        std::cout << "HEGV: eigenvector verification error (cuBLAS GEMM check): " << ev_error << std::endl;
        if (!common::is_error_acceptable<precision_type>(ev_error)) {
            std::cout << "Failure on generalized eigenvector verification" << std::endl;
            return 1;
        }
    }

    std::cout << "Success: eigenvalues match cuSolverDn<t>sygvd/sygvj";
    if constexpr (compute_vectors) {
        std::cout << " and eigenvectors pass cuBLAS GEMM check";
    }
    std::cout << std::endl;
    return 0;
}

template<int Arch>
struct hegv_batched_block_functor {
    int operator()() { return hegv_batched_block<Arch>(); }
};

int main() {
    return common::run_example_with_sm<hegv_batched_block_functor>();
}
