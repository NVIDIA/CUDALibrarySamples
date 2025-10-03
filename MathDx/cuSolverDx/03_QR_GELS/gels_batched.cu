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

#include <cusolverdx.hpp>

#include "../common/common.hpp"
#include "../common/cudart.hpp"
#include "../common/error_checking.hpp"
#include "../common/random.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/device_io.hpp"
#include "../common/print.hpp"
#include "../common/cusolver_reference_qr.hpp"

// This example demonstrates how to use cuSolverDx API to solve a batched least squares problem.
// The results are compared with the reference values obtained with cuSolver host API, cuSolverDnXgeqrf + cuSolverDnXormqr + cuBlasXtrsm.

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void kernel(DataType* A, const int lda_gmem, DataType* tau, DataType* B, const int ldb_gmem, const unsigned batches) {

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto k = Solver::k_size;

    const auto     one_batch_size_a_gmem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? lda_gmem * n : m * lda_gmem;
    const auto     one_batch_size_b_gmem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? ldb_gmem * k : m * ldb_gmem;
    constexpr auto lda_smem              = Solver::lda;
    constexpr auto ldb_smem              = Solver::ldb;
    constexpr auto one_batch_size_a_smem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? lda_smem * n : m * lda_smem;
    constexpr auto one_batch_size_b_smem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? ldb_smem * k : m * ldb_smem;

    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [As, Bs, tau_s] = cusolverdx::shared_memory::slice<DataType, DataType, DataType>(
        shared_mem,
        alignof(DataType), one_batch_size_a_smem * BatchesPerBlock,
        alignof(DataType), one_batch_size_b_smem * BatchesPerBlock,
        alignof(DataType)  // the size (number of elements) may be omitted for the last pointer
    );

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;
    auto Ag    = A + one_batch_size_a_gmem * batch_idx;
    auto Bg    = B + one_batch_size_b_gmem * batch_idx;
    auto tau_g = tau + min(m, n) * batch_idx;

    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a(Ag, lda_gmem, As, lda_smem);
    common::io<Solver, BatchesPerBlock>::load_b(Bg, ldb_gmem, Bs, ldb_smem);

    Solver().execute(As, lda_smem, tau_s, Bs, ldb_smem);

    // Store results back to global memory
    common::io<Solver, BatchesPerBlock>::store_a(As, lda_smem, Ag, lda_gmem);
    common::io<Solver, BatchesPerBlock>::store_b(Bs, ldb_smem, Bg, ldb_gmem);

    // store tau from shared memory to global memory
    int thread_id = threadIdx.x + Solver::block_dim.x * (threadIdx.y + Solver::block_dim.y * threadIdx.z);
    for (int i = thread_id; i < min(m, n) * BatchesPerBlock; i += Solver::max_threads_per_block) {
        tau_g[i] = tau_s[i];
    }
}

template<int Arch>
int gels_batched() {

    using namespace cusolverdx;
    using Base   = decltype(Size<20, 16, 1>() + Precision<double>() + Type<type::complex>() + Function<gels>() + Arrangement<arrangement::col_major, arrangement::row_major>() + TransposeMode<conj_trans>() + SM<Arch>() + Block() + BlockDim<64>());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<Solver>;
    using cuda_data_type = typename example::a_cuda_data_type_t<Solver>;
#else
    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
#endif
    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto k = Solver::k_size;
    const auto max_mn = m >= n ? m : n;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;
    constexpr bool is_col_maj_b = arrangement_of_v_b<Solver> == arrangement::col_major;

    const auto     lda          = is_col_maj_a ? m : n;
    const auto     ldb          = is_col_maj_b ? max_mn : k;

    constexpr auto input_size_a = m * n; // input A size is m x n per batch
    constexpr auto input_size_b = max_mn * k; // allocate B/X with max(m, n) x k per batch

    constexpr auto output_size_x = (transpose_mode_of_v<Solver> == trans || transpose_mode_of_v<Solver> == conj_trans) ? n * k : m * k;

    const auto batches        = 3;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * padded_batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, 2, 4, batches); // not symmetric and not diagonally dominant

    std::vector<data_type> B(input_size_b * padded_batches);
    common::fillup_random_matrix<data_type>(is_col_maj_b, max_mn, k, B.data(), ldb, false, false, -1, 1, batches);
    std::vector<data_type> X(input_size_b * padded_batches);


    std::vector<data_type> L(input_size_a * padded_batches);
    std::vector<data_type> tau(min(m, n) * padded_batches);
    std::vector<data_type> tau_ref(min(m, n) * padded_batches);
    data_type*             d_A   = nullptr;
    data_type*             d_B   = nullptr;
    data_type*             d_tau = nullptr;

    // Uncomment below to print the input matrices A and B
    // printf("A = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);
    // printf("B = \n");
    // common::print_matrix<data_type, max_mn, k, ldb, is_col_maj_b>(B.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_tau), sizeof(data_type) * tau.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda, d_tau, d_B, ldb, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(tau.data(), d_tau, sizeof(data_type) * tau.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));


    // Uncomment below to print the results after cuSolverDx execute
    // printf("=====\n");
    // printf(" after cuSolverDx\n");
    // printf("L = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(L.data(), batches);
    // printf("X = \n");
    // common::print_matrix<data_type, max_mn, k, ldb, is_col_maj_b>(X.data(), batches);


    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));
    CUDA_CHECK_AND_EXIT(cudaFree(d_tau));

    //=========================
    // cuSolver reference
    //=========================
    if (m >= n) {
        common::reference_cusolver_qr<data_type, cuda_data_type, true>(A, B, tau_ref, m, n, k, padded_batches, batches, is_col_maj_a, is_col_maj_b, (transpose_mode_of_v<Solver> == trans || transpose_mode_of_v<Solver> == conj_trans));

        // check A, tau and X
        const auto total_relative_error_a = common::check_error<data_type, data_type>(L.data(), A.data(), batches * input_size_a);
        std::cout << "GELS: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error_a << std::endl;
        const auto total_relative_error_tau = common::check_error<data_type, data_type>(tau.data(), tau_ref.data(), batches * min(m, n));
        std::cout << "GELS: relative error of tau between cuSolverDx and cuSolver results: " << total_relative_error_tau << std::endl;

        const auto total_relative_error_b = common::check_error<data_type, data_type>(X.data(), B.data(), batches * output_size_x);
        std::cout << "GELS: relative error of X between cuSolverDx and cuSolver results: " << total_relative_error_b << std::endl;

        // Uncomment below to print the results after cuSolver reference execute
        // printf("Lref = \n");
        // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);
        // printf("=====\n");
        // printf("Xref = \n");
        // common::print_matrix<data_type, max_mn, k, ldb, is_col_maj_b>(B.data(), batches);


        if (!common::is_error_acceptable<data_type>(total_relative_error_a)) {
            std::cout << "Failure compared with cuSolver API results A" << std::endl;
            return 1;
        }
        if (!common::is_error_acceptable<data_type>(total_relative_error_b)) {
            std::cout << "Failure compared with cuSolver API results X" << std::endl;
            return 1;
        }
        if (!common::is_error_acceptable<data_type>(total_relative_error_tau)) {
            std::cout << "Failure compared with cuSolver API results TAU" << std::endl;
            return 1;
        }

        std::cout << "Success compared with cuSolver API results, A and tau" << std::endl;
    } else {
        std::cout << "Comparing cuSolverDx GELS for m <= n cases with cuSolver and cuBlas APIs are not implemented in the example." << std::endl;
    }
    return 0;
}

template<int Arch>
struct gels_batched_functor {
    int operator()() { return gels_batched<Arch>(); }
};


int main() { return common::run_example_with_sm<gels_batched_functor>(); }