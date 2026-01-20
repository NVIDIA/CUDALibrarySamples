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
#include "../common/cusolver_reference_ungqr.hpp"

// This example demonstrates how to use cuSolverDx API to generate matrix Q from QR factorization of matrix A.
// The results are compared with the reference values obtained with cuSolver host API.
// 
// UNGQR generates an M-by-N real/complex matrix Q with orthonormal columns,
// which is defined as the first N columns of a product of K elementary
// reflectors of order M:
// 
// Q = H(0) * H(1) * ... * H(K-1)
// 
// as returned by GEQRF.

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void ungqr_kernel(DataType* A, const DataType* tau, const unsigned batches) {
    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto k = Solver::k_size;

    constexpr auto lda_smem = Solver::lda;
    constexpr auto lda_gmem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? m : n;
    constexpr auto one_batch_size_a_gmem = m * n;
    constexpr auto one_batch_size_a_smem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? lda_smem * n : m * lda_smem;

    extern __shared__ char shared_mem[];
    auto [As, taus] = cusolverdx::shared_memory::slice<DataType, DataType>(shared_mem, alignof(DataType), one_batch_size_a_smem * BatchesPerBlock, alignof(DataType));

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    auto this_A   = A + one_batch_size_a_gmem * batch_idx;
    auto this_tau = tau + k * batch_idx;

    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a<m, n>(this_A, lda_gmem, As, lda_smem);

    const int tid = threadIdx.x + threadIdx.y * Solver::block_dim.x + threadIdx.z * Solver::block_dim.x * Solver::block_dim.y;
    for (int i = tid; i < k * BatchesPerBlock; i += Solver::max_threads_per_block) {
        taus[i] = this_tau[i];
    }
    __syncthreads();

    Solver().execute(As, lda_smem, taus);

    // store
    __syncthreads();
    common::io<Solver, BatchesPerBlock>::store_a<m, n>(As, lda_smem, this_A, lda_gmem);
}

template<int Arch>
int ungqr_batched() {

    using namespace cusolverdx;
    using Base   = decltype(Size<26, 25, 23>() + Precision<float>() + Type<type::complex>() + Function<ungqr>() + Arrangement<arrangement::row_major>() + SM<Arch>() + Block());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto k = Solver::k_size;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;

    const auto     lda          = is_col_maj_a ? m : n;
    constexpr auto input_size_a = m * n;

    const auto batches        = 2;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Generate input matrices A and B
    std::vector<data_type> A(input_size_a * padded_batches);
    std::vector<data_type> A_result(input_size_a * padded_batches);
    std::vector<data_type> tau(k * padded_batches);

    // Fill A with random data (this would normally come from a previous GEQRF call)
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, -0.1, 0.1, batches);

    // Generate tau values (this would normally come from a previous GEQRF call)
    common::fillup_random_matrix<data_type>(true, k, 1, tau.data(), k, false, false, -2, 2, batches);

    data_type* d_A   = nullptr;
    data_type* d_tau = nullptr;

    // Uncomment below to print matrices
    // printf("A = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_tau), sizeof(data_type) * tau.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_tau, tau.data(), sizeof(data_type) * tau.size(), cudaMemcpyHostToDevice, stream));

    auto       sm_size = Solver::get_shared_memory_size(lda);
    const auto kernel  = ungqr_kernel<Solver, bpb>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

    // Invoke kernel to compute Q*B
    kernel<<<padded_batches / bpb, Solver::block_dim, sm_size, stream>>>(d_A, d_tau, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A_result.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    // Uncomment below to print results after cuSolverDx execute
    // printf("=====\n");
    // printf(" after cuSolverDx UNMQR\n");
    // printf("A_result = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A_result.data(), batches);

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_tau));

    //=========================
    // cuSolver reference
    //=========================

    // Use cuSolver UNMQR reference implementation
    bool ref_success = common::reference_cusolver_ungqr<data_type, cuda_data_type>(A,   
                                                                                   tau, 
                                                                                   m,   
                                                                                   n,   
                                                                                   k,   
                                                                                   padded_batches,
                                                                                   batches,
                                                                                   is_col_maj_a);

    if (!ref_success) {
        std::cout << "cuSolver reference computation failed" << std::endl;
        return 1;
    }

    // check A
    const auto total_relative_error_a = common::check_error<data_type, data_type>(A_result.data(), A.data(), batches * input_size_a);
    std::cout << "UNGQR: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error_a << std::endl;

    // Uncomment below to print results after reference execute
    // printf("A_ref = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);
    // printf("=====\n");

    if (!common::is_error_acceptable<data_type>(total_relative_error_a)) {
        std::cout << "Failure compared with cuSolver API results A" << std::endl;
        return 1;
    }

    std::cout << "Success compared with cuSolverDn<t>un/or_mqr API results, A" << std::endl;
    return 0;
}

template<int Arch>
struct ungqr_batched_functor {
    int operator()() { return ungqr_batched<Arch>(); }
};

int main() { return common::run_example_with_sm<ungqr_batched_functor>(); }
