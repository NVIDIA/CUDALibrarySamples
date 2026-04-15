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

#include "../common/cudart.hpp"
#include "../common/error_checking.hpp"
#include "../common/random.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/device_io.hpp"
#include "../common/print.hpp"
#include "../common/cusolver_reference_ungqr.hpp"

// This example demonstrates how to use cuSolverDx API to generate matrix Q from QR factorization of matrix A with Thread execution.
// The results are compared with the reference values obtained with cuSolver host API.
//
// UNGQR generates an M-by-N real/complex matrix Q with orthonormal columns,
// which is defined as the first N columns of a product of K elementary
// reflectors of order M:
//
// Q = H(0) * H(1) * ... * H(K-1)
//
// as returned by GEQRF.

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ void ungqr_kernel(DataType* A, const DataType* tau, const unsigned batches) {
    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto k = Solver::k_size;

    constexpr auto one_batch_size_a_gmem = m * n;

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches)
        return;

    auto this_A   = A + one_batch_size_a_gmem * batch_idx;
    auto this_tau = tau + k * batch_idx;

    Solver().execute(this_A, this_tau);
}

template<int Arch>
int ungqr_batched_thread() {

    using namespace cusolverdx;
    using Solver = decltype(Size<8, 5, 3>() + Precision<float>() + Type<type::complex>() + Function<ungqr>() + Arrangement<arrangement::row_major>() +
                            SM<Arch>() + Thread());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
    constexpr auto m     = Solver::m_size;
    constexpr auto n     = Solver::n_size;
    constexpr auto k     = Solver::k_size;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;

    const auto     lda          = is_col_maj_a ? m : n;
    constexpr auto input_size_a = m * n;

    const auto batches = 200;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Generate input matrices A and B
    std::vector<data_type> A(input_size_a * batches);
    std::vector<data_type> A_result(input_size_a * batches);
    std::vector<data_type> tau(k * batches);

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

    const auto nthreads = 64;
    ungqr_kernel<Solver><<<(batches + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_A, d_tau, batches);
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
    bool ref_success = common::reference_cusolver_ungqr<data_type, cuda_data_type>(A, tau, m, n, k, batches, batches, is_col_maj_a);

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
struct ungqr_batched_thread_functor {
    int operator()() { return ungqr_batched_thread<Arch>(); }
};

int main() { return common::run_example_with_sm<ungqr_batched_thread_functor>(); }
