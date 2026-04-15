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
#include "../common/cusolver_reference_geqrf_gels.hpp"

// This example demonstrates how to use cuSolverDx API to solve a batched least squares problem with Thread execution.
// The thread execution is used for the case where the problem size is small, so it could be more efficient to use Thread execution.
// The results are compared with reference values obtained with the cuSolver and cuBLAS APIs.

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ void gels_kernel(DataType* A, DataType* tau, DataType* B, const unsigned batches) {

    constexpr auto m      = Solver::m_size;
    constexpr auto n      = Solver::n_size;
    constexpr auto k      = Solver::k_size;
    constexpr auto max_mn = m >= n ? m : n;

    const auto one_batch_size_a_gmem = m * n;
    const auto one_batch_size_b_gmem = max_mn * k;

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches)
        return;

    auto Ag    = A + one_batch_size_a_gmem * batch_idx;
    auto Bg    = B + one_batch_size_b_gmem * batch_idx;
    auto tau_g = tau + min(m, n) * batch_idx;

    Solver().execute(Ag, tau_g, Bg);
}

template<int Arch>
int gels_batched_thread() {

    using namespace cusolverdx;
    using Solver = decltype(Size<8, 6, 3>() + Precision<double>() + Type<type::complex>() + Function<gels>() +
                            Arrangement<arrangement::col_major, arrangement::row_major>() + TransposeMode<conj_trans>() + SM<Arch>() + Thread());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr auto m      = Solver::m_size;
    constexpr auto n      = Solver::n_size;
    constexpr auto k      = Solver::k_size;
    const auto     max_mn = m >= n ? m : n;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;
    constexpr bool is_col_maj_b = arrangement_of_v_b<Solver> == arrangement::col_major;

    const auto lda = is_col_maj_a ? m : n;
    const auto ldb = is_col_maj_b ? max_mn : k;

    constexpr auto input_size_a = m * n; // input A size is m x n per batch
    constexpr auto input_size_b = max_mn * k; // allocate B/X with max(m, n) x k per batch

    constexpr auto output_size_x = (transpose_mode_of_v<Solver> == trans || transpose_mode_of_v<Solver> == conj_trans) ? n * k : m * k;

    const auto batches = 300;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, 2, 4, batches); // not symmetric and not diagonally dominant

    std::vector<data_type> B(input_size_b * batches);
    common::fillup_random_matrix<data_type>(is_col_maj_b, max_mn, k, B.data(), ldb, false, false, -1, 1, batches);
    std::vector<data_type> X(input_size_b * batches);


    std::vector<data_type> L(input_size_a * batches);
    std::vector<data_type> tau(min(m, n) * batches);
    std::vector<data_type> tau_ref(min(m, n) * batches);
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

    //Invokes kernel
    const auto nthreads = 64;
    gels_kernel<Solver><<<(batches + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_A, d_tau, d_B, batches);
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
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    //==================================================================================================================
    // cuSolver/cuBLAS reference using cusolverDnXgeqrf, cusolverDnunmqr, and cublas<t>trsm.
    //=================================================================================================================
    if (m >= n) {
        common::reference_cusolver_geqrf_gels<data_type, cuda_data_type, true, false>(
                A, B, tau_ref, m, n, k, batches, is_col_maj_a, is_col_maj_b, (transpose_mode_of_v<Solver> != non_trans), batches);

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


        std::cout << "Success compared with cuSolver/cuBLAS API results, A and tau" << std::endl;
    } else {
        std::cout << "Comparing cuSolverDx GELS for m <= n cases with cuSolver and cuBlas APIs are not implemented in the example." << std::endl;
    }
    return 0;
}

template<int Arch>
struct gels_batched_thread_functor {
    int operator()() { return gels_batched_thread<Arch>(); }
};


int main() { return common::run_example_with_sm<gels_batched_thread_functor>(); }
