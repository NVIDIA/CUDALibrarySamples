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
#include "../common/cublas_reference_geqrf_gels.hpp"

// This example demonstrates how to use cuSolverDx API to compute the QR factorization on a batched m x n matrix A using Thread execution.
// The example is for small problem size, so it could be more efficient to use Thread execution instead of Block execution.
// The results are compared with the reference values obtained with cuBLAS batched API.

// Note that if M <= N, cuSolverDx GEQRF skips computing the last reflector and tau[min(m, n) - 1] is set to 0
// This could lead to different results from cuSolver reference for complex data type

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ void kernel(DataType* A, DataType* tau, const unsigned batches) {

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;

    const auto one_batch_size = m * n;

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches)
        return;
    auto Ag    = A + one_batch_size * batch_idx;
    auto tau_g = tau + min(m, n) * batch_idx;

    Solver().execute(Ag, tau_g);
}

template<int Arch>
int geqrf_batched_thread() {

    using namespace cusolverdx;
    using Solver = decltype(Size<10, 6>() + Precision<float>() + Type<type::complex>() + Function<geqrf>() + Arrangement<arrangement::row_major>() +
                            SM<Arch>() + Thread());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr auto m      = Solver::m_size;
    constexpr auto n      = Solver::n_size;
    constexpr auto min_mn = m > n ? n : m;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;

    const auto     lda          = is_col_maj_a ? m : n;
    constexpr auto input_size_a = m * n;

    const auto batches = 1000;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, 2, 4, batches); // not symmetric and not diagonally dominant

    std::vector<data_type> L(input_size_a * batches);
    std::vector<data_type> tau(min_mn * batches);
    std::vector<data_type> tau_ref(min_mn * batches);
    data_type*             d_A   = nullptr;
    data_type*             d_tau = nullptr;

    // // Comment below to remove printing A matrix
    // printf("A = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_tau), sizeof(data_type) * tau.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    const auto nthreads = 64;
    kernel<Solver><<<(batches + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_A, d_tau, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(tau.data(), d_tau, sizeof(data_type) * tau.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    // Comment below to remove printing results after cuSolverDx execute
    // printf("=====\n");
    // printf(" after cuSolverDx\n");
    // printf("L = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(L.data(), batches);
    // printf("tau = \n");
    // common::print_matrix<data_type, min_mn, 1, min_mn, true>(tau.data(), batches);


    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_tau));

    //=========================
    // cuBLAS reference
    //=========================
    std::vector<data_type> dummy_b;
    common::reference_cublas_geqrf_gels<data_type, cuda_data_type>(A, dummy_b, tau_ref, m, n, 1, batches, is_col_maj_a);

    const auto total_relative_error_a = common::check_error<data_type, data_type>(L.data(), A.data(), batches * input_size_a);
    std::cout << "GEQRF: relative error of A between cuSolverDx and cuBLAS results: " << total_relative_error_a << std::endl;

    const auto total_relative_error_tau = common::check_error<data_type, data_type>(tau.data(), tau_ref.data(), batches * min_mn);
    std::cout << "GEQRF: relative error of tau between cuSolverDx and cuBLAS results: " << total_relative_error_tau << std::endl;

    // Comment below to remove printing results after cuSolver reference execute
    // printf("Lref = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);
    // printf("tau_ref = \n");
    // common::print_matrix<data_type, min_mn, 1, min_mn, true>(tau_ref.data(), batches);
    // printf("=====\n");


    if (!common::is_error_acceptable<data_type>(total_relative_error_a)) {
        std::cout << "Failure compared with cuBLAS API results A" << std::endl;
        return 1;
    }
    if (!common::is_error_acceptable<data_type>(total_relative_error_tau)) {
        std::cout << "Failure compared with cuBLAS API results TAU" << std::endl;
        return 1;
    }

    std::cout << "Success compared with cuBLAS API results, A and tau" << std::endl;
    return 0;
}

template<int Arch>
struct geqrf_batched_thread_functor {
    int operator()() { return geqrf_batched_thread<Arch>(); }
};


int main() { return common::run_example_with_sm<geqrf_batched_thread_functor>(); }
