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
#include "../common/measure.hpp"
#include "../common/print.hpp"
#include "../common/cusolver_reference_cholesky.hpp"

// This example demonstrates how to use cuSolverDx API with Thread execution to solve a batched small linear systems with multiple right hand side
// after performing Cholesky factorization of the batched symmetric, positive-definite matrix A.
// The results are compared with the reference values obtained with cuSolver host API.
// Each batch has small problem size, so it couble be more efficient to use Thread execution instead of Block execution

template<class POSV_thread>
__global__ void posv_kernel(
        typename POSV_thread::a_data_type* A, typename POSV_thread::b_data_type* B, typename POSV_thread::status_type* info, const unsigned int batches) {

    using namespace cusolverdx;
    constexpr auto m                = POSV_thread::m_size;
    constexpr auto nrhs             = POSV_thread::k_size;
    const auto     one_batch_size_a = m * m;
    const auto     one_batch_size_b = m * nrhs;

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches)
        return;

    auto this_A = A + one_batch_size_a * batch_idx;
    auto this_B = B + one_batch_size_b * batch_idx;

    POSV_thread().execute(this_A, this_B, &info[batch_idx]);
}

template<int Arch>
int posv_batched_thread() {

    using namespace cusolverdx;

    using POSV = decltype(Size<8, 8, 3>() + Precision<double>() + Type<type::real>() + Function<function::posv>() + FillMode<fill_mode::lower>() +
                          Arrangement<col_major, row_major>() + SM<Arch>() + Thread());

    using data_type      = typename POSV::a_data_type;
    using cuda_data_type = typename POSV::a_cuda_data_type;

    constexpr auto m    = POSV::m_size;
    constexpr auto n    = POSV::n_size;
    constexpr auto nrhs = POSV::k_size;
    static_assert(m == n, "posv is for Hermitian positive-definite matrix matrix only");

    constexpr bool is_col_maj_a = arrangement_of_v_a<POSV> == arrangement::col_major;
    constexpr bool is_col_maj_b = arrangement_of_v_b<POSV> == arrangement::col_major;

    printf("Size m = %d, n = %d, nrhs = %d\n", m, n, nrhs);
    const auto lda = m;
    const auto ldb = is_col_maj_b ? m : nrhs;

    const auto batches          = 100;
    const auto one_batch_size_A = m * m;
    const auto one_batch_size_B = m * nrhs;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(one_batch_size_A * batches);
    std::vector<data_type> L(one_batch_size_A * batches);

    common::fillup_random_diagonal_dominant_matrix<data_type>(arrangement_of_v_a<POSV> == col_major, m, m, A.data(), m, false, 2, 4, batches);

    std::vector<data_type> B(one_batch_size_B * batches);
    common::fillup_random_matrix<data_type>(arrangement_of_v_b<POSV> == col_major, m, nrhs, B.data(), ldb, false, false, -1, 1, batches);
    std::vector<data_type> X(one_batch_size_B * batches);

    std::vector<int> info(batches, 0);
    data_type*       d_A    = nullptr; /* device copy of A */
    data_type*       d_B    = nullptr; /* device copy of B */
    int*             d_info = nullptr; /* error info */

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    //Invokes kernel
    const auto kernel   = posv_kernel<POSV>;
    const int  nthreads = 128;
    const int  nblocks  = (batches + nthreads - 1) / nthreads;
    kernel<<<nblocks, nthreads, 0, stream>>>(d_A, d_B, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    if (std::accumulate(info.begin(), info.end(), 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx kernel \n";
        for (int j = 0; j < info.size(); j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }

    // Uncomment below to print the results after cuSolverDx execute
    // printf("after cuSolverDx execute\n");
    // printf("L = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(L.data(), batches);
    // printf("=====\n");
    // printf("X = \n");
    // common::print_matrix<data_type, n, nrhs, ldb, is_col_maj_b>(X.data(), batches);
    // printf("=====\n");

    //=======================================================
    // cuSolver reference with potrfBatched and portsBatched
    //=======================================================
    common::reference_cusolver_cholesky<data_type, cuda_data_type, true>(
            A, B, info.data(), m, nrhs, batches, (fill_mode_of_v<POSV> == fill_mode::lower), is_col_maj_a, is_col_maj_b, batches);

    auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), batches * one_batch_size_A);
    printf("BATCHED POSV: relative error of A between cuSolverDx and cuSolver results: = %e\n", total_relative_error);

    // Uncomment below to print the results after cuSolver reference execute
    // printf("after cuSolver API execute\n");
    // printf("A = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);
    // printf("B = \n");
    // common::print_matrix<data_type, n, nrhs, ldb, is_col_maj_b>(B.data(), batches);
    // printf("=====\n");

    total_relative_error = common::check_error<data_type, data_type>(X.data(), B.data(), batches * one_batch_size_B);
    printf("BATCHED POSV: relative error of B between cuSolverDx and cuSolver results: = %e\n", total_relative_error);
    if (common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Success compared to cuSolver potrSBatched Result " << std::endl;
    } else {
        std::cout << "Failure compared to cuSolver potrSBatched Result " << std::endl;
        return 1;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));

    CUDA_CHECK_AND_EXIT(cudaDeviceReset());

    return 0;
}

template<int Arch>
struct posv_batched_thread_functor {
    int operator()() { return posv_batched_thread<Arch>(); }
};


int main() { return common::run_example_with_sm<posv_batched_thread_functor>(); }
