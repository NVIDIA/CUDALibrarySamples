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

// This example demonstrates how to use cuSolverDx API with Thread execution to perform Cholesky factorization for a batched symmetric, positive-definite matrices. Each batch has small problem size, so it is more efficient to use Thread execution.
// The results are compared with the reference values obtained with cuSolver host API.

template<class POTRF>
__global__ void potrf_kernel(typename POTRF::a_data_type* A, typename POTRF::status_type* info, const unsigned int batches) {

    constexpr auto m                = POTRF::m_size;
    const auto     one_batch_size_a = POTRF::lda * m;

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches)
        return;

    auto Ag = A + size_t(one_batch_size_a) * batch_idx;

    POTRF().execute(Ag, &info[batch_idx]);
}

template<int Arch>
int potrf_batched_thread() {

    using namespace cusolverdx;

    using POTRF = decltype(Size<8>() + Precision<double>() + Type<type::real>() + Function<function::potrf>() + FillMode<fill_mode::lower>() +
                           Arrangement<col_major>() + SM<Arch>() + Thread());

    using data_type      = typename POTRF::a_data_type;
    using cuda_data_type = typename POTRF::a_cuda_data_type;

    constexpr auto m = POTRF::m_size;
    constexpr auto n = POTRF::n_size;
    static_assert(m == n, "potrf is for Hermitian positive-definite matrix matrix only");
    constexpr auto lda_smem = POTRF::lda;

    constexpr bool is_col_maj_a = arrangement_of_v_a<POTRF> == arrangement::col_major;

    // no padding for global memory
    constexpr auto lda              = m;
    const auto     batches          = 250;
    const auto     one_batch_size_A = lda * n; // no padding for global memory

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(one_batch_size_A * batches);
    std::vector<data_type> L(one_batch_size_A * batches);

    common::fillup_random_diagonal_dominant_matrix<data_type>(arrangement_of_v_a<POTRF> == col_major, m, n, A.data(), lda, false, 2, 4, batches);

    std::vector<int> info(batches, 0);
    data_type*       d_A    = nullptr; /* device copy of A */
    int*             d_info = nullptr; /* error info */

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * batches));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    //Invokes kernel
    const auto kernel   = potrf_kernel<POTRF>;
    const int  nthreads = 128;
    kernel<<<(batches + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_A, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
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

    // Uncomment below to print the results after cuSolverDx execute
    // printf("after cuSolverDx execute\n");
    // printf("L = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(L.data(), batches);
    // printf("=====\n");

    //=======================================================
    // cuSolver reference with potrfBatched
    //=======================================================
    std::vector<data_type> dummy_B;
    common::reference_cusolver_cholesky<data_type, cuda_data_type, false /* do_solver */>(A, dummy_B, info.data(), m, 1, batches,
            (fill_mode_of_v<POTRF> == fill_mode::lower), /* is_lower? */
            is_col_maj_a, true, batches);

    auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), batches * one_batch_size_A);
    printf("BATCHED POTRF: relative error of A between cuSolverDx and cuSolver results: = %e\n", total_relative_error);

    // Uncomment below to print the results after cuSolver reference execute
    // printf("after cuSolver API execute\n");
    // printf("A = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);
    // printf("=====\n");

    if (common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Success compared to cuSolver potrfBatched Result " << std::endl;
    } else {
        std::cout << "Failure compared to cuSolver potrfBatched Result " << std::endl;
        return 1;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));

    CUDA_CHECK_AND_EXIT(cudaDeviceReset());

    return 0;
}

template<int Arch>
struct potrf_batched_thread_functor {
    int operator()() { return potrf_batched_thread<Arch>(); }
};


int main() { return common::run_example_with_sm<potrf_batched_thread_functor>(); }
