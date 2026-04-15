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
#include "../common/cublas_reference_trsm.hpp"

// This example builds based on the previous example trsm_batched_thread.cu, with a more performant approach, i.e., each thread solves one RHS of a batch of B
// The results are compared with the reference values obtained with cublasXtrsm API.

using namespace cusolverdx;

// The problem size is the same as the previous example trsm_batched_thread.cu
// If left side: Each batch of A is 10 x 10 matrix, and B is 10 x 12 matrix with 12 right hand sides. But we set the size operator to be Size<M, 1> to indicate each thread solves one RHS of a batch of B
constexpr unsigned M = 10;
constexpr unsigned N = 12;
using TRSM_base      = decltype(Size<M, 1>() + Precision<double>() + Type<type::real>() + Function<function::trsm>() + Side<side::left>() + FillMode<lower>() + TransposeMode<trans>() +
                           Diag<diag::non_unit>() + Arrangement<col_major, row_major>() + Thread());

// if right side: Each batch of A is 12 x 12 matrix, and B is 10 x 12 matrix with 10 rows of right hand sides. We should set the size operator to be Size<1, N> to indicate each thread solves one RHS of one batch of B. Uncomment below to use the right side configuration.
// using TRSM_base      = decltype(Size<1, N>() + Precision<double>() + Type<type::real>() + Function<function::trsm>() + Side<side::right>() + FillMode<lower>() + TransposeMode<trans>() +
//                            Diag<diag::non_unit>() + Arrangement<col_major, row_major>() + Thread());


template<class TRSM, class DataType = typename TRSM::a_data_type>
__global__ void trsm_kernel(const DataType* A, DataType* B, const unsigned int batches) {
    constexpr auto a_m  = (TRSM::side == cusolverdx::side::left) ? M : N;
    constexpr auto nrhs = TRSM::side == cusolverdx::side::left ? N : M;

    const auto tid       = threadIdx.x + blockIdx.x * blockDim.x;
    const auto batch_idx = tid / nrhs;
    const auto rhs_idx   = tid % nrhs;

    if (batch_idx * nrhs + rhs_idx >= batches * nrhs)
        return;

    const auto one_batch_size_a = a_m * a_m;
    auto       this_A           = A + one_batch_size_a * batch_idx;

    const auto one_batch_size_b = M * N;
    // Find the start pointer of the RHS for the current batch of B
    const auto b_rhs_start = ((TRSM::side == cusolverdx::side::left) == (TRSM::b_arrangement == cusolverdx::arrangement::col_major)) ? a_m : 1;
    const auto b_rhs_stride = ((TRSM::side == cusolverdx::side::left) == (TRSM::b_arrangement == cusolverdx::arrangement::col_major)) ? 1 : nrhs;
    auto       this_B        = B + one_batch_size_b * batch_idx + rhs_idx * b_rhs_start;

    TRSM().execute(this_A, a_m, this_B, b_rhs_stride);
}

template<int Arch>
int trsm_thread_execution() {
    using TRSM           = decltype(TRSM_base() + SM<Arch>());
    using data_type      = typename TRSM::a_data_type;
    using cuda_data_type = typename TRSM::a_cuda_data_type;

    const auto     batches          = 300;
    constexpr auto a_m              = TRSM::side == side::left ? M : N;
    constexpr auto nrhs             = TRSM::side == side::left ? N : M;
    const auto     one_batch_size_A = a_m * a_m;
    const auto     one_batch_size_B = M * N;
    constexpr bool is_col_maj_a     = TRSM::a_arrangement == arrangement::col_major;
    constexpr bool is_col_maj_b     = TRSM::b_arrangement == arrangement::col_major;
    constexpr auto ldb              = is_col_maj_b ? M : N;

    printf("TRSM Size M = %d, N = %d, %s side\n", TRSM::m_size, TRSM::n_size, (TRSM::side == side::left) ? "left" : "right");
    printf("matrix A size is %dx%d, matrix B size is %dx%d, total batches = %d\n", a_m, a_m, M, N, batches);

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(one_batch_size_A * batches);
    std::vector<data_type> B(one_batch_size_B * batches);
    std::vector<data_type> X(one_batch_size_B * batches);

    // Fill A with a triangular matrix
    common::fillup_random_matrix<data_type>(is_col_maj_a, a_m, a_m, A.data(), a_m, false /* symm */, true /* diag_dom */, 1, 2, batches);

    // Fill B with random values
    common::fillup_random_matrix<data_type>(is_col_maj_b, M, N, B.data(), ldb, false, false, -2, 1, batches);

    data_type* d_A = nullptr; /* device copy of A */
    data_type* d_B = nullptr; /* device copy of B */

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    const auto nthreads = 32;
    const auto nblocks  = (batches * nrhs + nthreads - 1) / nthreads; // each thread solvers one RHS of a batch of B
    printf("Launching kernel with GridSize = %d, BlockSize= %d -> total # of threads = %d\n", nblocks, nthreads, nblocks * nthreads);
    printf("# of batches x # of RHSs = %d x %d = %d\n", batches, nrhs, batches * nrhs);

    trsm_kernel<TRSM><<<nblocks, nthreads, 0, stream>>>(d_A, d_B, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    // // Uncomment below to print the results after cuSolverDx execute
    // // print A
    // printf("A = \n");
    // common::print_matrix<data_type, a_m, a_m, lda, is_col_maj_a>(A.data(), batches);
    // printf("=====\n");
    // // print B
    // printf("B = \n");
    // common::print_matrix<data_type, M, N, ldb, is_col_maj_b>(B.data(), batches);
    // printf("=====\n");
    // printf("after cuSolverDx execute\n");
    // printf("X = \n");
    // common::print_matrix<data_type, M, N, ldb, is_col_maj_b>(X.data(), batches);
    // printf("=====\n");

    //=======================================================
    // Reference using cuBLAS trsm
    //=======================================================
    common::reference_cublas_trsm<data_type, cuda_data_type, false /* check_blas_trsm_perf */>(A,
                                                                                               B,
                                                                                               M,
                                                                                               N,
                                                                                               batches,
                                                                                               side_of_v<TRSM> == side::left,
                                                                                               fill_mode_of_v<TRSM> == fill_mode::lower,
                                                                                               diag_of_v<TRSM> == diag::unit,
                                                                                               (transpose_mode_of_v<TRSM> == transpose::non_transposed) ? false : true,
                                                                                               is_col_maj_a,
                                                                                               is_col_maj_b,
                                                                                               batches);

    // printf("cuBlas TRSM referece results\n");
    // common::print_matrix<data_type, M, N, ldb, is_col_maj_b>(B.data(), batches);
    // printf("=====\n");

    auto total_relative_error = common::check_error<data_type, data_type>(X.data(), B.data(), batches * one_batch_size_B);
    printf("TRSM: relative error between cuSolverDx and reference results: = %e\n", total_relative_error);
    if (common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Success compared to reference Result " << std::endl;
    } else {
        std::cout << "Failure compared to reference Result " << std::endl;
        return 1;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));

    CUDA_CHECK_AND_EXIT(cudaDeviceReset());

    return 0;
}

template<int Arch>
struct trsm_thread_execution_functor {
    int operator()() { return trsm_thread_execution<Arch>(); }
};

int main() { return common::run_example_with_sm<trsm_thread_execution_functor>(); }
