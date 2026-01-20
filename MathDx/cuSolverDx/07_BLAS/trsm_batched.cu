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
#include "../common/cublas_reference_trsm.hpp"

// This example demonstrates how to use cuSolverDx API to solve triangular systems
// The results are compared with the reference values obtained with cublasXtrsm API.

template<class TRSM, unsigned int BatchesPerBlock, class DataType = typename TRSM::a_data_type>
__global__ __launch_bounds__(TRSM::max_threads_per_block) void trsm_kernel(const DataType* A, const unsigned int lda_gmem, DataType* B, const unsigned int ldb_gmem, const unsigned int batches) {
    constexpr auto a_m = (cusolverdx::side_of_v<TRSM> == cusolverdx::side::left) ? TRSM::m_size : TRSM::n_size;

    constexpr auto one_batch_size_a = a_m * a_m;
    constexpr auto one_batch_size_b = TRSM::m_size * TRSM::n_size;

    constexpr auto lda_smem              = TRSM::lda;
    constexpr auto ldb_smem              = TRSM::ldb;
    const auto     one_batch_size_a_smem = lda_smem * a_m;
    const auto     one_batch_size_b_smem = ldb_smem * (cusolverdx::arrangement_of_v_b<TRSM> == cusolverdx::col_major ? TRSM::n_size : TRSM::m_size);


    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [As, Bs] = cusolverdx::shared_memory::slice<DataType, DataType>(shared_mem,
                                                                         alignof(DataType),
                                                                         one_batch_size_a_smem * BatchesPerBlock,
                                                                         alignof(DataType) // the size (number of elements) may be omitted for the last pointer
    );

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    auto this_A = A + one_batch_size_a * batch_idx;
    auto this_B = B + one_batch_size_b * batch_idx;

    // Load data from global memory to shared memory
    common::io<TRSM, BatchesPerBlock>::load_a<a_m, a_m, cusolverdx::arrangement_of_v_a<TRSM>>(this_A, lda_gmem, As, lda_smem);
    common::io<TRSM, BatchesPerBlock>::load_b<TRSM::m_size, TRSM::n_size>(this_B, ldb_gmem, Bs, ldb_smem);

    TRSM().execute(As, lda_smem, Bs, ldb_smem);

    // store
    common::io<TRSM, BatchesPerBlock>::store_b<TRSM::m_size, TRSM::n_size>(Bs, ldb_smem, this_B, ldb_gmem);
}

template<int Arch>
int simple_trsm() {
    using namespace cusolverdx;

    // The Size Operator for TRSM only takes M, N. If K is specified, it is ignored. 
    using TRSM = decltype(Size<32, 33>() + Precision<double>() + Type<type::real>() + Function<function::trsm>() + Side<side::right>() + FillMode<lower>() + TransposeMode<non_trans>() +
                          Diag<diag::non_unit>() + Arrangement<col_major, col_major>() + SM<Arch>() + Block());

    constexpr unsigned bpb = TRSM::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Using Suggested BlockDim = " << TRSM::suggested_block_dim.x << std::endl;
    std::cout << "Using Specified BlockDim = " << TRSM::block_dim.x << std::endl;


    using data_type      = typename TRSM::a_data_type;
    using cuda_data_type = typename TRSM::a_cuda_data_type;

    constexpr auto m        = TRSM::m_size;
    constexpr auto n        = TRSM::n_size;
    constexpr auto lda_smem = TRSM::lda;
    constexpr auto ldb_smem = TRSM::ldb;

    std::cout << "lda_smem = " << lda_smem << ", ldb_smem = " << ldb_smem << std::endl;

    constexpr bool is_col_maj_a = arrangement_of_v_a<TRSM> == arrangement::col_major;
    constexpr bool is_col_maj_b = arrangement_of_v_b<TRSM> == arrangement::col_major;

    // no padding for global memory
    constexpr auto a_m = (side_of_v<TRSM> == side::left) ? m : n;
    constexpr auto lda = a_m;
    constexpr auto ldb = is_col_maj_b ? m : n;

    printf("TRSM Size m = %d, n = %d\n", m, n);
    std::cout << "Using leading dimension LDA = " << lda_smem << ", LDB = " << ldb_smem << std::endl;

    const auto batches        = 1;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    const auto one_batch_size_A = lda * a_m; // no padding for global memory
    const auto one_batch_size_B = m * n;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(one_batch_size_A * padded_batches);
    std::vector<data_type> B(one_batch_size_B * padded_batches);
    std::vector<data_type> X(one_batch_size_B * padded_batches);

    // Fill A with a triangular matrix
    common::fillup_random_matrix<data_type>(is_col_maj_a, a_m, a_m, A.data(), lda, false /* symm */, true /* diag_dom */, 1, 2, batches);

    // Fill B with random values
    common::fillup_random_matrix<data_type>(is_col_maj_b, m, n, B.data(), ldb, false, false, -2, 1, batches);

    data_type* d_A = nullptr; /* device copy of A */
    data_type* d_B = nullptr; /* device copy of B */

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    // Increase max dynamic shared memory for the kernel if needed.
    const auto sm_size = TRSM::shared_memory_size;
    std::cout << "sm_size = " << sm_size << std::endl;

    const auto kernel = trsm_kernel<TRSM, TRSM::batches_per_block>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

    //Invokes kernel
    kernel<<<(batches + bpb - 1) / bpb, TRSM::block_dim, sm_size, stream>>>(d_A, lda, d_B, ldb, batches);
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
    // common::print_matrix<data_type, m, n, ldb, is_col_maj_b>(B.data(), batches);
    // printf("=====\n");
    // printf("after cuSolverDx execute\n");
    // printf("X = \n");
    // common::print_matrix<data_type, m, n, ldb, is_col_maj_b>(X.data(), padded_batches);
    // printf("=====\n");

    //=======================================================
    // Reference using cuBLAS trsm
    //=======================================================
    common::reference_cublas_trsm<data_type, cuda_data_type, false /* check_blas_trsm_perf */>(A,
                                                                                               B,
                                                                                               m,
                                                                                               n,
                                                                                               padded_batches,
                                                                                               side_of_v<TRSM> == side::left,
                                                                                               fill_mode_of_v<TRSM> == fill_mode::lower,
                                                                                               diag_of_v<TRSM> == diag::unit,
                                                                                               (transpose_mode_of_v<TRSM> == transpose::non_transposed) ? false : true,
                                                                                               is_col_maj_a,
                                                                                               is_col_maj_b,
                                                                                               batches);

    // printf("cuBlas TRSM referece results\n");
    // common::print_matrix<data_type, m, n, ldb, is_col_maj_b>(B.data(), padded_batches);
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
struct simple_trsm_functor {
    int operator()() { return simple_trsm<Arch>(); }
};

int main() { return common::run_example_with_sm<simple_trsm_functor>(); }
