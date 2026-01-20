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
#include "../common/cusolver_reference_unmqr.hpp"

// This example demonstrates how to use cuSolverDx API to multiply a batched matrix B by an orthogonal/unitary matrix Q,
// computed by QR factorization GEQRF. The results are compared with the reference values obtained with cuSolver host API.
//
// UNMQR performs matrix multiplication with an orthogonal/unitary matrix Q:
//   op(Q) * B    (if side = left)
//   B * op(Q)    (if side = right)
// where op(Q) = Q, Q^T, or Q^H depending on the transpose mode.
//
// The matrix Q is represented by its elementary reflectors from QR factorization GEQRF:
//   Q = H(0) * H(1) * ... * H(K-1)
// where each Householder reflector is:
//   H(i) = I - tau(i) * v(i) * v(i)^T
// The v(i) is stored in the i-th column of A below the diagonal.
//

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void unmqr_kernel(const DataType* A, const DataType* tau, DataType* B, const unsigned batches) {

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto k = Solver::k_size;

    // For UNMQR: AM = m, AN = k, BM = m, BN = n
    constexpr auto AM = (cusolverdx::side_of_v<Solver> == cusolverdx::side::left ? m : n);
    constexpr auto AN = k;
    constexpr auto BM = m;
    constexpr auto BN = n;

    const auto lda_gmem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? AM : AN;
    const auto ldb_gmem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? BM : BN;
    const auto lda_smem = Solver::lda;
    const auto ldb_smem = Solver::ldb;

    constexpr auto one_batch_size_a_gmem = AM * AN;
    constexpr auto one_batch_size_b_gmem = BM * BN;
    const auto     one_batch_size_a_smem = cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major ? lda_smem * AN : AM * lda_smem;
    const auto     one_batch_size_b_smem = cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major ? ldb_smem * BN : BM * ldb_smem;

    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [As, Bs, taus] = cusolverdx::shared_memory::slice<DataType, DataType, DataType>(
        shared_mem, alignof(DataType), one_batch_size_a_smem * BatchesPerBlock, alignof(DataType), one_batch_size_b_smem * BatchesPerBlock, alignof(DataType), k * BatchesPerBlock);

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;
    auto this_A   = A + one_batch_size_a_gmem * batch_idx;
    auto this_B   = B + one_batch_size_b_gmem * batch_idx;
    auto this_tau = tau + k * batch_idx;

    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a<AM, AN>(this_A, lda_gmem, As, lda_smem);
    common::io<Solver, BatchesPerBlock>::load_b<BM, BN>(this_B, ldb_gmem, Bs, ldb_smem);

    // Load tau from global memory to shared memory
    const int tid = threadIdx.x + threadIdx.y * Solver::block_dim.x + threadIdx.z * Solver::block_dim.x * Solver::block_dim.y;
    for (int i = tid; i < k * BatchesPerBlock; i += Solver::max_threads_per_block) {
        taus[i] = this_tau[i];
    }
    __syncthreads();

    Solver().execute(As, lda_smem, taus, Bs, ldb_smem);

    // Store results back to global memory
    common::io<Solver, BatchesPerBlock>::store_b<BM, BN>(Bs, ldb_smem, this_B, ldb_gmem);
}

template<int Arch>
int unmqr_batched() {

    using namespace cusolverdx;
    using Base   = decltype(Size<32, 16, 14>() + Precision<float>() + Type<type::complex>() + Function<unmqr>() + Arrangement<arrangement::row_major, arrangement::row_major>() +
                          TransposeMode<conj_trans>() + Side<side::right>() + SM<Arch>() + Block());
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
    constexpr bool is_col_maj_b = arrangement_of_v_b<Solver> == arrangement::col_major;

    constexpr auto a_m = (side_of_v<Solver> == side::left ? m : n);
    constexpr auto a_n = k;
    constexpr auto b_m = m;
    constexpr auto b_n = n;

    const auto     lda          = is_col_maj_a ? a_m : a_n;
    const auto     ldb          = is_col_maj_b ? b_m : b_n;
    constexpr auto input_size_a = a_m * a_n;
    constexpr auto input_size_b = b_m * b_n;

    const auto batches        = 2;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Generate input matrices A and B
    std::vector<data_type> A(input_size_a * padded_batches);
    std::vector<data_type> B(input_size_b * padded_batches);
    std::vector<data_type> tau(k * padded_batches);

    // Fill A with random data (this would normally come from a previous GEQRF call)
    common::fillup_random_matrix<data_type>(is_col_maj_a, a_m, a_n, A.data(), lda, false, false, -0.1, 0.1, batches);

    // Fill B with random data (the matrix to be multiplied by Q)
    common::fillup_random_matrix<data_type>(is_col_maj_b, b_m, b_n, B.data(), ldb, false, false, 1, 2, batches);

    // Generate tau values (this would normally come from a previous GEQRF call)
    common::fillup_random_matrix<data_type>(true, k, 1, tau.data(), k, false, false, -2, 2, batches);

    std::vector<data_type> B_result(input_size_b * padded_batches);

    data_type* d_A   = nullptr;
    data_type* d_tau = nullptr;
    data_type* d_B   = nullptr;

    // Uncomment below to print matrices
    // printf("A = \n");
    // common::print_matrix<data_type, a_m, a_n, lda, is_col_maj_a>(A.data(), batches);
    // printf("B = \n");
    // common::print_matrix<data_type, b_m, b_n, ldb, is_col_maj_b>(B.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_tau), sizeof(data_type) * tau.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_tau, tau.data(), sizeof(data_type) * tau.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    auto       sm_size = Solver::get_shared_memory_size(lda, ldb);
    const auto kernel  = unmqr_kernel<Solver, bpb>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

    // Invoke kernel to compute Q*B
    kernel<<<padded_batches / bpb, Solver::block_dim, sm_size, stream>>>(d_A, d_tau, d_B, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(B_result.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    // Uncomment below to print results after cuSolverDx execute
    // printf("=====\n");
    // printf(" after cuSolverDx UNMQR\n");
    // printf("B_result = \n");
    // common::print_matrix<data_type, b_m, b_n, ldb, is_col_maj_b>(B_result.data(), batches);

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_tau));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));

    //=========================
    // cuSolver reference
    //=========================

    // Use cuSolver UNMQR reference implementation
    bool ref_success = common::reference_cusolver_unmqr<data_type, cuda_data_type>(A,   
                                                                                   B,   
                                                                                   tau, 
                                                                                   m,   
                                                                                   n,   
                                                                                   k,   
                                                                                   padded_batches,
                                                                                   batches,
                                                                                   side_of_v<Solver> == side::left,
                                                                                   is_col_maj_a,
                                                                                   is_col_maj_b,
                                                                                   transpose_mode_of_v<Solver> != transpose::non_transposed);

    if (!ref_success) {
        std::cout << "cuSolver reference computation failed" << std::endl;
        return 1;
    }

    // check B
    const auto total_relative_error_b = common::check_error<data_type, data_type>(B_result.data(), B.data(), batches * input_size_b);
    std::cout << "UNMQR: relative error of B between cuSolverDx and cuSolver results: " << total_relative_error_b << std::endl;

    // Uncomment below to print results after reference execute
    // printf("B_ref = \n");
    // common::print_matrix<data_type, b_m, b_n, ldb, is_col_maj_b>(B.data(), batches);
    // printf("=====\n");

    if (!common::is_error_acceptable<data_type>(total_relative_error_b)) {
        std::cout << "Failure compared with cuSolver API results B" << std::endl;
        return 1;
    }

    std::cout << "Success compared with cuSolverDn<t>un/or_mqr API results, B" << std::endl;
    return 0;
}

template<int Arch>
struct unmqr_batched_functor {
    int operator()() { return unmqr_batched<Arch>(); }
};

int main() { return common::run_example_with_sm<unmqr_batched_functor>(); }
