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
#include "../common/cublas_reference_geqrf_gels.hpp"

// This example demonstrates how to use cuSolverDx API to compute the QR factorization on a batched m x n matrix A.
// The results are compared with the reference values obtained with cuBLAS batched API.

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void kernel(DataType* A, const int lda_gmem, DataType* tau, const unsigned batches) {

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;

    const auto     one_batch_size_a_gmem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? lda_gmem * n : m * lda_gmem;
    constexpr auto lda_smem              = Solver::lda;
    constexpr auto one_batch_size_a_smem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? lda_smem * n : m * lda_smem;

    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [As, tau_s] = cusolverdx::shared_memory::slice<DataType, DataType>(
        shared_mem,
        alignof(DataType), one_batch_size_a_smem * BatchesPerBlock,
        alignof(DataType)  // the size (number of elements) may be omitted for the last pointer
    );

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;
    auto Ag    = A + one_batch_size_a_gmem * batch_idx;
    auto tau_g = tau + min(m, n) * batch_idx;

    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a(Ag, lda_gmem, As, lda_smem);

    Solver().execute(As, lda_smem, tau_s);

    // Store results back to global memory
    common::io<Solver, BatchesPerBlock>::store_a(As, lda_smem, Ag, lda_gmem);

    // store tau from shared memory to global memory
    int thread_id = threadIdx.x + Solver::block_dim.x * (threadIdx.y + Solver::block_dim.y * threadIdx.z);
    for (int i = thread_id; i < min(m, n) * BatchesPerBlock; i += Solver::max_threads_per_block) {
        tau_g[i] = tau_s[i];
    }
    
}

template<int Arch>
int geqrf_batched() {

    using namespace cusolverdx;
    using Base   = decltype(Size<16, 20>() + Precision<float>() + Type<type::real>() + Function<geqrf>() + Arrangement<arrangement::row_major>() + SM<Arch>() + Block());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto min_mn = m > n ? n : m;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;

    const auto     lda          = is_col_maj_a ? m : n;
    constexpr auto input_size_a = m * n;

    const auto batches        = 2;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * padded_batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, 2, 4, batches); // not symmetric and not diagonally dominant

    std::vector<data_type> L(input_size_a * padded_batches);
    std::vector<data_type> tau(min_mn * padded_batches);
    std::vector<data_type> tau_ref(min_mn * padded_batches);
    data_type*             d_A   = nullptr;
    data_type*             d_tau = nullptr;

    // Comment below to remove printing A matrix
    printf("A = \n");
    common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_tau), sizeof(data_type) * tau.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda, d_tau, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(tau.data(), d_tau, sizeof(data_type) * tau.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    // Comment below to remove printing results after cuSolverDx execute
    printf("=====\n");
    printf(" after cuSolverDx\n");
    printf("L = \n");
    common::print_matrix<data_type, m, n, lda, is_col_maj_a>(L.data(), batches);
    printf("tau = \n");
    common::print_matrix<data_type, min_mn, 1, min_mn, true>(tau.data(), batches);


    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_tau));

    //=========================
    // cuBLAS reference
    //=========================
    std::vector<data_type> dummy_b;
    common::reference_cublas_geqrf_gels<data_type, cuda_data_type>(A, dummy_b, tau_ref, m, n, 1, batches, is_col_maj_a);

    // check A and tau
    // Note that if M <= N, cuSolverDx GEQRF skips computing the last reflector and tau[min(m, n) - 1] is set to 0
    // This could lead to different results from cuSolver reference for complex data type
    const auto total_relative_error_a = common::check_error<data_type, data_type>(L.data(), A.data(), batches * input_size_a);
    std::cout << "GEQRF: relative error of A between cuSolverDx and cuBLAS results: " << total_relative_error_a << std::endl;
    const auto total_relative_error_tau = common::check_error<data_type, data_type>(tau.data(), tau_ref.data(), batches * min_mn);
    std::cout << "GEQRF: relative error of tau between cuSolverDx and cuBLAS results: " << total_relative_error_tau << std::endl;

    // Comment below to remove printing results after cuSolver reference execute
    printf("Lref = \n");
    common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), batches);
    printf("tau_ref = \n");
    common::print_matrix<data_type, min_mn, 1, min_mn, true>(tau_ref.data(), batches);
    printf("=====\n");


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
struct geqrf_batched_functor {
    int operator()() { return geqrf_batched<Arch>(); }
};


int main() { return common::run_example_with_sm<geqrf_batched_functor>(); }
