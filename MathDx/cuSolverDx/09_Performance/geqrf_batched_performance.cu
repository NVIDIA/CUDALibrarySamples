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
#include "../common/measure.hpp"
#include "../common/print.hpp"
#include "../common/cublas_reference_geqrf_gels.hpp"

// This example demonstrates how to use cuSolverDx API to compute and measure performance of QR factorization on a batched m x n matrix

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void kernel(DataType* A, DataType* A_out, const int lda_gmem, DataType* tau, const unsigned batches) {

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
        alignof(DataType) // the size (number of elements) may be omitted for the last pointer
    );

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;
    auto Ag      = A + one_batch_size_a_gmem * batch_idx;
    auto A_out_g = A_out + one_batch_size_a_gmem * batch_idx;
    auto tau_g   = tau + min(m, n) * batch_idx;

    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a(Ag, lda_gmem, As, lda_smem);

    Solver().execute(As, lda_smem, tau_s);

    // store data from shared memory to global memory
    common::io<Solver, BatchesPerBlock>::store_a(As, lda_smem, A_out_g, lda_gmem);

    // store tau from shared memory to global memory
    int thread_id = threadIdx.x + Solver::block_dim.x * (threadIdx.y + Solver::block_dim.y * threadIdx.z);
    for (int i = thread_id; i < min(m, n) * BatchesPerBlock; i += Solver::max_threads_per_block) {
        tau_g[i] = tau_s[i];
    }
}

template<int Arch>
int geqrf_batched_performance() {

    using namespace cusolverdx;
    using Base   = decltype(Size<128, 32>() + Precision<float>() + Type<type::real>() + Function<geqrf>() + SM<Arch>() + Block());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;

    const auto     lda          = is_col_maj_a ? m : n;
    constexpr auto input_size_a = m * n;

    const auto batches        = 10000;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * padded_batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, 2, 4, batches); // not symmetric and not diagonally dominant

    std::vector<data_type> tau(min(m, n) * padded_batches, 0);
    data_type*             d_A     = nullptr;
    data_type*             d_A_out = nullptr;
    data_type*             d_tau   = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A_out), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_tau), sizeof(data_type) * tau.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    auto execute_solver = [&](cudaStream_t str) {
        kernel<Solver, bpb><<<batches, Solver::block_dim, Solver::shared_memory_size, str>>>(d_A, d_A_out, lda, d_tau, batches);
    };

    const int warmup_repeats = 1;
    const int kernel_repeats = 5;

    double ms = common::measure::execution(execute_solver, warmup_repeats, kernel_repeats, stream) / kernel_repeats;

    double seconds_per_giga_batch = ms / 1e3 / batches * 1e9;
    double gb_s                   = input_size_a * sizeof(data_type) * 2 / seconds_per_giga_batch;
    double gflops                 = common::get_flops_geqrf<data_type>(m, n) / seconds_per_giga_batch;

    common::print_perf("cuSolverDx-GEQRF", batches, m, n, 1, gflops, gb_s, ms, Solver::block_dim.x);

    // Compare with cublas batched GEQRF
    std::vector<data_type> dummyB;
    common::reference_cublas_geqrf_gels<data_type, cuda_data_type, false /* do solver */, true /* check blas perf */>(A, dummyB, tau, m, n, 1, batches, is_col_maj_a);
    
    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_A_out));
    CUDA_CHECK_AND_EXIT(cudaFree(d_tau));

    return 0;
}

template<int Arch>
struct geqrf_batched_performance_functor {
    int operator()() { return geqrf_batched_performance<Arch>(); }
};


int main() { return common::run_example_with_sm<geqrf_batched_performance_functor>(); }
