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
#include "../common/cusolver_reference_gesvd.hpp"

// This example demonstrates how to use cuSolverDx API to compute singular values of a batched m x n general matrix A.
// The results are compared with the reference values obtained with cuSolver host API, cusolverDnXgesvdjBatched and cusolverDnXgesvd API.

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type, typename PrecisionType = typename Solver::a_precision>
__global__ __launch_bounds__(Solver::max_threads_per_block)
void kernel(DataType* A, const int lda_gmem, PrecisionType* sigma, DataType* workspace, typename Solver::status_type* info, const unsigned batches) {

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    constexpr auto m                     = Solver::m_size;
    constexpr auto n                     = Solver::n_size;
    constexpr auto one_batch_size_a_gmem = m * n;
    constexpr auto lda_smem              = Solver::lda;
    const auto     one_batch_size_a_smem = lda_smem * (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major ? n : m);
    constexpr auto workspace_size        = Solver::workspace_size;

    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [A_s, sigma_s, workspace_s] =
        cusolverdx::shared_memory::slice<DataType, PrecisionType, DataType>(shared_mem,
                                                                            alignof(DataType),
                                                                            one_batch_size_a_smem * BatchesPerBlock,
                                                                            alignof(PrecisionType),
                                                                            n * BatchesPerBlock, 
                                                                            alignof(DataType) // the size can be omitted for the last pointer
        );

    auto A_g      = A + one_batch_size_a_gmem * batch_idx;
    auto sigma_g = sigma + n * batch_idx;

    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a<m, n>(A_g, lda_gmem, A_s, lda_smem);

    Solver().execute(A_s, lda_smem, sigma_s, workspace_s, &info[batch_idx]);

    // Store results back to global memory
    cusolverdx::copy_2d<Solver, n, 1, cusolverdx::arrangement::col_major, BatchesPerBlock, PrecisionType>(sigma_s, n, sigma_g, n);

}

template<int Arch>
int gesvd_batched() {

    using namespace cusolverdx;
    using Base = decltype(Size<16, 10>() + Precision<float>() + Type<type::real>() + Function<gesvd>() + Arrangement<arrangement::row_major>() + SM<Arch>() + Block());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
    using precision_type = typename Solver::a_precision;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    static_assert(m >= n, "m must be greater than or equal to n for gesvd");

    constexpr bool is_col_maj_a  = arrangement_of_v_a<Solver> == arrangement::col_major;
    const auto     lda_gmem     = is_col_maj_a ? m : n;
    constexpr auto input_size_a = m * n;

    const auto batches        = 2;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * padded_batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda_gmem, false /*symmetric*/, false /*diagonal dominant*/, -2, 4, batches);

    std::vector<precision_type> S(n * padded_batches, 0);
    std::vector<precision_type> S_ref(n * padded_batches, 0);
    std::vector<data_type>      workspace(Solver::workspace_size * padded_batches, 0);
    std::vector<int>            info(padded_batches, 0);
    data_type*                  d_A         = nullptr;
    precision_type*             d_S         = nullptr;
    data_type*                  d_workspace = nullptr;
    int*                        d_info      = nullptr;

    // // Uncomment below to print A matrix
    // printf("A = \n");
    // common::print_matrix<data_type, m, m, lda_gmem, is_col_maj_a>(A.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_S), sizeof(precision_type) * S.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_workspace), sizeof(data_type) * Solver::workspace_size));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda_gmem, d_S, d_workspace, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(S.data(), d_S, sizeof(precision_type) * S.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.begin() + batches, 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx GESVD kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }

    // Uncomment below to print results after cuSolverDx execute
    // printf("=====\n");
    // printf("lambda = \n");
    // common::print_matrix<precision_type, n, 1, n, true>(S.data(), batches);
    // }

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_S));
    CUDA_CHECK_AND_EXIT(cudaFree(d_workspace));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    //=========================
    // cuSolver reference
    //=========================
    if constexpr (m >= n) {
        common::reference_cusolver_gesvd<data_type, cuda_data_type, precision_type, false /*check_perf*/>(A, S_ref, info.data(), m, n, padded_batches, is_col_maj_a, batches);

        const auto total_relative_error_S = common::check_error<precision_type, precision_type>(S.data(), S_ref.data(), batches * n);
        std::cout << "GESVD: relative error of S between cuSolverDx and cuSolver reference results: " << total_relative_error_S << std::endl;

        // // Uncomment below to print results after cuSolver reference execute
        // printf("S_ref = \n");
        // common::print_matrix<precision_type, n, 1, n, true>(S_ref.data(), batches);
        // printf("=====\n");

        if (!common::is_error_acceptable<precision_type>(total_relative_error_S)) {
            std::cout << "Failure compared with cuSolver API results S" << std::endl;
            //Print out S for debugging. Do not delete
            for (int i = 0; i < n * batches; ++i) {
                if (abs(S[i] - S_ref[i]) / abs(S_ref[i]) > 1e-05) {
                    printf("S[%d] = %10.3f, S_ref[%d] = %10.3f  differ \n", i, S[i], i, S_ref[i]);
                }
            }
            return 1;
        }

        std::cout << "Success compared singular values with the cuSolverDn reference results" << std::endl;
    } else {
        std::cout << "GESVD: m < n is not supported in cuSolverDn<t>getsvd API. Checking with cuSolverDn reference is disabled. Successfully executed with cuSolverDx GESVD API." << std::endl;
    }
    return 0;
}

template<int Arch>
struct gesvd_batched_functor {
    int operator()() { return gesvd_batched<Arch>(); }
};


int main() { return common::run_example_with_sm<gesvd_batched_functor>(); }
