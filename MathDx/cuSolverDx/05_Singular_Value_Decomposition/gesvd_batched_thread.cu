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
#include "../common/cusolver_reference_gesvd.hpp"

// This example demonstrates how to use cuSolverDx API to compute singular values of a batched m x n general matrix A with Thread execution.
// The results are compared with the reference values obtained with cuSolver host API, cusolverDnXgesvdjBatched and cusolverDnXgesvd API.

template<class Solver, typename DataType = typename Solver::a_data_type, typename PrecisionType = typename Solver::a_precision>
__global__ void kernel(DataType* A, PrecisionType* sigma, DataType* workspace, typename Solver::status_type* info, const unsigned batches) {

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches)
        return;

    constexpr auto m                     = Solver::m_size;
    constexpr auto n                     = Solver::n_size;
    constexpr auto one_batch_size_a_gmem = m * n;
    constexpr auto workspace_size        = Solver::workspace_size;

    auto A_g         = A + one_batch_size_a_gmem * batch_idx;
    auto sigma_g     = sigma + n * batch_idx;
    auto workspace_g = workspace + workspace_size * batch_idx;

    Solver().execute(A_g, sigma_g, workspace_g, &info[batch_idx]);
}

template<int Arch>
int gesvd_batched_thread() {

    using namespace cusolverdx;
    using Solver = decltype(Size<12, 5>() + Precision<float>() + Type<type::real>() + Function<gesvd>() + Arrangement<arrangement::row_major>() + SM<Arch>() +
                            Thread());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
    using precision_type = typename Solver::a_precision;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    static_assert(m >= n, "m must be greater than or equal to n for gesvd");

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;
    const auto     lda_gmem     = is_col_maj_a ? m : n;
    constexpr auto input_size_a = m * n;

    const auto batches = 500;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda_gmem, false /*symmetric*/, false /*diagonal dominant*/, -2, 4, batches);

    std::vector<precision_type> S(n * batches, 0);
    std::vector<precision_type> S_ref(n * batches, 0);
    std::vector<data_type>      workspace(Solver::workspace_size * batches, 0);
    std::vector<int>            info(batches, 0);
    data_type*                  d_A         = nullptr;
    precision_type*             d_S         = nullptr;
    data_type*                  d_workspace = nullptr;
    int*                        d_info      = nullptr;

    // // Uncomment below to print A matrix
    // printf("A = \n");
    // common::print_matrix<data_type, m, m, lda_gmem, is_col_maj_a>(A.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_S), sizeof(precision_type) * S.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_workspace), sizeof(data_type) * Solver::workspace_size * batches));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    //Invokes kernel
    constexpr auto nthreads = 64;
    kernel<Solver><<<(batches + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_A, d_S, d_workspace, d_info, batches);
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
        std::vector<data_type> dummy;
        common::reference_cusolver_gesvd<data_type, cuda_data_type, precision_type, false /*check_perf*/>(
                A, S_ref, dummy, dummy, 'N', 'N', info.data(), m, n, batches, is_col_maj_a, batches);

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
        std::cout << "GESVD: m < n is not supported in cuSolverDn<t>getsvd API. Checking with cuSolverDn reference is disabled. Successfully executed with "
                     "cuSolverDx GESVD API."
                  << std::endl;
    }
    return 0;
}

template<int Arch>
struct gesvd_batched_thread_functor {
    int operator()() { return gesvd_batched_thread<Arch>(); }
};


int main() { return common::run_example_with_sm<gesvd_batched_thread_functor>(); }
