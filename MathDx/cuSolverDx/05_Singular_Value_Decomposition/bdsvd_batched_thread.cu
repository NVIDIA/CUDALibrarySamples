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

// This example demonstrates how to use cuSolverDx API to compute singular values of a batched bidiagonal matrix A with Thread execution.

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ void kernel(DataType* d, DataType* e, typename Solver::status_type* info, const unsigned batches) {

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches)
        return;

    constexpr auto m = Solver::m_size;

    auto this_d = d + m * batch_idx;
    auto this_e = e + (m -1) * batch_idx;

    Solver().execute(this_d, this_e, &info[batch_idx]);
}

template<int Arch>
int bdsvd_batched_thread() {

    using namespace cusolverdx;
    using Solver = decltype(Size<25>() + Precision<float>() + Type<type::real>() + Function<bdsvd>() + SM<Arch>() + Job<job::no_vectors>() + Thread());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr auto m = Solver::m_size;
    constexpr auto e_size = m - 1; // the size of the off-diagonal elements

    const auto batches = 1000;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> d(m * batches), d_input(m * batches);
    std::vector<data_type> e(e_size * batches), e_input(e_size * batches);
    // Fill the diagonal and off-diagonal elements of the bidiagonal matrix
    common::fillup_random_matrix<data_type>(true, m, 1, d_input.data(), m, false /*symmetric*/, false /*diagonal dominant*/, 2, 4, batches);
    common::fillup_random_matrix<data_type>(true, e_size, 1, e_input.data(), e_size, false /*symmetric*/, false /*diagonal dominant*/, -5, 20, batches);

    std::vector<int> info(batches, 0);
    data_type*       d_d    = nullptr;
    data_type*       d_e    = nullptr;
    int*             d_info = nullptr;

    // Uncomment below to print A matrix
    // printf("d_input = \n");
    // common::print_matrix<data_type, m, 1, m, true>(d_input.data(), batches);
    // printf("e_input = \n");
    // common::print_matrix<data_type, m-1, 1, m-1, true>(e_input.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_d), sizeof(data_type) * d.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_e), sizeof(data_type) * e.size()));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_d, d_input.data(), sizeof(data_type) * d_input.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_e, e_input.data(), sizeof(data_type) * e_input.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    //Invokes kernel
    constexpr auto nthreads = 128;
    kernel<Solver><<<(batches + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_d, d_e, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d.data(), d_d, sizeof(data_type) * d.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(e.data(), d_e, sizeof(data_type) * e.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.begin() + batches, 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx BDSVD kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }

    // Uncomment below to print results after cuSolverDx execute
    // printf("d = \n");
    // common::print_matrix<data_type, m, 1, m, true>(d.data(), batches);

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_d));
    CUDA_CHECK_AND_EXIT(cudaFree(d_e));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    std::cout << "The largest singular value is " << d[0] << " and the smallest singular value is " << d[m - 1] << std::endl;
    std::cout << "The superdiagonal elements e[0] = " << e[0] << " and e[m-2] = " << e[m - 2] << std::endl;
    std::cout << "Success in computing singular values" << std::endl;


    return 0;
}

template<int Arch>
struct bdsvd_batched_thread_functor {
    int operator()() { return bdsvd_batched_thread<Arch>(); }
};


int main() { return common::run_example_with_sm<bdsvd_batched_thread_functor>(); }
