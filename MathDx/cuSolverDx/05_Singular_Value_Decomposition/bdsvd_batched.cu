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

// This example demonstrates how to use cuSolverDx API to compute singular values of a batched bidiagonal matrix A.


template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block)
void kernel(DataType* d, DataType* e, typename Solver::status_type* info, const unsigned batches) {
    using namespace cusolverdx;

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    constexpr auto m = Solver::m_size;
    
    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [d_s, e_s] = shared_memory::slice<DataType, DataType>(shared_mem, alignof(DataType), m * BatchesPerBlock, alignof(DataType));

    auto this_d = d + m * batch_idx;
    auto this_e = e + (m - 1) * batch_idx;
    common::io<Solver, BatchesPerBlock>::load_a<m, 1, col_major>(this_d, m, d_s, m);
    common::io<Solver, BatchesPerBlock>::load_a<m - 1, 1, col_major>(this_e, m - 1, e_s, m - 1);

    Solver().execute(d_s, e_s, &info[batch_idx]);

    common::io<Solver, BatchesPerBlock>::store_a<m, 1, col_major>(d_s, m, this_d, m);
    common::io<Solver, BatchesPerBlock>::store_a<m - 1, 1, col_major>(e_s, m - 1, this_e, m - 1);

}

template<int Arch>
int bdsvd_batched() {

    using namespace cusolverdx;
    using Solver = decltype(Size<64>() + Precision<float>() + Type<type::real>() + Function<bdsvd>() + SM<Arch>() + Job<job::no_vectors>() + Block());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using " << bpb << " Batches per block \n";
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;

    const auto batches        = 2;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> d(m * padded_batches), d_input(m * padded_batches);
    std::vector<data_type> e((m-1) * padded_batches), e_input((m-1) * padded_batches);
    // Fill the diagonal and off-diagonal elements of the bidiagonal matrix
    common::fillup_random_matrix<data_type>(true, m, 1, d_input.data(), m, false /*symmetric*/, false /*diagonal dominant*/, 2, 4, batches);
    common::fillup_random_matrix<data_type>(true, m-1, 1, e_input.data(), m-1, false /*symmetric*/, false /*diagonal dominant*/, -5, 20, batches);

    std::vector<int> info(padded_batches, 0);
    data_type*       d_d    = nullptr;
    data_type*       d_e    = nullptr;
    int*             d_info = nullptr;

    // // Uncomment below to print A matrix
    // printf("d_input = \n");
    // common::print_matrix<data_type, m, 1, m, true>(d_input.data(), batches);
    // printf("e_input = \n");
    // common::print_matrix<data_type, m-1, 1, m-1, true>(e_input.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_d), sizeof(data_type) * d.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_e), sizeof(data_type) * e.size()));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_d, d_input.data(), sizeof(data_type) * d_input.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_e, e_input.data(), sizeof(data_type) * e_input.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    const auto shared_memory_size = Solver::shared_memory_size;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    std::cout << "shared_memory_size = " << shared_memory_size << " bytes" << std::endl;

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_d, d_e, d_info, batches);
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

    // // Uncomment below to print results after cuSolverDx execute
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
struct bdsvd_batched_functor {
    int operator()() { return bdsvd_batched<Arch>(); }
};


int main() { return common::run_example_with_sm<bdsvd_batched_functor>(); }
