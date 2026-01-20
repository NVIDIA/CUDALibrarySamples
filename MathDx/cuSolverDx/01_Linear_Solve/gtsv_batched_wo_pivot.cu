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

#define CUSOLVERDX_IGNORE_NVBUG_5288270_ASSERT
#include <cusolverdx.hpp>

#include "../common/common.hpp"
#include "../common/cudart.hpp"
#include "../common/error_checking.hpp"
#include "../common/random.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/device_io.hpp"
#include "../common/measure.hpp"
#include "../common/print.hpp"
#include "../common/cusolver_reference_gtsv.hpp"


// This example demonstrates how to use cuSolverDx API to solve a batched linear tridiagonal systems with multiple right hand side after performing LU
// factorization without pivoting.  The results are compared with the reference values obtained with cuSparse host API, cuSparse<t>gtsv2StridedBatch().

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void kernel(const DataType*               dl,
                                                                        const DataType*               d,
                                                                        const DataType*               du,
                                                                        DataType*                     B,
                                                                        unsigned int                  ldb_gmem,
                                                                        typename Solver::status_type* info,
                                                                        const unsigned int            batches) {

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto k = Solver::k_size;
    static_assert(m == n, "m and n must be the same");

    const auto     one_batch_size_b_gmem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? ldb_gmem * k : n * ldb_gmem;
    constexpr auto ldb_smem              = Solver::ldb;
    constexpr auto one_batch_size_b_smem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? ldb_smem * k : n * ldb_smem;

    extern __shared__ char shared_mem[];
    // Slice shared memory into pointers
    auto [dl_s, d_s, du_s, B_s] = cusolverdx::shared_memory::slice<DataType, DataType, DataType, DataType>(shared_mem,
                                                                                                           alignof(DataType),
                                                                                                           (m - 1) * BatchesPerBlock,
                                                                                                           alignof(DataType),
                                                                                                           m * BatchesPerBlock,
                                                                                                           alignof(DataType),
                                                                                                           (m - 1) * BatchesPerBlock,
                                                                                                           alignof(DataType),
                                                                                                           one_batch_size_b_gmem * BatchesPerBlock);

    auto dl_g = dl + (m - 1) * batch_idx;
    auto d_g  = d + m * batch_idx;
    auto du_g = du + (m - 1) * batch_idx;
    auto B_g  = B + one_batch_size_b_gmem * batch_idx;

    // Load data from global memory to shared memory
    cusolverdx::copy_2d<Solver, m - 1, 1, cusolverdx::arrangement::col_major, BatchesPerBlock>(dl_g, m - 1, dl_s, m - 1);
    cusolverdx::copy_2d<Solver, m, 1, cusolverdx::arrangement::col_major, BatchesPerBlock>(d_g, m, d_s, m);
    cusolverdx::copy_2d<Solver, m - 1, 1, cusolverdx::arrangement::col_major, BatchesPerBlock>(du_g, m - 1, du_s, m - 1);

    common::io<Solver, BatchesPerBlock>::load_b(B_g, ldb_gmem, B_s, ldb_smem);

    Solver().execute(dl_s, d_s, du_s, B_s, ldb_smem, &info[batch_idx]);

    // Store results back to global memory
    common::io<Solver, BatchesPerBlock>::store_b(B_s, ldb_smem, B_g, ldb_gmem);
}

template<int Arch>
int gtsv_batched_wo_pivot() {

    using namespace cusolverdx;
    using Base   = decltype(Size<18, 18, 4>() + Precision<float>() + Type<type::complex>() + Function<gtsv_no_pivot>() + Arrangement<arrangement::col_major, row_major>() + // Arrangement of A is ignored
                          LeadingDimension<5, 19>() + // Leading dimension of A is ignored
                          SM<Arch>() + Block());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr auto m            = Solver::m_size;
    constexpr auto k            = Solver::k_size;
    const auto     ldb          = arrangement_of_v_b<Solver> == col_major ? m : k;
    const auto     input_size_b = m * k;

    const auto batches        = 4;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> dl((m - 1) * padded_batches);
    std::vector<data_type> d((m)*padded_batches);
    std::vector<data_type> du((m - 1) * padded_batches);

    // Fill up a general diagonal dominant matrix A and copy to dl, d, du
    std::vector<data_type> A(m * m * padded_batches);
    common::fillup_random_diagonal_dominant_matrix<data_type>(true /* is_col_major */, m, m, A.data(), m, false /*symmetric*/, -2, 4, batches);
    for (int b = 0; b < batches; b++) {
        for (int i = 0; i < m; i++) {
            d[b * m + i] = A[b * m * m + i * m + i];
            if (i < m - 1) {
                du[b * (m - 1) + i] = A[b * m * m + i * m + i + 1];
            }
            if (i > 0) {
                dl[b * (m - 1) + i - 1] = A[b * m * m + i * m + i - 1];
            }
        }
    }
    A.clear();
    
    std::vector<data_type> B(input_size_b * padded_batches);
    common::fillup_random_matrix<data_type>(arrangement_of_v_b<Solver> == col_major, m, k, B.data(), ldb, false, false, -2, 1, batches);
    std::vector<data_type> X(input_size_b * padded_batches);

    // Uncomment below to print dl, d, du. B
    // printf("dl = \n");
    // common::print_matrix<data_type, m - 1, 1, m, true>(dl.data(), batches);
    // printf("d = \n");
    // common::print_matrix<data_type, m, 1, m, true>(d.data(), batches);
    // printf("du = \n");
    // common::print_matrix<data_type, m-1, 1, m, true>(du.data(), batches);
    // printf("B = \n");
    // common::print_matrix<data_type, m, k, ldb, arrangement_of_v_b<Solver> == col_major>(B.data(), batches);

    std::vector<int> info(padded_batches, 0);
    data_type*       d_dl   = nullptr; /* device copy of dl */
    data_type*       d_d    = nullptr; /* device copy of d */
    data_type*       d_du   = nullptr; /* device copy of du */
    data_type*       d_B    = nullptr; /* device copy of B */
    int*             d_info = nullptr; /* error info */

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_dl), sizeof(data_type) * dl.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_d), sizeof(data_type) * d.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_du), sizeof(data_type) * du.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_dl, dl.data(), sizeof(data_type) * dl.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_d, d.data(), sizeof(data_type) * d.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_du, du.data(), sizeof(data_type) * du.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_dl, d_d, d_du, d_B, ldb, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    // Uncomment below to print X
    // printf("X after cusolverdx gtsv = \n");
    // common::print_matrix<data_type, m, k, ldb, arrangement_of_v_b<Solver> == col_major>(X.data(), batches);

    if (std::accumulate(info.begin(), info.end(), 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }

    //=========================
    // cuSolver reference
    //=========================
    common::reference_cusolver_gtsv<data_type, cuda_data_type, false /* is_perf */>(dl, d, du, B, m, k, padded_batches, (arrangement_of_v_b<Solver> == arrangement::col_major), batches);

    // Uncomment below to print X reference
    // printf("X after cusolver reference gtsv = \n");
    // common::print_matrix<data_type, m, k, ldb, arrangement_of_v_b<Solver> == col_major>(B.data(), batches);

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_dl));
    CUDA_CHECK_AND_EXIT(cudaFree(d_d));
    CUDA_CHECK_AND_EXIT(cudaFree(d_du));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));

    // check result
    auto total_relative_error = common::check_error<data_type, data_type>(X.data(), B.data(), batches * input_size_b);
    printf("GTSV: relative error of B between cuSolverDx and cuSolver results: = %e\n", total_relative_error);
    if (common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Success compared to cuSolver gtsv Result " << std::endl;
        return 0;
    } else {
        std::cout << "Failure compared to cuSolver getrs Result " << std::endl;
        return 1;
    }
    CUDA_CHECK_AND_EXIT(cudaDeviceReset());
    return 0;
}

template<int Arch>
struct gtsv_batched_wo_pivot_functor {
    int operator()() { return gtsv_batched_wo_pivot<Arch>(); }
};


int main() { return common::run_example_with_sm<gtsv_batched_wo_pivot_functor>(); }
