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
#include <cusolverdx_io.hpp>

#include <algorithm>
#include <numeric>

#include "../common/cudart.hpp"
#include "../common/device_io.hpp"
#include "../common/error_checking.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/random.hpp"
#include "../common/cublas_reference_hegst.hpp"

// This example demonstrates how to use cuSolverDx block execution API to reduce a batched Hermitian/symmetric
// generalized eigenvalue problem to standard form with HEGST.

template<class POTRF, class HEGST, typename DataType = typename HEGST::a_data_type>
__global__ __launch_bounds__(HEGST::max_threads_per_block) void potrf_hegst_kernel(DataType*                    A,
                                                                                   const int                    lda_gmem,
                                                                                   DataType*                    B,
                                                                                   const int                    ldb_gmem,
                                                                                   typename POTRF::status_type* info,
                                                                                   const unsigned               batches) {
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(HEGST);

    static_assert(HEGST::batches_per_block == 1 && POTRF::batches_per_block == 1,
                  "This fused example intentionally uses one batch per block");

    constexpr auto m                     = HEGST::m_size;
    constexpr auto lda_smem              = HEGST::lda;
    constexpr auto ldb_smem              = HEGST::ldb;
    constexpr auto workspace_size        = HEGST::workspace_size;
    constexpr auto one_batch_size_a_smem = m * lda_smem;
    constexpr auto one_batch_size_b_smem = m * ldb_smem;

    const auto batch_idx = blockIdx.x;
    if (batch_idx >= batches) {
        return;
    }

    extern __shared__ __align__(16) cusolverdx::byte shared_mem[];
    auto [A_s, B_s, workspace_s] = cusolverdx::shared_memory::slice<DataType, DataType, DataType>(shared_mem,
                                                                                                  alignof(DataType),
                                                                                                  one_batch_size_a_smem,
                                                                                                  alignof(DataType),
                                                                                                  one_batch_size_b_smem,
                                                                                                  alignof(DataType),
                                                                                                  workspace_size);

    auto A_g = A + batch_idx * lda_gmem * m;
    auto B_g = B + batch_idx * ldb_gmem * m;

    common::io<HEGST>::load_b<m, m>(B_g, ldb_gmem, B_s, ldb_smem);
    __shared__ typename POTRF::status_type potrf_info;
    POTRF().execute(B_s, ldb_smem, &potrf_info);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        info[batch_idx] = potrf_info;
    }

    if (potrf_info == 0) {
        common::io<HEGST>::load_a<m, m>(A_g, lda_gmem, A_s, lda_smem);

        // HEGST block execution requires a workspace pointer, and HEGST::shared_memory_size already accounts for HEGST::workspace_size.
        HEGST().execute(A_s, lda_smem, B_s, ldb_smem, workspace_s);

        common::io<HEGST>::store_a<m, m>(A_s, lda_smem, A_g, lda_gmem);
    }

    common::io<HEGST>::store_b<m, m>(B_s, ldb_smem, B_g, ldb_gmem);
}

template<int Arch>
int hegst_batched_block() {
    using namespace cusolverdx;

    using HEGST =
        decltype(Size<32>() + Precision<float>() + Type<type::complex>() + Function<hegst>() + FillMode<fill_mode::upper>() +
                 Arrangement<arrangement::row_major, arrangement::col_major>() + EigType<3>() + SM<Arch>() + Block());
    using POTRF = decltype(Size<HEGST::m_size>() + Precision<typename HEGST::a_precision>() + Type<HEGST::type>() +
                           Function<potrf>() + FillMode<HEGST::fill_mode>() + Arrangement<HEGST::b_arrangement>() + SM<Arch>() +
                           BlockDim<HEGST::block_dim.x, HEGST::block_dim.y, HEGST::block_dim.z>() +
                           BatchesPerBlock<HEGST::batches_per_block>() + Block());

    using data_type = typename HEGST::a_data_type;

    constexpr auto     m            = HEGST::m_size;
    constexpr bool     is_col_maj_a = arrangement_of_v_a<HEGST> == arrangement::col_major;
    constexpr bool     is_col_maj_b = arrangement_of_v_b<HEGST> == arrangement::col_major;
    constexpr unsigned lda_gmem     = m;
    constexpr unsigned ldb_gmem     = m;
    constexpr unsigned batch_size   = m * m;
    constexpr unsigned batches      = 200;

    std::cout << "Suggested BlockDim = " << HEGST::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << HEGST::block_dim.x << std::endl;
    std::cout << "HEGST workspace size = " << HEGST::workspace_size << std::endl;
    std::cout << "HEGST shared memory size = " << HEGST::shared_memory_size << std::endl;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(batch_size * batches);
    std::vector<data_type> B(batch_size * batches);

    // The input matrix could be non-symmetric as only the specified part of the matrix is processed by HEGST.
    common::fillup_random_matrix<data_type>(
        is_col_maj_a, m, m, A.data(), lda_gmem, false /*symmetric*/, false /*diagonal dominant*/, -2, 4, batches);
    common::fillup_random_matrix<data_type>(
        is_col_maj_b, m, m, B.data(), ldb_gmem, false /*symmetric*/, true /*diagonal dominant*/, -2, 1, batches);

    auto A_input = A;

    data_type*       d_A    = nullptr;
    data_type*       d_B    = nullptr;
    int*             d_info = nullptr;
    std::vector<int> info(batches, 0);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        potrf_hegst_kernel<POTRF, HEGST>, cudaFuncAttributeMaxDynamicSharedMemorySize, HEGST::shared_memory_size));

    potrf_hegst_kernel<POTRF, HEGST>
        <<<batches, HEGST::block_dim, HEGST::shared_memory_size, stream>>>(d_A, lda_gmem, d_B, ldb_gmem, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.end(), 0) != 0) {
        std::cout << "POTRF failed" << std::endl;
        for (unsigned j = 0; j < batches; ++j) {
            if (info[j] != 0) {
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
            }
        }
        return -1;
    }

    std::vector<data_type> A_reference;
    common::cublas_reference_hegst<eig_type_of_v<HEGST>,
                                   fill_mode_of_v<HEGST>,
                                   arrangement_of_v_a<HEGST>,
                                   arrangement_of_v_b<HEGST>>(A_input, B, A_reference, m, batches);

    double     total_relative_error = 0.0;
    const auto dx_active = common::extract_active_triangle<fill_mode_of_v<HEGST>, arrangement_of_v_a<HEGST>>(A, m, batches);
    const auto reference_active =
        common::extract_active_triangle<fill_mode_of_v<HEGST>, arrangement_of_v_a<HEGST>>(A_reference, m, batches);
    total_relative_error =
        common::check_error<data_type, data_type>(dx_active.data(), reference_active.data(), dx_active.size());

    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    std::cout << "HEGST: relative error compared with host reference: " << total_relative_error << std::endl;
    if (!common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Failure compared with host reference results" << std::endl;
        return 1;
    }

    std::cout << "Success" << std::endl;
    return 0;
}

template<int Arch>
struct hegst_batched_block_functor {
    int operator()() { return hegst_batched_block<Arch>(); }
};

int main() {
    return common::run_example_with_sm<hegst_batched_block_functor>();
}
