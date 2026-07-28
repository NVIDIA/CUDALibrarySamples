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

#include <algorithm>
#include <numeric>

#include "../common/cudart.hpp"
#include "../common/error_checking.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/random.hpp"
#include "../common/cublas_reference_hegst.hpp"

// This example demonstrates how to use cuSolverDx thread execution API to reduce a batched Hermitian/symmetric
// generalized eigenvalue problem to standard form with HEGST. B is first turned into a Cholesky factor with POTRF, then
// the cuSolverDx output is validated using a cuBLAS TRSM or TRMM reference.

template<class POTRF, typename DataType = typename POTRF::a_data_type>
__global__ void potrf_kernel(DataType* A, typename POTRF::status_type* info, unsigned batches) {
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(POTRF);

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches) {
        return;
    }

    constexpr auto m          = POTRF::m_size;
    constexpr auto batch_size = m * m;

    auto this_A = A + batch_size * batch_idx;
    POTRF().execute(this_A, &info[batch_idx]);
}

template<class HEGST, typename DataType = typename HEGST::a_data_type>
__global__ void hegst_kernel(DataType* A, const DataType* B, unsigned batches) {
    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(HEGST);

    const auto batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (batch_idx >= batches) {
        return;
    }

    constexpr auto m          = HEGST::m_size;
    constexpr auto batch_size = m * m;

    auto this_A = A + batch_size * batch_idx;
    auto this_B = B + batch_size * batch_idx;

    HEGST().execute(this_A, m, this_B, m, nullptr);
}

template<class HEGST, int Arch>
int run_hegst_case() {
    using namespace cusolverdx;

    using data_type = typename HEGST::a_data_type;

    constexpr auto     m            = HEGST::m_size;
    constexpr bool     is_col_maj_a = arrangement_of_v_a<HEGST> == arrangement::col_major;
    constexpr bool     is_col_maj_b = arrangement_of_v_b<HEGST> == arrangement::col_major;
    constexpr unsigned batch_size   = m * m;
    const unsigned     batches      = 250;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(batch_size * batches);
    std::vector<data_type> B(batch_size * batches);

    // The input matrix could be non-symmetric as only the specified part of the matrix is processed by HEGST.
    common::fillup_random_matrix<data_type>(
        is_col_maj_a, m, m, A.data(), m, false /*symmetric*/, false /*diagonal dominant*/, -2, 4, batches);
    common::fillup_random_matrix<data_type>(
        is_col_maj_b, m, m, B.data(), m, false /*symmetric*/, true /*diagonal dominant*/, -2, 1, batches);

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

    constexpr unsigned nthreads = 128;

    // Prepare input B with Cholesky factorization
    using POTRF =
        decltype(Size<HEGST::m_size>() + Precision<typename HEGST::a_precision>() + Type<HEGST::type>() + Function<potrf>() +
                 FillMode<HEGST::fill_mode>() + Arrangement<HEGST::b_arrangement>() + SM<Arch>() + Thread());

    potrf_kernel<POTRF><<<(batches + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_B, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(B.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.end(), 0) != 0) {
        std::cout << "POTRF failed" << '\n';
        for (unsigned j = 0; j < batches; ++j) {
            if (info[j] != 0) {
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
            }
        }
        return -1;
    }

    // HEGST thread execution does not require workspace
    hegst_kernel<HEGST><<<(batches + nthreads - 1) / nthreads, nthreads, 0, stream>>>(d_A, d_B, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    std::vector<data_type> A_reference;
    common::cublas_reference_hegst<eig_type_of_v<HEGST>,
                                   fill_mode_of_v<HEGST>,
                                   arrangement_of_v_a<HEGST>,
                                   arrangement_of_v_b<HEGST>>(A_input, B, A_reference, m, batches);

    double total_relative_error = 0.0;
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
        std::cout << "Failure compared with cuBLAS reference results" << std::endl;
        return 1;
    }
    return 0;
}

template<int Arch>
struct hegst_batched_thread_functor {
    int operator()() {
        using namespace cusolverdx;
        using HEGST =
            decltype(Size<8>() + Precision<float>() + Type<type::complex>() + Function<hegst>() + FillMode<fill_mode::lower>() +
                     Arrangement<arrangement::col_major, arrangement::row_major>() + EigType<1>() + SM<Arch>() + Thread());

        const int status = run_hegst_case<HEGST, Arch>();
        if (status == 0) {
            std::cout << "Success comparing HEGST reduced matrix to cuBLAS TRMM/TRSM reference" << std::endl;
        }
        return status;
    }
};

int main() {
    return common::run_example_with_sm<hegst_batched_thread_functor>();
}
