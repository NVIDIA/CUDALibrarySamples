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

// This example demonstrates how to use cuSolverDx API to compute singular values, and optionally singular vectors, of a batched bidiagonal matrix A with Block execution.
// The results are verified by comparing A with U * \Sigma * VT, where U and VT are the left and right singular vectors of A.


template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void kernel(DataType*                     d,
                                                                        DataType*                     e,
                                                                        DataType*                     U,
                                                                        DataType*                     VT,
                                                                        typename Solver::status_type* info,
                                                                        const unsigned                batches) {
    using namespace cusolverdx;

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    constexpr auto m = Solver::m_size;
    constexpr auto e_size = m - 1;
    constexpr auto ldu = Solver::lda;
    constexpr auto ldvt = Solver::ldb;

    extern __shared__ __align__(16) cusolverdx::byte shared_mem[];
    // Slice shared memory into pointers
    auto [d_s, e_s, U_s, VT_s] = shared_memory::slice<DataType, DataType, DataType, DataType>(shared_mem,
                                                                                              alignof(DataType),
                                                                                              m * BatchesPerBlock,
                                                                                              alignof(DataType),
                                                                                              e_size * BatchesPerBlock,
                                                                                              alignof(DataType),
                                                                                              ldu * m * BatchesPerBlock,
                                                                                              alignof(DataType));

    auto this_d = d + m * batch_idx;
    auto this_e = e + (e_size) * batch_idx;
    common::io<Solver, BatchesPerBlock>::load_a<m, 1, col_major>(this_d, m, d_s, m);
    common::io<Solver, BatchesPerBlock>::load_a<e_size, 1, col_major>(this_e, e_size, e_s, e_size);

    DataType* this_U = nullptr;
    DataType* this_VT = nullptr;
    if constexpr (Solver::jobu != job::no_vectors) {
        this_U = U + m * m * batch_idx;
        common::io<Solver, BatchesPerBlock>::load_a<m, m, Solver::a_arrangement>(this_U, m, U_s, ldu);
    }
    if constexpr (Solver::jobvt != job::no_vectors) {
        this_VT = VT + m * m * batch_idx;
        common::io<Solver, BatchesPerBlock>::load_a<m, m, Solver::b_arrangement>(this_VT, m, VT_s, ldvt);
    }

    Solver().execute(d_s, e_s, U_s, ldu, VT_s, ldvt, &info[batch_idx]);

    common::io<Solver, BatchesPerBlock>::store_a<m, 1, col_major>(d_s, m, this_d, m);
    common::io<Solver, BatchesPerBlock>::store_a<e_size, 1, col_major>(e_s, e_size, this_e, e_size);
    if constexpr (Solver::jobu != job::no_vectors) {
        common::io<Solver, BatchesPerBlock>::store_a<m, m, Solver::a_arrangement>(U_s, ldu, this_U, m);
    }
    if constexpr (Solver::jobvt != job::no_vectors) {
        common::io<Solver, BatchesPerBlock>::store_a<m, m, Solver::b_arrangement>(VT_s, ldvt, this_VT, m);
    }
}

template<int Arch>
int bdsvd_batched_block() {

    using namespace cusolverdx;
    using Base = decltype(Size<30>() + Precision<float>() + Type<type::real>() + Function<bdsvd>() + SM<Arch>() + Block()); 

    // Job option for bdsvd can be no_vectors, all_vectors/some_vectors or multiply_vectors for either left or right singular vectors
    // job::no_vectors: compute singular values only
    // using Solver = decltype(Base() + Job<job::no_vectors, job::no_vectors>());
    //
    // If job is job::all_vectors or job::some_vectors for either left or right singular vectors, then U and VT need to be allocated
    //    on exit, U and VT will be overwritten with the left and right singular vectors of the bidiagonal matrix
    using Solver = decltype(Base() + Job<job::all_vectors, job::all_vectors>() + Arrangement<col_major, row_major>());
    //
    // If job is job::multiply_vectors, then U and VT need to be allocated and with initial values
    //    on exit, U and VT will be overwritten with the left and right singular vectors of the bidiagonal matrix multiplied by the input of U and VT
    // using Solver = decltype(Base() + Job<job::multiply_vectors, job::multiply_vectors>() + Arrangement<col_major, row_major>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using " << bpb << " Batches per block \n";
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m      = Solver::m_size; // the size of the diagonal elements
    constexpr auto e_size = m - 1;         // the size of the off-diagonal elements

    constexpr bool                  compute_vectors_u = Solver::jobu != job::no_vectors;
    constexpr bool                  compute_vectors_vt = Solver::jobvt != job::no_vectors;
    [[maybe_unused]] constexpr bool is_col_maj_u    = Solver::a_arrangement == arrangement::col_major; // only useful when left singular vectors are computed
    [[maybe_unused]] constexpr bool is_col_maj_vt   = Solver::b_arrangement == arrangement::col_major; // only useful when right singular vectors are computed

    const auto batches        = 2;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> d(m * padded_batches), d_input(m * padded_batches);
    std::vector<data_type> e(e_size * padded_batches), e_input(e_size * padded_batches);
    // Fill the diagonal and off-diagonal elements of the bidiagonal matrix
    common::fillup_random_matrix<data_type>(true, m, 1, d_input.data(), m, false /*symmetric*/, false /*diagonal dominant*/, 2, 4, batches);
    common::fillup_random_matrix<data_type>(true, e_size, 1, e_input.data(), e_size, false /*symmetric*/, false /*diagonal dominant*/, -5, 20, batches);

    std::vector<data_type> U, VT;
    if constexpr (compute_vectors_u) {
        U.resize(m * m * padded_batches, 0);
        // Set input U to be the identity matrix, if job is job::multiply_vectors
        if constexpr (jobu_of_v<Solver> == job::multiply_vectors) {
            for (int b = 0; b < batches; ++b) {
                for (int row = 0; row < m; ++row) {
                    U[row + row * m + b * m * m] = 1;
                }
            }
        }
    }
    if constexpr (compute_vectors_vt) {
        VT.resize(m * m * padded_batches, 0);
        // Set input VT to be the identity matrix, if job is job::multiply_vectors
        if constexpr (jobvt_of_v<Solver> == job::multiply_vectors) {
            for (int b = 0; b < batches; ++b) {
                for (int row = 0; row < m; ++row) {
                    VT[row + row * m + b * m * m] = 1;
                }
            }
        }
    }

    std::vector<int> info(padded_batches, 0);
    data_type*       d_d    = nullptr;
    data_type*       d_e    = nullptr;
    int*             d_info = nullptr;

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_d), sizeof(data_type) * d.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_e), sizeof(data_type) * e.size()));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_d, d_input.data(), sizeof(data_type) * d_input.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_e, e_input.data(), sizeof(data_type) * e_input.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));


    data_type* d_U = nullptr;
    data_type* d_VT = nullptr;

    if constexpr (compute_vectors_u) {
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_U), sizeof(data_type) * U.size()));
        if constexpr (jobu_of_v<Solver> == job::multiply_vectors) {
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_U, U.data(), sizeof(data_type) * U.size(), cudaMemcpyHostToDevice, stream));
        }
    }
    if constexpr (compute_vectors_vt) {
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_VT), sizeof(data_type) * VT.size()));
        if constexpr (jobvt_of_v<Solver> == job::multiply_vectors) {
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_VT, VT.data(), sizeof(data_type) * VT.size(), cudaMemcpyHostToDevice, stream));
        }
    }

    const auto shared_memory_size = Solver::shared_memory_size;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    std::cout << "shared_memory_size = " << shared_memory_size << " bytes" << std::endl;

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_d, d_e, d_U, d_VT, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d.data(), d_d, sizeof(data_type) * d.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(e.data(), d_e, sizeof(data_type) * e.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));

    if constexpr (compute_vectors_u) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(U.data(), d_U, sizeof(data_type) * U.size(), cudaMemcpyDeviceToHost, stream));
    }
    if constexpr (compute_vectors_vt) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(VT.data(), d_VT, sizeof(data_type) * VT.size(), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.begin() + batches, 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx BDSVD kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_d));
    CUDA_CHECK_AND_EXIT(cudaFree(d_e));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    if constexpr (compute_vectors_u) {
        CUDA_CHECK_AND_EXIT(cudaFree(d_U));
    }
    if constexpr (compute_vectors_vt) {
        CUDA_CHECK_AND_EXIT(cudaFree(d_VT));
    }
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));


    std::cout << "The largest singular value is " << d[0] << " and the smallest singular value is " << d[e_size] << std::endl;
    std::cout << "The superdiagonal elements e[0] = " << e[0] << " and e[m-2] = " << e[m - 2] << std::endl;
    if constexpr (compute_vectors_u && compute_vectors_vt) {
        // Verify the computed singular values and vectors
        std::vector<double> A_expected(m * m);
        std::vector<double> A_reconstructed(m * m);
        // U and VT could be either col-major or row-major, so for convenience transpose them if they are row-major
        if constexpr (!is_col_maj_u) {
            common::transpose_matrix<data_type>(U, m, m, batches);
        }
        if constexpr (!is_col_maj_vt) {
            common::transpose_matrix<data_type>(VT, m, m, batches);
        }

        
        for (int b = 0; b < batches; ++b) {
            // output sigma is stored in d
            const auto* sigma_b = d.data() + b * m;
            const auto* U_b = U.data() + b * m * m;
            const auto* VT_b = VT.data() + b * m * m;
            const auto* d_input_b = d_input.data() + b * m;
            const auto* e_input_b = e_input.data() + b * e_size;

            // Compare A_expected and A_reconstructed = U * \Sigma * VT
            A_expected.assign(m * m, 0.0);
            A_reconstructed.assign(m * m, 0.0);
            for (int i = 0; i < m; i += 1) {
                for (int j = 0; j < m; j += 1) {

                    for (int l = 0; l < m; ++l) {
                        A_reconstructed[i + j * m] += double(sigma_b[l]) * double(U_b[i + l * m]) * double(VT_b[l + j * m]);
                    }
                    A_expected[i + j * m] = (i == j) ? double(d_input_b[i]) : (i == j - 1) ? double(e_input_b[i]) : 0.0;
                }
            }
            const auto total_relative_error = common::check_error<double, double>(A_reconstructed.data(), A_expected.data(), m * m);
            std::cout << "BDSVD: relative error of A - U Sigma VT: for batch " << b << " : " << total_relative_error << std::endl;
            if (!common::is_error_acceptable<data_type>(total_relative_error)) {
                printf("A_reconstructed = \n");
                common::print_matrix(m, m, true, A_reconstructed.data(), m);
                printf("A_expected = \n");
                common::print_matrix(m, m, true, A_expected.data(), m);
                std::cout << "Relative error of A - U Sigma VT is too large for batch " << b << std::endl;
                return 1;
            }
        }
        std::cout << "Success in verifying singular values and vectors" << std::endl;
    } else {
        std::cout << "Success " << std::endl;
    }

    return 0;
}

template<int Arch>
struct bdsvd_batched_block_functor {
    int operator()() { return bdsvd_batched_block<Arch>(); }
};


int main() {
    return common::run_example_with_sm<bdsvd_batched_block_functor>();
}
