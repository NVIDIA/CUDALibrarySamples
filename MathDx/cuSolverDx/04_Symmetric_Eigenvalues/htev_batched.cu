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

// This example demonstrates how to use cuSolverDx API to compute eigenvalues, and optionally eigenvectors, of a batched symmetric tridiagonal matrix A.
// The results are verified by comparing A * V with V * diagm(lambda), where V is the eigenvectors and diagm(lambda) is the diagonal matrix of eigenvalues.

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block)
void kernel(DataType* d, DataType* e, DataType* V, typename Solver::status_type* info, const unsigned batches) {

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    constexpr auto m = Solver::m_size;
    
    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    // if compute_vectors is false, V_s is a dummy null pointer
    auto [d_s, e_s, V_s] =
        cusolverdx::shared_memory::slice<DataType, DataType, DataType>(shared_mem, alignof(DataType), m * BatchesPerBlock, alignof(DataType), (m - 1) * BatchesPerBlock, alignof(DataType));

    auto this_d = d + m * batch_idx;
    auto this_e = e + (m - 1) * batch_idx;
    common::io<Solver, BatchesPerBlock>::load_a<m, 1, cusolverdx::col_major>(this_d, m, d_s, m);
    common::io<Solver, BatchesPerBlock>::load_a<m - 1, 1, cusolverdx::col_major>(this_e, m - 1, e_s, m - 1);

    DataType* this_V = nullptr;
    if constexpr (Solver::job == cusolverdx::job::no_vectors) {
        Solver().execute(d_s, e_s, &info[batch_idx]);
    } else {
        this_V = V + m * m * batch_idx;
        if constexpr (Solver::job == cusolverdx::job::multiply_vectors) {
            common::io<Solver, BatchesPerBlock>::load_a<m, m, Solver::a_arrangement>(this_V, m, V_s, Solver::lda);
        }

        Solver().execute(d_s, e_s, V_s, Solver::lda, &info[batch_idx]);
    }

    common::io<Solver, BatchesPerBlock>::store_a<m, 1, cusolverdx::col_major>(d_s, m, this_d, m);
    common::io<Solver, BatchesPerBlock>::store_a<m - 1, 1, cusolverdx::col_major>(e_s, m - 1, this_e, m - 1);

    if constexpr (Solver::job != cusolverdx::job::no_vectors) {
        common::io<Solver, BatchesPerBlock>::store_a<m, m, Solver::a_arrangement>(V_s, Solver::lda, this_V, m);
    }
}

template<int Arch>
int htev_batched() {

    using namespace cusolverdx;
    using Base = decltype(Size<64>() + Precision<float>() + Type<type::real>() + Function<htev>() + SM<Arch>() + Block());
    // Job can be job::no_vectors, job::all_vectors or job::multiply_vectors
    // If job is job::all_vectors, then Vs only need to be allocated, and on exit, Vs will be overwritten with the eigenvectors of the tridiagonal matrix
    // If job is job::multiply_vectors, then Vs need to be allocated and with initial values, and on exit, Vs will be overwritten with the eigenvectors of the tridiagonal matrix multiplied by the input of Vs
    // job::multiply_vectors is useful when a symmetric matrix is reduced to a tridiagonal matrix and the householder reflectors are saved in the input of A, then the eigenvectors of the reduced matrix are needed to be multiplied back to get the eigenvectors of the original symmetric matrix
    //using Solver = decltype(Base() + Job<job::all_vectors>() + Arrangement<row_major>());
    //using Solver = decltype(Base() + Job<job::multiply_vectors>() + Arrangement<row_major>());
    using Solver = decltype(Base() + Job<job::no_vectors>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using " << bpb << " Batches per block \n";
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;

    constexpr bool compute_vectors = Solver::job != job::no_vectors;
    [[maybe_unused]] constexpr bool is_col_maj_v    = Solver::a_arrangement == arrangement::col_major; // only useful when eigenvectors are computed

    const auto batches        = 2;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> d(m * padded_batches), d_input(m * padded_batches);
    std::vector<data_type> e((m-1) * padded_batches), e_input((m-1) * padded_batches);
    // Fill the diagonal and subdiagonal elements of the tridiagonal matrix
    common::fillup_random_matrix<data_type>(true, m, 1, d_input.data(), m, false /*symmetric*/, false /*diagonal dominant*/, 2, 4, batches);
    common::fillup_random_matrix<data_type>(true, m-1, 1, e_input.data(), m-1, false /*symmetric*/, false /*diagonal dominant*/, -5, 20, batches);

    std::vector<data_type> V;
    if constexpr (compute_vectors) {
        V.resize(m * m * padded_batches, 0);
        // Set input V to be the identity matrix, if job is job::multiply_vectors
        if constexpr (job_of_v<Solver> == job::multiply_vectors) {
            for (int b = 0; b < batches; ++b) {
                for (int row = 0; row < m; ++row) {
                    V[row + row * m + b * m * m] = 1;
                }
            }
        }
    }
    std::vector<int> info(padded_batches, 0);
    data_type*       d_d    = nullptr;
    data_type*       d_e    = nullptr;
    data_type*       d_V    = nullptr;
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

    if constexpr (compute_vectors) {
        CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_V), sizeof(data_type) * V.size()));
        if constexpr (job_of_v<Solver> == job::multiply_vectors) {
            CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_V, V.data(), sizeof(data_type) * V.size(), cudaMemcpyHostToDevice, stream));
        }
    }

    const auto shared_memory_size = Solver::shared_memory_size;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    std::cout << "shared_memory_size = " << shared_memory_size << " bytes" << std::endl;

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_d, d_e, d_V, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d.data(), d_d, sizeof(data_type) * d.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));
    if constexpr (compute_vectors) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(V.data(), d_V, sizeof(data_type) * V.size(), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.begin() + batches, 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx HTEV kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }

    // // Uncomment below to print results after cuSolverDx execute
    // printf("d = \n");
    // common::print_matrix<data_type, m, 1, m, true>(d.data(), batches);
    // if constexpr (compute_vectors) {
    //     printf("V = \n");
    //     common::print_matrix<data_type, m, m, m, is_col_maj_v>(V.data(), batches);
    // }

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_d));
    CUDA_CHECK_AND_EXIT(cudaFree(d_e));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    if constexpr (compute_vectors)
        CUDA_CHECK_AND_EXIT(cudaFree(d_V));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));


    // check the computed eigenvalues and eigenvectors of the tridiagonal matrix by comparing A * V versus V * diagm(lambda)
    if constexpr (compute_vectors) {

        std::vector<data_type> A_v(m * m);
        std::vector<data_type> lambda_v(m * m);
        constexpr unsigned col_stride = is_col_maj_v ? m : 1;
        constexpr unsigned row_stride = is_col_maj_v ? 1 : m;

        for (int b = 0; b < batches; ++b) {
            data_type* A_db = d_input.data() + b * m;
            data_type* A_eb = e_input.data() + b * (m - 1);
            data_type* Vb = V.data() + b * m * m;   
            data_type* lambda_b = d.data() + b * m;

            // Compute A * V (tridiagonal matrix-matrix multiplication) and V * diagm(lambda) (scalar-vector multiplication)
            A_v.assign(m * m, 0);
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col < m; ++col) {
                    lambda_v[row * row_stride + col * col_stride] = lambda_b[col] * Vb[row * row_stride + col * col_stride];
                    
                    // Tridiagonal matrix multiplication: A[row][k] * V[k][col]
                    // For tridiagonal matrix, only non-zero elements are:
                    // - diagonal: A[row][row] = A_db[row]
                    // - subdiagonal: A[row][row-1] = A_eb[row-1] (if row > 0)
                    // - superdiagonal: A[row][row+1] = A_eb[row] (if row < m-1)
                    
                    // Diagonal element
                    A_v[row * row_stride + col * col_stride] += A_db[row] * Vb[row * row_stride + col * col_stride];
                    
                    // Subdiagonal element (if not first row)
                    if (row > 0) {
                        A_v[row * row_stride + col * col_stride] += A_eb[row - 1] * Vb[(row - 1) * row_stride + col * col_stride];
                    }
                    
                    // Superdiagonal element (if not last row)
                    if (row < m - 1) {
                        A_v[row * row_stride + col * col_stride] += A_eb[row] * Vb[(row + 1) * row_stride + col * col_stride];
                    }
                }
            }
            
            // Compare A * V with V * diagm(lambda) to verify eigenvector property
            const auto total_relative_error_lambda = common::check_error<data_type, data_type>(A_v.data(), lambda_v.data(), m * m);
            std::cout << "HTEV: relative error of A * V - V * diagm(lambda): for batch " << b << " : " << total_relative_error_lambda << std::endl;
            if (!common::is_error_acceptable<data_type>(total_relative_error_lambda)) {
                std::cout << "Relative error of A * V - V * diagm(lambda) is too large for batch " << b << std::endl;
                return 1;
            }
        }

        std::cout << "Success in verifying eigenvectors with A * V = V * diagm(lambda)" << std::endl;
    } else {
        std::cout << "Success in computing eigenvalues" << std::endl;
    }

    return 0;
}

template<int Arch>
struct htev_batched_functor {
    int operator()() { return htev_batched<Arch>(); }
};


int main() { return common::run_example_with_sm<htev_batched_functor>(); }
