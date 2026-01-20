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
#include "../common/cusolver_reference_heev.hpp"

// This example demonstrates how to use cuSolverDx API to compute eigenvalues, and optionally eigenvectors, of a batched m x m symmetric matrix A.
// The results are compared with the reference values obtained with cuSolver host API, cusolverDnXsyevBatched and cusolverDn<t>syevjBatched.

template<class Solver, unsigned int BatchesPerBlock, bool compute_vectors = false, typename DataType = typename Solver::a_data_type, typename PrecisionType = typename Solver::a_precision>
__global__ __launch_bounds__(Solver::max_threads_per_block)
void kernel(DataType* A, const int lda_gmem, PrecisionType* lambda, DataType* workspace, typename Solver::status_type* info, const unsigned batches) {

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    constexpr auto m                     = Solver::m_size;
    constexpr auto one_batch_size_a_gmem = m * m;
    constexpr auto lda_smem              = Solver::lda;
    const auto     one_batch_size_a_smem = lda_smem * m;
    constexpr auto workspace_size        = Solver::workspace_size;

    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [A_s, lambda_s, workspace_s] =
        cusolverdx::shared_memory::slice<DataType, PrecisionType, DataType>(shared_mem,
                                                                            alignof(DataType),
                                                                            one_batch_size_a_smem * BatchesPerBlock,
                                                                            alignof(PrecisionType),
                                                                            m * BatchesPerBlock, // lambda
                                                                            alignof(DataType)    // workspace
        );

    auto A_g      = A + one_batch_size_a_gmem * batch_idx;
    auto lambda_g = lambda + m * batch_idx;

    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a<m, m>(A_g, m, A_s, lda_smem);

    Solver().execute(A_s, lda_smem, lambda_s, workspace_s, &info[batch_idx]);

    // Store results back to global memory
    cusolverdx::copy_2d<Solver, m, 1, cusolverdx::arrangement::col_major, BatchesPerBlock, PrecisionType>(lambda_s, m, lambda_g, m);
    if constexpr (compute_vectors) {
        common::io<Solver, BatchesPerBlock>::store_a<m, m, Solver::a_arrangement>(A_s, lda_smem, A_g, m);
    }
}

template<int Arch>
int heev_batched() {

    using namespace cusolverdx;
    // Job can be either job::overwrite_vectors or job::no_vectors
    using Base = decltype(Size<16>() + Precision<float>() + Type<type::complex>() + Function<heev>() + FillMode<fill_mode::upper>() + Arrangement<arrangement::row_major>() + Job<job::overwrite_vectors>() + SM<Arch>() + Block());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
    using precision_type = typename Solver::a_precision;

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;

    constexpr bool is_col_maj_a  = arrangement_of_v_a<Solver> == arrangement::col_major;
    constexpr bool is_lower_fill = fill_mode_of_v<Solver> == fill_mode::lower;
    constexpr bool compute_vectors = job_of_v<Solver> != job::no_vectors;

    const auto     lda_gmem     = m;
    constexpr auto input_size_a = m * m;

    const auto batches        = 5;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * padded_batches);
    // Fill with symmetric matrix data
    // If the input matrix is not symmetric, it is OK because the function only processes the specified part of the matrix and leaves the rest unchanged
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, m, A.data(), lda_gmem, true /*symmetric*/, false /*diagonal dominant*/, -2, 4, batches);
    
    std::vector<data_type> V, A_input;
    if constexpr (compute_vectors) {
        A_input.assign(A.begin(), A.end());
        V.resize(input_size_a * padded_batches);
    }
    std::vector<precision_type> lambda(m * padded_batches, 0);
    std::vector<precision_type> lambda_ref(m * padded_batches, 0);
    std::vector<int>            info(padded_batches, 0);
    data_type*                  d_A         = nullptr;
    precision_type*             d_lambda    = nullptr;
    data_type*                  d_workspace = nullptr;
    int*                        d_info      = nullptr;

    // // Uncomment below to print A matrix
    // printf("A = \n");
    // common::print_matrix<data_type, m, m, lda_gmem, is_col_maj_a>(A.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_lambda), sizeof(precision_type) * lambda.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_workspace), sizeof(data_type) * Solver::workspace_size));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb, compute_vectors>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver, bpb, compute_vectors><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda_gmem, d_lambda, d_workspace, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(lambda.data(), d_lambda, sizeof(precision_type) * lambda.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));
    if constexpr (compute_vectors) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(V.data(), d_A, sizeof(data_type) * V.size(), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.begin() + batches, 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx HEEV kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }

    // Uncomment below to print results after cuSolverDx execute
    // printf("=====\n");
    // printf("lambda = \n");
    // common::print_matrix<precision_type, m, 1, m, true>(lambda.data(), batches);
    // if constexpr (compute_vectors) {
    //     printf("V = \n");
    //     common::print_matrix<data_type, m, m, m, is_col_maj_a>(V.data(), batches);
    // }

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_lambda));
    CUDA_CHECK_AND_EXIT(cudaFree(d_workspace));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    //=========================
    // cuSolver reference
    //=========================
    common::reference_cusolver_heev<data_type, cuda_data_type, precision_type, false /*use_syevj*/, false /*check_perf*/>(
        A, lambda_ref, info.data(), m, padded_batches, is_lower_fill, is_col_maj_a, compute_vectors, batches);

    const auto total_relative_error_lambda = common::check_error<precision_type, precision_type>(lambda.data(), lambda_ref.data(), batches * m);
    std::cout << "HEEV: relative error of lambda between cuSolverDx and cuSolver reference results: " << total_relative_error_lambda << std::endl;

    // Uncomment below to print results after cuSolver reference execute
    // printf("lambda_ref = \n");
    // common::print_matrix<precision_type, m, 1, m, true>(lambda_ref.data(), batches);
    // printf("=====\n");

    if (!common::is_error_acceptable<precision_type>(total_relative_error_lambda)) {
        std::cout << "Failure compared with cuSolver API results lambda" << std::endl;
        //Print out lambda for debugging
        for (int i = 0; i < m * batches; ++i) {
            if (abs(lambda[i] - lambda_ref[i]) / abs(lambda_ref[i]) > 1e-05) {
                printf("lambda[%d] = %10.3f, lambda_ref[%d] = %10.3f  differ \n", i, lambda[i], i, lambda_ref[i]);
            }
        }
        return 1;
    }
    
    // check eigenvectors. As the eigenvectors are not unique, instead of comparing the eigenvectors directly with cuSolver reference, we compare A * V versus V * diagm(lambda)
    if constexpr (compute_vectors) {
        // cuSolverDx and cuSolver sets the diagonal elements of input Hermitian matrix A to be real. So before computing A * V we need to set the diagonal elements of A to be real, if case they are not.
        if constexpr (common::is_complex<data_type>()) {
            for (int b = 0; b < batches; ++b) {
                for (int row = 0; row < m; ++row) {
                    A_input[row + row * m + b * input_size_a].y = 0;
                }
            }
        }

        std::vector<data_type> A_v(input_size_a);
        std::vector<data_type> lambda_v(input_size_a);
        constexpr unsigned col_stride = is_col_maj_a ? m : 1;
        constexpr unsigned row_stride = is_col_maj_a ? 1 : m;
        
        for (int b = 0; b < batches; ++b) {
            data_type* Ab = A_input.data() + b * input_size_a;
            data_type* Vb = V.data() + b * input_size_a;   // Use cuSolverDx eigenvectors
            //data_type* Vb = A.data() + b * m * m; // Use cuSolver reference eigenvectors
            precision_type* lambda_b = lambda.data() + b * m;

            // Compute A * V (matrix-matrix multiplication) and V * diagm(lambda) (practically scalar-vector multiplication)
            A_v.assign(m * m, common::convert<data_type, precision_type>(0.));
            for (int row = 0; row < m; ++row) {
                for (int col = 0; col < m; ++col) {
                    lambda_v[row * row_stride + col * col_stride] = common::convert<data_type, precision_type>(lambda_b[col]) * Vb[row * row_stride + col * col_stride];
                    for (int k = 0; k < m; ++k) {
                        A_v[row * row_stride + col * col_stride] += Ab[row * row_stride + k * col_stride] * Vb[k * row_stride + col * col_stride];
                    }
                }
            }
            
            // Compare A * V with V * diagm(lambda) to verify eigenvector property
            const auto total_relative_error_lambda = common::check_error<data_type, data_type>(A_v.data(), lambda_v.data(), m * m);
            std::cout << "HEEV: relative error of A * V - V * diagm(lambda): for batch " << b << " : " << total_relative_error_lambda << std::endl;
            if (!common::is_error_acceptable<data_type>(total_relative_error_lambda)) {
                std::cout << "Relative error of A * V - V * diagm(lambda) is too large for batch " << b << std::endl;
                return 1;
            }
        }
    }

    std::cout << "Success compared eigenvalues with the cuSolver API results, and if vectors are computed, verified eigenvectors with A * V - V * diagm(lambda)" << std::endl;
    return 0;
}

template<int Arch>
struct heev_batched_functor {
    int operator()() { return heev_batched<Arch>(); }
};


int main() { return common::run_example_with_sm<heev_batched_functor>(); }
