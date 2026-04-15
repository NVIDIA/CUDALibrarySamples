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

// This example demonstrates how to use cuSolverDx API to compute singular values, and optionally singular vectors, of a batched m x n general matrix A with Block execution.
// The singular value results are compared with the reference values obtained with cuSolver host API, cusolverDnXgesvdjBatched and cusolverDnXgesvd API.
// The singular vector results are verified by comparing A with U * \Sigma * VT, where U and VT are the left and right singular vectors of A.

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type, typename PrecisionType = typename Solver::a_precision>
__global__ __launch_bounds__(Solver::max_threads_per_block) void kernel(
        DataType* A, const int lda_gmem, PrecisionType* sigma, DataType* U, const int ldu_gmem, DataType* VT, const int ldvt_gmem, typename Solver::status_type* info, const unsigned batches) {
    using namespace cusolverdx;

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto min_mn = m >= n ? n : m;
    
    const auto     one_batch_size_a_gmem = (Solver::a_arrangement == col_major ? lda_gmem * n : m * lda_gmem);
    constexpr auto lda_smem              = Solver::lda;
    constexpr auto one_batch_size_a_smem = (Solver::a_arrangement == col_major ? lda_smem * n : m * lda_smem);

    constexpr auto m_u  = m;
    constexpr auto n_u  = Solver::jobu == job::all_vectors ? m : min_mn;
    constexpr auto m_vt = Solver::jobvt == job::all_vectors ? n : min_mn;
    constexpr auto n_vt = n;

    const auto one_batch_size_u_gmem  = (Solver::jobu == job::no_vectors || Solver::jobu == job::overwrite_vectors)
                                            ? 0
                                            : (Solver::b_arrangement == col_major ? ldu_gmem * n_u : n_u * ldu_gmem);
    const auto one_batch_size_vt_gmem = (Solver::jobvt == job::no_vectors || Solver::jobvt == job::overwrite_vectors)
                                            ? 0
                                            : (Solver::c_arrangement == col_major ? ldvt_gmem * n_vt : n_vt * ldvt_gmem);

    constexpr auto ldu_smem               = (Solver::jobu == job::no_vectors) ? 0 : Solver::ldb;
    constexpr auto ldvt_smem              = (Solver::jobvt == job::no_vectors) ? 0 : Solver::ldc;
    constexpr auto one_batch_size_u_smem  = (Solver::b_arrangement == col_major ? ldu_smem * n_u : m_u * ldu_smem);
    constexpr auto one_batch_size_vt_smem = (Solver::c_arrangement == col_major ? ldvt_smem * n_vt : m_vt * ldvt_smem);

    constexpr auto workspace_size = Solver::workspace_size;

    extern __shared__ __align__(16) cusolverdx::byte shared_mem[];
    // Slice shared memory into pointers
    auto [A_s, sigma_s, U_s, VT_s, workspace_s] = cusolverdx::shared_memory::slice<DataType, PrecisionType, DataType, DataType, DataType>(
        shared_mem,
        alignof(DataType),
        one_batch_size_a_smem * BatchesPerBlock,
        alignof(PrecisionType),
        min_mn * BatchesPerBlock,
        alignof(DataType),
        one_batch_size_u_smem * BatchesPerBlock,
        alignof(DataType),
        one_batch_size_vt_smem * BatchesPerBlock,
        alignof(DataType) // the size can be omitted for the last pointer
    );

    auto A_g     = A + one_batch_size_a_gmem * batch_idx;
    auto sigma_g = sigma + min_mn * batch_idx;
    auto U_g     = U + one_batch_size_u_gmem * batch_idx;
    auto VT_g    = VT + one_batch_size_vt_gmem * batch_idx;
    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a<m, n>(A_g, lda_gmem, A_s, lda_smem);

    Solver().execute(A_s, lda_smem, sigma_s, U_s, ldu_smem, VT_s, ldvt_smem, workspace_s, &info[batch_idx]);

    // Store results back to global memory
    common::io<Solver, BatchesPerBlock>::store_a<m, n>(A_s, lda_smem, A_g, lda_gmem);
    cusolverdx::copy_2d<Solver, min_mn, 1, cusolverdx::arrangement::col_major, BatchesPerBlock, PrecisionType>(sigma_s, min_mn, sigma_g, min_mn);

    if constexpr (Solver::jobu == cusolverdx::job::all_vectors || Solver::jobu == cusolverdx::job::some_vectors) {
        cusolverdx::copy_2d<Solver, m_u, n_u, Solver::b_arrangement, BatchesPerBlock, DataType>(U_s, ldu_smem, U_g, ldu_gmem);
    } else if constexpr (Solver::jobu == cusolverdx::job::overwrite_vectors) {
        cusolverdx::copy_2d<Solver, m, n, Solver::a_arrangement, BatchesPerBlock, DataType>(A_s, lda_smem, A_g, lda_gmem);
    }
    if constexpr (Solver::jobvt == cusolverdx::job::all_vectors || Solver::jobvt == cusolverdx::job::some_vectors) {
        cusolverdx::copy_2d<Solver, m_vt, n_vt, Solver::c_arrangement, BatchesPerBlock, DataType>(VT_s, ldvt_smem, VT_g, ldvt_gmem);
    } else if constexpr (Solver::jobvt == cusolverdx::job::overwrite_vectors) {
        cusolverdx::copy_2d<Solver, m, n, Solver::a_arrangement, BatchesPerBlock, DataType>(A_s, lda_smem, A_g, lda_gmem);
    }
}

template<int Arch>
int gesvd_batched_block() {

    using namespace cusolverdx;
    using Base = decltype(Size<14, 7>() + Precision<float>() + Type<type::complex>() + Function<gesvd>() + SM<Arch>() + Block());

    // Job option for gesvd can be no_vectors, all_vectors, some_vectors or overwrite_vectors for either left or right singular vectors
    // Uses can test the following options and change the job/arrangement/leading dimensions as needed
    // using Base_with_jobs = decltype(Base() + Job<job::no_vectors, job::no_vectors>() + Arrangement<row_major>()); // compute singular values only
    using Base_with_jobs = decltype(Base() + Job<job::all_vectors, job::all_vectors>() + Arrangement<col_major, col_major, col_major>()); 
    // using Base_with_jobs = decltype(Base() + Job<job::all_vectors, job::all_vectors>() + Arrangement<row_major, row_major, col_major>());
    // using Base_with_jobs = decltype(Base() + Job<job::overwrite_vectors, job::all_vectors>() + Arrangement<row_major, row_major, col_major>()); 
    // using Base_with_jobs = decltype(Base() + Job<job::all_vectors, job::some_vectors>() + Arrangement<col_major, row_major, row_major>());
    // using Base_with_jobs = decltype(Base() + Job<job::all_vectors, job::overwrite_vectors>() + Arrangement<row_major, col_major, row_major>()); 
    //using Base_with_jobs = decltype(Base() + Job<job::overwrite_vectors, job::all_vectors>() + Arrangement<row_major, row_major, col_major>()); 

    using Solver = decltype(Base_with_jobs() + BatchesPerBlock<Base_with_jobs::suggested_batches_per_block>());

    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
    using precision_type = typename Solver::a_precision;
    
    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;
    
    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto min_mn = m >= n ? n : m;
    
    constexpr bool is_col_maj_a = Solver::a_arrangement == arrangement::col_major;
    constexpr bool is_col_maj_u = Solver::b_arrangement == arrangement::col_major;
    constexpr bool is_col_maj_vt = Solver::c_arrangement == arrangement::col_major;
    constexpr bool compute_vectors_u = Solver::jobu != job::no_vectors;
    constexpr bool compute_vectors_vt = Solver::jobvt != job::no_vectors;
    
    // Size of A, U, VT in global memory. U and VT storage depends on the job configuration.
    // jobu = job::all_vectors: U(m x m)
    // jobu = job::some_vectors: U(m x min_mn)
    // jobu = job::overwrite_vectors: U(m x min_mn) using Arrangement and Leading Dimension specified for A
    // jobvt = job::all_vectors: VT(n x n)
    // jobvt = job::some_vectors: VT(min_mn x n)
    // jobvt = job::overwrite_vectors: VT(min_mn x n) using Arrangement and Leading Dimension specified for A
    constexpr auto input_size_a = m * n;
    constexpr auto lda     = is_col_maj_a ? m : n;
    
    constexpr auto act_m_u = m;
    constexpr auto act_n_u = Solver::jobu == job::all_vectors ? m : min_mn;
    constexpr auto act_m_vt = Solver::jobvt == job::all_vectors ? n : min_mn;
    constexpr auto act_n_vt = n;    
    constexpr auto ldu        = (is_col_maj_u ? act_m_u : act_n_u);
    constexpr auto ldvt       = (is_col_maj_vt ? act_m_vt : act_n_vt);
    constexpr auto input_size_u = compute_vectors_u ? act_m_u * act_n_u : 0;
    constexpr auto input_size_vt = compute_vectors_vt ? act_m_vt * act_n_vt : 0;
    
    const auto batches        = 10;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    std::cout << "Solver: " << Solver::lda << ", " << Solver::ldb << ", " << Solver::ldc << std::endl;
    std::cout << "actual m_u = " << act_m_u << ", actual n_u = " << act_n_u << ", actual m_vt = " << act_m_vt << ", actual n_vt = " << act_n_vt << std::endl;
    std::cout << "input_size_a = " << input_size_a << ", input_size_u = " << input_size_u << ", input_size_vt = " << input_size_vt << std::endl;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A_input(input_size_a * padded_batches), A_ref(input_size_a * padded_batches);
    std::vector<data_type> U(input_size_u * padded_batches), U_ref(input_size_u * padded_batches);
    std::vector<data_type> VT(input_size_vt * padded_batches), VT_ref(input_size_vt * padded_batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A_input.data(), lda, false /*symmetric*/, false /*diagonal dominant*/, -2, 4, batches);
    A_ref.assign(A_input.begin(), A_input.end());

    std::vector<data_type> A_overwrite;
    if constexpr (Solver::jobu == job::overwrite_vectors || Solver::jobvt == job::overwrite_vectors) {
        A_overwrite.resize(input_size_a * padded_batches);
    }

    std::vector<precision_type> S(min_mn * padded_batches, 0), S_ref(min_mn * padded_batches, 0);
    std::vector<data_type>      workspace(Solver::workspace_size * padded_batches);
    std::vector<int>            info(padded_batches, 0);
    data_type*                  d_A    = nullptr;
    data_type*                  d_U    = nullptr;
    data_type*                  d_VT   = nullptr;
    precision_type*             d_S    = nullptr;
    int*                        d_info = nullptr;

    // Uncomment below to print A matrix
    // printf("A = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A_input.data(), batches);

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A_input.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_U), sizeof(data_type) * U.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_VT), sizeof(data_type) * VT.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_S), sizeof(precision_type) * S.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A_input.data(), sizeof(data_type) * A_input.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda, d_S, d_U, ldu, d_VT, ldvt, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(S.data(), d_S, sizeof(precision_type) * S.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));

    if constexpr (Solver::jobu == job::all_vectors || Solver::jobu == job::some_vectors) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(U.data(), d_U, sizeof(data_type) * U.size(), cudaMemcpyDeviceToHost, stream));
    } else if constexpr (Solver::jobu == job::overwrite_vectors) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A_overwrite.data(), d_A, sizeof(data_type) * A_overwrite.size(), cudaMemcpyDeviceToHost, stream));
    }
    if constexpr (Solver::jobvt == job::all_vectors || Solver::jobvt == job::some_vectors) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(VT.data(), d_VT, sizeof(data_type) * VT.size(), cudaMemcpyDeviceToHost, stream));
    } else if constexpr (Solver::jobvt == job::overwrite_vectors) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(A_overwrite.data(), d_A, sizeof(data_type) * A_overwrite.size(), cudaMemcpyDeviceToHost, stream));
    }
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
    // printf("===== After cuSolverDx execute\n");
    // printf("S = \n");
    // common::print_matrix<precision_type, min_mn, 1, min_mn, true>(S.data(), batches);
    // if constexpr (compute_vectors_u && Solver::jobu != job::overwrite_vectors) {  
    //     printf("=====\n");
    //     printf("U = \n");
    //     common::print_matrix<data_type, act_m_u, act_n_u, ldu, is_col_maj_u>(U.data(), batches);
    // }
    // if constexpr (compute_vectors_u && Solver::jobu == job::overwrite_vectors) {  
    //     printf("=====\n");
    //     printf("A_overwrite = \n");
    //     common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A_overwrite.data(), batches);
    // }
    // if constexpr (compute_vectors_vt && Solver::jobvt != job::overwrite_vectors) {
    //     printf("=====\n");
    //     printf("VT = \n");
    //     common::print_matrix<data_type, act_m_vt, act_n_vt, ldvt, is_col_maj_vt>(VT.data(), batches);
    //     printf("=====\n");
    // }
    // if constexpr (compute_vectors_vt && Solver::jobvt == job::overwrite_vectors) {
    //     printf("=====\n");
    //     printf("A_overwrite = \n");
    //     common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A_overwrite.data(), batches);
    // }

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_S));
    if (d_U != nullptr) {
        CUDA_CHECK_AND_EXIT(cudaFree(d_U));
    }
    if (d_VT != nullptr) {
        CUDA_CHECK_AND_EXIT(cudaFree(d_VT));
    }
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    //=========================================================
    // Check computed singular values with cuSolver reference
    //=========================================================
    std::cout << "The largest singular value is " << S[0] << " and the smallest singular value is " << S[min_mn - 1] << std::endl;
    if constexpr (m >= n) {
        signed char jobu = 'N', jobvt = 'N';
        if constexpr (Solver::jobu == job::all_vectors) {
            jobu = 'A';
        } else if constexpr (Solver::jobu == job::some_vectors) {
            jobu = 'S';
        } else if constexpr (Solver::jobu == job::overwrite_vectors) {
            jobu = 'O';
        }
        if constexpr (Solver::jobvt == job::all_vectors) {
            jobvt = 'A';
        } else if constexpr (Solver::jobvt == job::some_vectors) {
            jobvt = 'S';
        } else if constexpr (Solver::jobvt == job::overwrite_vectors) {
            jobvt = 'O';
        }
        // All the output A/U/VT matrices remain in col-major format after return of the function
        common::reference_cusolver_gesvd<data_type, cuda_data_type, precision_type, false /*check_perf*/>(
                A_ref, S_ref, U_ref, VT_ref, jobu, jobvt, info.data(), m, n, padded_batches, is_col_maj_a, batches);

        const auto total_relative_error_S = common::check_error<precision_type, precision_type>(S.data(), S_ref.data(), batches * min_mn);
        std::cout << "GESVD: relative error of S between cuSolverDx and cuSolver reference results: " << total_relative_error_S << std::endl;

        // Uncomment below to print results after cuSolver reference execute
        // printf("S_ref = \n");
        // common::print_matrix<precision_type, min_mn, 1, min_mn, true>(S_ref.data(), batches);
        // if constexpr (compute_vectors_u) {
        //     printf("=====\n");
        //     if constexpr (Solver::jobu == job::overwrite_vectors) {
        //         printf("A_overwrite_ref = \n");
        //         common::print_matrix<data_type, m, n, m, true>(A_ref.data(), batches);
        //     } else {
        //         printf("U_ref = \n");
        //         common::print_matrix<data_type, act_m_u, act_n_u, act_m_u, true>(U_ref.data(), batches);
        //     }
        // }
        // if constexpr (compute_vectors_vt) {
        //     if constexpr (Solver::jobvt == job::overwrite_vectors) {
        //         printf("A_overwrite_ref = \n");
        //         common::print_matrix<data_type, m, n, m, true>(A_ref.data(), batches);
        //     } else {
        //         printf("VT_ref = \n");
        //         common::print_matrix<data_type, act_m_vt, act_n_vt, act_m_vt, true>(VT_ref.data(), batches);
        //     }
        // }   

        if (!common::is_error_acceptable<precision_type>(total_relative_error_S)) {
            std::cout << "Failure compared with cuSolver API results S" << std::endl;
            for (int i = 0; i < min_mn * batches; ++i) {
                if (abs(S[i] - S_ref[i]) > 1e-05) {
                    printf("S[%d] = %10.3f, S_ref[%d] = %10.3f  differ \n", i, S[i], i, S_ref[i]);
                }
            }
            return 1;
        }
        std::cout << "Success compared singular values with the cuSolverDn reference results" << std::endl;
    } else {
        std::cout << "GESVD: m < n is not supported in cuSolverDn<t>getsvd API. Checking with cuSolverDn reference is disabled"
                  << std::endl;
    }

    //============================================================
    // Check computed left and right singular vectors by comparing:
    //     A_input = A_reconstructed = U * \Sigma * VT
    //============================================================
    if constexpr (compute_vectors_u && compute_vectors_vt) {
        // A/U/VT could be either col-major or row-major. For convenience of testing, transpose them if they are row-major
        if constexpr (!is_col_maj_a) {
            common::transpose_matrix<data_type>(A_input, m, n, batches);
        }
        if constexpr (!is_col_maj_u && Solver::jobu != job::overwrite_vectors) {
            common::transpose_matrix<data_type>(U, act_m_u, act_n_u, batches);
        }
        if constexpr (!is_col_maj_a && Solver::jobu == job::overwrite_vectors) {
            common::transpose_matrix<data_type>(A_overwrite, m, n, batches);
        }
        if constexpr ((!is_col_maj_vt && Solver::jobvt != job::overwrite_vectors)) {
            common::transpose_matrix<data_type>(VT, act_m_vt, act_n_vt, batches);
        }
        if constexpr (!is_col_maj_a && Solver::jobvt == job::overwrite_vectors) {
            common::transpose_matrix<data_type>(A_overwrite, m, n, batches);
        }

        const bool check_ref = false;
        data_type *AA, *UU, *VVT;
        if (check_ref) {
            AA = A_ref.data();
            UU = U_ref.data();
            VVT = VT_ref.data();
        } else {
            AA = A_overwrite.data();
            UU = U.data();
            VVT = VT.data();
        }
        std::vector<data_type> A_reconstructed(input_size_a);
        for (int b = 0; b < batches; ++b) {
            // output sigma is stored in d
            const auto* sigma_b   = S.data() + b * min_mn;
            const auto* U_b       = Solver::jobu == job::overwrite_vectors ? AA + b * input_size_a : UU + b * input_size_u;
            const auto* VT_b      = Solver::jobvt == job::overwrite_vectors ? AA + b * input_size_a : VVT + b * input_size_vt;
            const auto  ld_u      = Solver::jobu == job::overwrite_vectors ? m : act_m_u;
            const auto  ld_vt     = Solver::jobvt == job::overwrite_vectors ? m : act_m_vt;
            const auto* A_input_b = A_input.data() + b * input_size_a;

            // Compare A_input and A_reconstructed = U * \Sigma * VT
            A_reconstructed.assign(m * n, common::convert<data_type, float>(0.f));
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    for (int l = 0; l < min_mn; ++l) {
                        A_reconstructed[i + j * m] += common::convert<data_type, precision_type>(sigma_b[l]) * U_b[i + l * ld_u] * VT_b[l + j * ld_vt];
                    }
                }
            }
            const auto total_relative_error = common::check_error<data_type, data_type>(A_reconstructed.data(), A_input_b, input_size_a);
            std::cout << "GESVD: relative error of A - U Sigma VT: for batch " << b << " : " << total_relative_error << std::endl;
            if (!common::is_error_acceptable<data_type>(total_relative_error)) {
                printf("A_reconstructed = \n");
                common::print_matrix(m, n, true, A_reconstructed.data(), m);
                printf("A_input = \n");
                common::print_matrix(m, n, true, A_input_b, m);
                std::cout << "Relative error of A - U Sigma VT is too large for batch " << b << std::endl;
                return 1;
            }
        }
       std::cout << "Success in verifying singular values and vectors for all batches for GESVD" << std::endl;


    } else {
        std::cout << "Success " << std::endl;
    }
    return 0;
}

template<int Arch>
struct gesvd_batched_block_functor {
    int operator()() { return gesvd_batched_block<Arch>(); }
};


int main() { return common::run_example_with_sm<gesvd_batched_block_functor>(); }
