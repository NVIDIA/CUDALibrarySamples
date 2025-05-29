// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <cusolverdx.hpp>

#include "../common/common.hpp"
#include "../common/cudart.hpp"
#include "../common/error_checking.hpp"
#include "../common/random.hpp"
#include "../common/example_sm_runner.hpp"
#include "../common/device_io.hpp"
#include "../common/measure.hpp"
#include "../common/print.hpp"
#include "../common/cusolver_reference_cholesky.hpp"

// This example demonstrates how to use cuSolverDx API to solve a batched linear systems with multiple right hand side
// after performing Cholesky factorization of the batched symmetric, positive-definite matrix A.
// The results are compared with the reference values obtained with cuSolver host API.

template<class POSV, unsigned int BatchesPerBlock, class DataType = typename POSV::a_data_type>
__global__ __launch_bounds__(POSV::max_threads_per_block) void posv_kernel(DataType* A, const unsigned int lda_gmem, DataType* B, const unsigned int ldb_gmem, typename POSV::status_type* info, const unsigned int batches) {

    using namespace cusolverdx;
    constexpr auto m                     = POSV::m_size;
    constexpr auto nrhs                  = POSV::k_size;
    const auto     one_batch_size_a_gmem = lda_gmem * m;
    const auto     one_batch_size_b_gmem = (arrangement_of_v_b<POSV> == arrangement::col_major) ? ldb_gmem * nrhs : m * ldb_gmem;

    constexpr auto lda_smem              = POSV::lda;
    constexpr auto ldb_smem              = POSV::ldb;
    constexpr auto one_batch_size_a_smem = lda_smem * m;
    constexpr auto one_batch_size_b_smem = (arrangement_of_v_b<POSV> == arrangement::col_major) ? ldb_smem * nrhs : m * ldb_smem;

    extern __shared__ __align__(sizeof(DataType)) char shared_mem[];

    DataType* As = reinterpret_cast<DataType*>(shared_mem);
    DataType* Bs = As + one_batch_size_a_smem * BatchesPerBlock;

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;

    auto Ag = A + size_t(one_batch_size_a_gmem) * batch_idx;
    auto Bg = B + size_t(one_batch_size_b_gmem) * batch_idx;

    // Load data from global memory to shared memory
    common::io<POSV, BatchesPerBlock>::load_a(Ag, lda_gmem, As, lda_smem);
    common::io<POSV, BatchesPerBlock>::load_b(Bg, ldb_gmem, Bs, ldb_smem);

    POSV().execute(As, lda_smem, Bs, &info[batch_idx]);

    // Store results back to global memory
    common::io<POSV, BatchesPerBlock>::store_a(As, lda_smem, Ag, lda_gmem);
    common::io<POSV, BatchesPerBlock>::store_b(Bs, ldb_smem, Bg, ldb_gmem);
}

template<int Arch>
int simple_posv_batched() {

    using namespace cusolverdx;

    using POSV = decltype(Size<32 /* = m */, 32 /* = n */, 1 /* = k */>() + Precision<double>() + Type<type::complex>() + Function<function::posv>() + FillMode<lower>() +
                          Arrangement<col_major /* A, X, and B */>() + SM<Arch>() + Block());

    constexpr unsigned bpb = POSV::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Using Suggested BlockDim = " << POSV::suggested_block_dim.x << std::endl;
    std::cout << "Using Specified BlockDim = " << POSV::block_dim.x << std::endl;

#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<POSV>;
    using cuda_data_type = typename example::a_cuda_data_type_t<POSV>;
#else
    using data_type      = typename POSV::a_data_type;
    using cuda_data_type = typename POSV::a_cuda_data_type;
#endif
    constexpr auto m    = POSV::m_size;
    constexpr auto n    = POSV::n_size;
    constexpr auto nrhs = POSV::k_size;
    static_assert(m == n, "posv is for Hermitian positive-definite matrix matrix only");
    constexpr auto lda_smem = POSV::lda;
    constexpr auto ldb_smem = POSV::ldb;

    constexpr bool is_col_maj_a = arrangement_of_v_a<POSV> == arrangement::col_major;
    constexpr bool is_col_maj_b = arrangement_of_v_b<POSV> == arrangement::col_major;

    // no padding for global memory
    constexpr auto lda = m;
    constexpr auto ldb = is_col_maj_b ? m : nrhs;

    printf("Size m = %d, n = %d, nrhs = %d\n", m, n, nrhs);
    std::cout << "Using leading dimension LDA = " << lda_smem << ", LDB = " << ldb_smem << std::endl;

    const auto batches        = 2;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    const auto one_batch_size_A = lda * n; // no padding for global memory
    const auto one_batch_size_B = m * nrhs;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(one_batch_size_A * padded_batches);
    std::vector<data_type> L(one_batch_size_A * padded_batches);

    common::fillup_random_diagonal_dominant_matrix<data_type>(arrangement_of_v_a<POSV> == col_major, m, n, A.data(), lda, false, 2, 4, batches);

    // To get around cuSolver potrsBatched bug for CUDA <=12.2, set the diagonal elements of input matrix A to be real
#if (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ <= 2)
    if constexpr (common::is_complex<data_type>()) {
        for (auto batch = 0; batch < batches; batch++) {
            for (unsigned int row = 0; row < m; row++) {
                A[row + row * lda + batch * one_batch_size_A].y = 0;
            }
        }
    }
#endif

    std::vector<data_type> B(one_batch_size_B * padded_batches);
    common::fillup_random_matrix<data_type>(arrangement_of_v_b<POSV> == col_major, m, nrhs, B.data(), ldb, false, false, -1, 1, batches);
    std::vector<data_type> X(one_batch_size_B * padded_batches);

    std::vector<int> info(padded_batches, 0);
    data_type*       d_A    = nullptr; /* device copy of A */
    data_type*       d_B    = nullptr; /* device copy of B */
    int*             d_info = nullptr; /* error info */


    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(data_type) * B.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * padded_batches));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    // Increase max dynamic shared memory for the kernel if needed.
    const auto sm_size = POSV::shared_memory_size;

    const auto kernel = posv_kernel<POSV, POSV::batches_per_block>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

    //Invokes kernel
    kernel<<<(batches + bpb - 1) / bpb, POSV::block_dim, sm_size, stream>>>(d_A, lda, d_B, ldb, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(X.data(), d_B, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * batches, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    if (std::accumulate(info.begin(), info.end(), 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }

    // Uncomment below to print the results after cuSolverDx execute
    //printf("after cuSolverDx execute\n");
    //printf("L = \n");
    //common::print_matrix<data_type, m, n, lda, is_col_maj_a>(L.data(), padded_batches);
    //printf("=====\n");
    //printf("X = \n");
    // common::print_matrix<data_type, n, nrhs, ldb, is_col_maj_b>(X.data(), padded_batches);
    // printf("=====\n");

    //=======================================================
    // cuSolver reference with potrfBatched and portsBatched
    //=======================================================
    common::reference_cusolver_cholesky<data_type, cuda_data_type, true>(A,
                                                                         B,
                                                                         info.data(),
                                                                         m,
                                                                         nrhs,
                                                                         padded_batches,
                                                                         (fill_mode_of_v<POSV> == fill_mode::lower),           /* is_lower? */
                                                                         is_col_maj_a,
                                                                         is_col_maj_b,
                                                                         batches);

    auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), batches * one_batch_size_A);
    printf("BATCHED POSV: relative error of A between cuSolverDx and cuSolver results: = %e\n", total_relative_error);

    // Uncomment below to print the results after cuSolver reference execute
    // printf("after cuSolver API execute\n");
    // printf("A = \n");
    // common::print_matrix<data_type, m, n, lda, is_col_maj_a>(A.data(), padded_batches);
    // printf("B = \n");
    // common::print_matrix<data_type, n, nrhs, ldb, is_col_maj_b>(B.data(), padded_batches);
    // printf("=====\n");

    total_relative_error = common::check_error<data_type, data_type>(X.data(), B.data(), batches * one_batch_size_B);
    printf("BATCHED POSV: relative error of B between cuSolverDx and cuSolver results: = %e\n", total_relative_error);
    if (common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Success compared to cuSolver potrSBatched Result " << std::endl;
    } else {
        std::cout << "Failure compared to cuSolver potrSBatched Result " << std::endl;
        return 1;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_B));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));

    CUDA_CHECK_AND_EXIT(cudaDeviceReset());

    return 0;
}

template<int Arch>
struct simple_posv_batched_functor {
    int operator()() { return simple_posv_batched<Arch>(); }
};


int main() { return common::run_example_with_sm<simple_posv_batched_functor>(); }
