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
#include "../common/cusolver_reference_lu.hpp"

// This example demonstrates how to use cuSolverDx API to solve a batched linear systems with multiple right hand side after performing LU
// factorization with pivoting of the batched general matrix A.  The results are compared with the reference values obtained with cuSolver host API.

template<class Solver, unsigned int BatchesPerBlock, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void kernel(DataType* A, const int lda_gmem, int* ipiv, DataType* B, const int ldb_gmem, typename Solver::status_type* info, const unsigned batches) {

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    static_assert(m == n, "gesv kernels only allow square matrices");
    constexpr auto k = Solver::k_size;

    const auto     one_batch_size_a_gmem = lda_gmem * m;
    const auto     one_batch_size_b_gmem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? ldb_gmem * k : n * ldb_gmem;
    constexpr auto lda_smem              = Solver::lda;
    constexpr auto ldb_smem              = Solver::ldb;
    constexpr auto one_batch_size_a_smem = lda_smem * m;
    constexpr auto one_batch_size_b_smem = (cusolverdx::arrangement_of_v_b<Solver> == cusolverdx::col_major) ? ldb_smem * k : n * ldb_smem;

    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [As, Bs, ipivs] = cusolverdx::shared_memory::slice<DataType, DataType, int>(
        shared_mem,
        alignof(DataType), one_batch_size_a_smem * BatchesPerBlock,
        alignof(DataType), one_batch_size_b_smem * BatchesPerBlock,
        alignof(int) // the size (number of elements) may be omitted for the last pointer
    );

    const auto batch_idx = blockIdx.x * BatchesPerBlock;
    if (batch_idx >= batches)
        return;
    auto Ag    = A + size_t(one_batch_size_a_gmem) * batch_idx;
    auto Bg    = B + size_t(one_batch_size_b_gmem) * batch_idx;
    auto ipivg = ipiv + batch_idx * n;

    // Load data from global memory to shared memory
    common::io<Solver, BatchesPerBlock>::load_a(Ag, lda_gmem, As, lda_smem);
    common::io<Solver, BatchesPerBlock>::load_b(Bg, ldb_gmem, Bs, ldb_smem);

    Solver().execute(As, ipivs, Bs, &info[batch_idx]);

    // Store results back to global memory
    common::io<Solver, BatchesPerBlock>::store_a(As, lda_smem, Ag, lda_gmem);
    common::io<Solver, BatchesPerBlock>::store_b(Bs, ldb_smem, Bg, ldb_gmem);

    // store ipiv from shared memory to global memory
    int thread_id = threadIdx.x + Solver::block_dim.x * (threadIdx.y + Solver::block_dim.y * threadIdx.z);
    for (int i = thread_id; i < n * BatchesPerBlock; i += Solver::max_threads_per_block) {
        ipivg[i] = ipivs[i];
    }
}

template<int Arch>
int gesv_batched_partial_pivot() {

    using namespace cusolverdx;
    using Base   = decltype(Size<18, 18, 1>() + Precision<float>() + Type<type::complex>() + Function<gesv_partial_pivot>() + Arrangement<arrangement::col_major, arrangement::col_major>() +
                          TransposeMode<non_trans>() + LeadingDimension<18, 20>() + SM<Arch>() + Block() + BlockDim<64>());
    using Solver = decltype(Base() + BatchesPerBlock<Base::suggested_batches_per_block>());

#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<Solver>;
    using cuda_data_type = typename example::a_cuda_data_type_t<Solver>;
#else
    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
#endif
    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Using Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    constexpr auto k = Solver::k_size;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;
    constexpr bool is_col_maj_b = arrangement_of_v_b<Solver> == arrangement::col_major;

    const auto     lda          = is_col_maj_a ? m : n;
    const auto     ldb          = is_col_maj_b ? n : k;
    constexpr auto input_size_a = m * n;
    constexpr auto input_size_b = n * k;

    const auto batches        = 3;
    const auto padded_batches = (batches + bpb - 1) / bpb * bpb;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a * padded_batches);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, 2, 4, batches); // not symmetric and not diagonally dominant

    std::vector<data_type> B(input_size_b * padded_batches);
    common::fillup_random_matrix<data_type>(is_col_maj_b, m, k, B.data(), ldb, false, false, -1, 1, batches);
    std::vector<data_type> X(input_size_b * padded_batches);

    std::vector<data_type> L(input_size_a * padded_batches);
    std::vector<int>       ipiv(min(m, n) * padded_batches, 0);
    std::vector<int64_t>   ipiv_ref(min(m, n) * padded_batches, 0);
    std::vector<int>       info(padded_batches, 1);
    data_type*             d_A    = nullptr; /* device copy of A */
    data_type*             d_b    = nullptr; /* device copy of b */
    int*                   d_info = nullptr; /* error info */
    int*                   d_ipiv = nullptr;


    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(data_type) * B.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_ipiv), sizeof(int) * ipiv.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_b, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver, bpb>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver, bpb><<<padded_batches / bpb, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda, d_ipiv, d_b, ldb, d_info, batches);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(X.data(), d_b, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(ipiv.data(), d_ipiv, sizeof(int) * ipiv.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    if (std::accumulate(info.begin(), info.begin() + batches, 0) != 0) {
        std::cout << "non-zero d_info returned for at least one of the batches after cuSolverDx kernel \n";
        for (int j = 0; j < batches; j++) {
            if (info[j] != 0)
                std::cout << "info[" << j << "]=" << info[j] << std::endl;
        }
        return -1;
    }


    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_b));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ipiv));

    //=========================
    // cuSolver reference
    //=========================
    common::reference_cusolver_lu<data_type, cuda_data_type, true>(A,
                                                                   B,
                                                                   info.data(),
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   padded_batches,
                                                                   true /* is_pivot */,
                                                                   is_col_maj_a,
                                                                   is_col_maj_b,
                                                                   (transpose_mode_of_v<Solver> == trans || transpose_mode_of_v<Solver> == conj_trans),
                                                                   ipiv_ref.data(),
                                                                   batches); // is_pivot, col_maj_a, col_maj_b, do_solver, ipiv


    // check A
    const auto total_relative_error_a = common::check_error<data_type, data_type>(L.data(), A.data(), batches * input_size_a);
    std::cout << "GESV partial pivoting: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error_a << std::endl;
    const auto total_relative_error_b = common::check_error<data_type, data_type>(X.data(), B.data(), batches * input_size_b);
    std::cout << "GESV partial pivoting: relative error of X between cuSolverDx and cuSolver results: " << total_relative_error_b << std::endl;

    if (!common::is_error_acceptable<data_type>(total_relative_error_a)) {
        std::cout << "Failure compared with cuSolver API results A" << std::endl;
        return 1;
    }
    if (!common::is_error_acceptable<data_type>(total_relative_error_b)) {
        std::cout << "Failure compared with cuSolver API results X" << std::endl;
        return 1;
    }

    // check ipiv
    for (int i = 0; i < min(m, n) * batches; ++i) {
        if (ipiv[i] != ipiv_ref[i]) {
            printf("ipiv[%d] = %d, ipiv_ref[%d] = %ld differ! \n", i, ipiv[i], i, ipiv_ref[i]);
            std::cout << "Failure compared with cuSolver API results ipiv" << std::endl;
            return 1;
        }
    }

    std::cout << "Success compared with cuSolver API results, ipiv, A and B" << std::endl;
    return 0;
}

template<int Arch>
struct gesv_batched_partial_pivot_functor {
    int operator()() { return gesv_batched_partial_pivot<Arch>(); }
};


int main() { return common::run_example_with_sm<gesv_batched_partial_pivot_functor>(); }
