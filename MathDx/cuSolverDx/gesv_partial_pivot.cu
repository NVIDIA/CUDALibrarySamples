// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <iostream>

#include <cusolverdx.hpp>
#include "common.hpp"

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ void kernel(DataType* A, int lda_gmem, int* ipiv, DataType* B, int ldb_gmem, typename Solver::status_type* info) {

    using namespace cusolverdx;
    constexpr auto n            = Solver::n_size; // square matrix
    constexpr auto nrhs         = Solver::nrhs;
    constexpr int  lda_smem     = Solver::lda;
    constexpr int  ldb_smem     = Solver::ldb;
    constexpr bool is_col_maj_b = arrangement_of_v_b<Solver> == arrangement::col_major;

    extern __shared__ __align__(16) char shared_mem[];
    DataType*                            As    = reinterpret_cast<DataType*>(shared_mem);
    DataType*                            Bs    = As + lda_smem * n;
    int*                                 ipivs = reinterpret_cast<int*>(Bs + (is_col_maj_b ? ldb_smem * nrhs : n * ldb_smem));

    // Load data from global memory to registers
    common::io<Solver>::load(A, lda_gmem, As, lda_smem);
    common::io<Solver>::load_rhs(B, ldb_gmem, Bs, ldb_smem);

    Solver().execute(As, ipivs, Bs, info);

    // store
    common::io<Solver>::store(As, lda_smem, A, lda_gmem);
    common::io<Solver>::store_rhs(Bs, ldb_smem, B, ldb_gmem);

    int thread_id = threadIdx.x + Solver::block_dim.x * (threadIdx.y + Solver::block_dim.y * threadIdx.z);
    for (int i = thread_id; i < n; i += Solver::max_threads_per_block) {
        ipiv[i] = ipivs[i];
    }
}

template<int Arch>
int simple_gesv() {

    using namespace cusolverdx;
    using Solver = decltype(Size<16, 16>() + Precision<float>() + Type<type::real>() + Function<gesv_partial_pivot>() + Arrangement<arrangement::col_major>() + SM<Arch>() + Block() + BlockDim<32>());
#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<Solver>;
    using cuda_data_type = typename example::a_cuda_data_type_t<Solver>;
#else
    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
#endif
    constexpr auto m    = Solver::m_size;
    constexpr auto n    = Solver::n_size;
    constexpr auto nrhs = Solver::nrhs;
    constexpr auto lda  = Solver::lda;
    constexpr auto ldb  = Solver::lda;

    constexpr bool is_col_maj_a = arrangement_of_v_a<Solver> == arrangement::col_major;
    constexpr bool is_col_maj_b = arrangement_of_v_b<Solver> == arrangement::col_major;
    constexpr auto input_size_a = lda * n;
    constexpr auto input_size_b = is_col_maj_b ? lda * nrhs : m * lda;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size_a);
    common::fillup_random_matrix<data_type>(is_col_maj_a, m, n, A.data(), lda, false, false, 2, 4); // not symmetric and not diagonally dominant
    //const std::vector<data_type> A = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};

    std::vector<data_type> B(input_size_b);
    common::fillup_random_matrix<data_type>(is_col_maj_b, m, nrhs, B.data(), ldb, false, false, -1, 1);
    std::vector<data_type> X(input_size_b);

    std::vector<data_type> L(input_size_a);
    std::vector<int>       ipiv(min(m, n), 0);
    std::vector<int64_t>   ipiv_ref(min(m, n), 0);
    int                    info   = 0;
    data_type*             d_A    = nullptr; /* device copy of A */
    data_type*             d_b    = nullptr; /* device copy of b */
    int*                   d_info = nullptr; /* error info */
    int*                   d_ipiv = nullptr;

    const unsigned int b_dim_fast = (arrangement_of_v_b<Solver> == col_major) ? n : nrhs;
    // printf("A = \n");
    // common::print_matrix(m, n, A.data(), lda);
    // printf("=====\n");
    // printf("B = \n");
    // common::print_matrix(b_dim_fast, B.size() / b_dim_fast, B.data(), ldb);
    // printf("=====\n");

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(data_type) * B.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_ipiv), sizeof(int) * ipiv.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_b, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver><<<1, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda, d_ipiv, d_b, ldb, d_info);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(X.data(), d_b, sizeof(data_type) * B.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(ipiv.data(), d_ipiv, sizeof(int) * ipiv.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    printf("after cuSolverDx gesv kernel: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    // printf("L = \n");
    // common::print_matrix(m, n, L.data(), lda);
    // for (int i = 0; i < min(m, n); ++i) {
    //     std::cout << i << " " << ipiv[i] << std::endl;
    // }
    // printf("X = \n");
    // common::print_matrix(b_dim_fast, X.size() / b_dim_fast, X.data(), ldb);
    // printf("=====\n");

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_b));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ipiv));

    //=========================
    // cuSolver reference
    //=========================
    common::reference_cusolver_lu<data_type, cuda_data_type>(A,
                                                             B,
                                                             &info,
                                                             m,
                                                             n,
                                                             nrhs,
                                                             1 /* padded_batches */,
                                                             true /* is_pivot */,
                                                             is_col_maj_a,
                                                             is_col_maj_b,
                                                             false /* is_trans_a */,
                                                             true /* do_solver */,
                                                             ipiv_ref.data()); // is_pivot, col_maj_a, col_maj_b, do_solver, ipiv


    // check A
    const auto total_relative_error_a = common::check_error<data_type, data_type>(L.data(), A.data(), A.size());
    std::cout << "GESV partial pivoting: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error_a << std::endl;
    const auto total_relative_error_b = common::check_error<data_type, data_type>(X.data(), B.data(), B.size());
    std::cout << "GESV partial pivoting: relative error of X between cuSolverDx and cuSolver results: " << total_relative_error_b << std::endl;

    // printf("Lref = \n");
    // common::print_matrix(m, n, A.data(), lda);
    // for (int i = 0; i < min(m, n); ++i) {
    //     std::cout << i << " " << ipiv_ref[i] << std::endl;
    // }
    // printf("B = \n");
    // common::print_matrix(b_dim_fast, B.size() / b_dim_fast, B.data(), ldb);
    // printf("=====\n");


    if (!common::is_error_acceptable<data_type>(total_relative_error_a)) {
        std::cout << "Failure compared with cuSolver API results A" << std::endl;
        return 1;
    }
    if (!common::is_error_acceptable<data_type>(total_relative_error_b)) {
        std::cout << "Failure compared with cuSolver API results X" << std::endl;
        return 1;
    }

    // check ipiv
    for (int i = 0; i < min(m, n); ++i) {
        if (ipiv[i] != ipiv_ref[i]) {
            printf("ipiv[%d] = %d, ipiv_ref[%d] = %ld differ! \n", i, ipiv[i], i, ipiv_ref[i]);
            std::cout << "Failure compared with cuSolver API results ipiv" << std::endl;
            return 1;
        }
    }

    std::cout << "Success compared with cuSolver API results" << std::endl;
    return 0;
}

template<int Arch>
struct simple_gesv_functor {
    int operator()() { return simple_gesv<Arch>(); }
};


int main() { return common::run_example_with_sm<simple_gesv_functor>(); }
