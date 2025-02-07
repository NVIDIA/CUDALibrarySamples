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
__global__ void kernel(DataType* A, typename Solver::status_type* info, int* d_ipiv) {

    Solver().execute(A, d_ipiv, info);
}

template<int Arch>
int simple_getrf() {

    using namespace cusolverdx;
    using Solver =
        decltype(Size<32, 32>() + Precision<double>() + Type<type::real>() + Function<getrf_partial_pivot>() + Arrangement<arrangement::col_major>() + SM<Arch>() + Block() + BlockDim<96, 1, 1>());
#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<Solver>;
    using cuda_data_type = typename example::a_cuda_data_type_t<Solver>;
#else
    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
#endif
    constexpr auto m   = Solver::m_size;
    constexpr auto n   = Solver::n_size;
    constexpr auto lda = Solver::lda;

    constexpr bool is_col_maj = arrangement_of_v_a<Solver> == arrangement::col_major;
    constexpr auto input_size = is_col_maj ? lda * n : m * lda;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size);
    common::fillup_random_matrix<data_type>(is_col_maj, m, n, A.data(), lda, false, false, 2, 4); // not symmetric and not diagonally dominant
    //const std::vector<data_type> A = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};

    std::vector<data_type> L(input_size);
    std::vector<int>       ipiv(min(m, n), 0);
    std::vector<int64_t>   ipiv_ref(min(m, n), 0);
    int                    info   = 0;
    data_type*             d_A    = nullptr; /* device copy of A */
    int*                   d_info = nullptr; /* error info */
    int*                   d_ipiv = nullptr;

    // printf("A = \n");
    // common::print_matrix(m, n, A.data(), lda);
    // printf("=====\n");

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_ipiv), sizeof(int) * ipiv.size()));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver><<<1, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, d_info, d_ipiv);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(ipiv.data(), d_ipiv, sizeof(int) * ipiv.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    printf("after cuSolverDx getrf kernel: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    // printf("L = \n");
    // common::print_matrix(m, n, L.data(), lda);
    // for (int i = 0; i < min(m, n); ++i) {
    //     std::cout << i << " " << ipiv[i] << std::endl;
    // }

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ipiv));

    //=========================
    // cuSolver reference
    //=========================
    std::vector<data_type> dummy_B;
    common::reference_cusolver_lu<data_type, cuda_data_type>(A,
                                                             dummy_B,
                                                             &info,
                                                             m,
                                                             n,
                                                             1,
                                                             1 /* padded_batches */,
                                                             true /* is_pivot */,
                                                             is_col_maj,
                                                             true /* is_col_major_b */,
                                                             false /* is_trans_a */,
                                                             false /* do_solver */,
                                                             ipiv_ref.data());


    // check A
    const auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), A.size());
    std::cout << "GETRF partial pivoting: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error << std::endl;

    // printf("Lref = \n");
    // common::print_matrix(m, n, A.data(), lda);
    // for (int i = 0; i < min(m, n); ++i) {
    //     std::cout << i << " " << ipiv_ref[i] << std::endl;
    // }


    if (!common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Failure compared with cuSolver API results A" << std::endl;
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
struct simple_getrf_functor {
    int operator()() { return simple_getrf<Arch>(); }
};


int main() { return common::run_example_with_sm<simple_getrf_functor>(); }
