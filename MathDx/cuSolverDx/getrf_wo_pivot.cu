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

// This example demonstrates using cuSolverDx API to perform LU factorization without pivoting on a general MxN matrix.
// The results are compared with the reference values obtained with cuSolver host API.

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ void kernel(DataType* A, unsigned int lda, typename Solver::status_type* info) {

    extern __shared__ unsigned char shared_mem[];

    DataType* As = reinterpret_cast<DataType*>(shared_mem);

    // Load data from global memory to registers
    common::io<Solver>::load(A, lda, As, lda);

    Solver().execute(As, lda, info);

    // store
    common::io<Solver>::store(As, lda, A, lda);
}

template<int Arch>
int getrf_wo_pivot() {

    using namespace cusolverdx;
    using Solver = decltype(Size<60, 64>() + Precision<double>() + Type<type::real>() + Function<getrf_no_pivot>() + Arrangement<arrangement::col_major>() + SM<Arch>() + Block() + BlockDim<256>());

#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<Solver>;
    using cuda_data_type = typename example::a_cuda_data_type_t<Solver>;
#else
    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
#endif
    constexpr auto m          = Solver::m_size;
    constexpr auto n          = Solver::n_size;
    const auto     lda        = m;
    const auto     input_size = lda * n;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size);
    common::fillup_random_diagonal_dominant_matrix<data_type>(arrangement_of_v_a<Solver> == arrangement::col_major, m, n, A.data(), lda, false, 2, 4); // not symmetric

    std::vector<data_type> L(input_size);
    int                    info   = 0;
    data_type*             d_A    = nullptr; /* device copy of A */
    int*                   d_info = nullptr; /* error info */

    // printf("A = \n");
    // common::print_matrix(m, n, A.data(), lda);
    // printf("=====\n");

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel<Solver>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));

    //Invokes kernel
    kernel<Solver><<<1, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, lda, d_info);
    CUDA_CHECK_AND_EXIT(cudaGetLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    printf("after cuSolverDx getrf kernel: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        return 1;
    }
    // printf("L = \n");
    // common::print_matrix(m, n, L.data(), lda);

    //=========================
    // cuSolver reference
    //=========================
    std::vector<data_type> dummy_B;
    common::reference_cusolver_lu<data_type, cuda_data_type>(A, dummy_B, &info, m, n);

    // printf("A = \n");
    // common::print_matrix(m, n, A.data(), lda);

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));

    // check result
    const auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), L.size());
    std::cout << "GETRF no pivoting: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error << std::endl;

    if (common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Success compared with cuSolver API results" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

template<int Arch>
struct getrf_wo_pivot_functor {
    int operator()() { return getrf_wo_pivot<Arch>(); }
};


int main() { return common::run_example_with_sm<getrf_wo_pivot_functor>(); }
