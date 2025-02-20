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

// This example shows how to perform Cholesky factorization for a Hermitian positive-definite matrix
// using cuSolverDx, and compare the result factors with cuSolver host API.

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ void potrf_kernel(DataType* A, typename Solver::status_type* info) {

    extern __shared__ unsigned char shared_mem[];
    DataType* As = reinterpret_cast<DataType*>(shared_mem);

    constexpr auto lda_smem = Solver::lda;
    constexpr auto lda_gmem = Solver::m_size;

    // Load data from global memory to registers
    common::io<Solver>::load(A, lda_gmem, As, lda_smem);

    Solver().execute(As, info);

    // store
    common::io<Solver>::store(As, lda_smem, A, lda_gmem);
}

template<int Arch>
int simple_potrf() {

    using namespace cusolverdx;
    using Solver = decltype(Size<32, 32>() + Precision<double>() + Type<type::complex>() + Function<potrf>() + Block() + LeadingDimension<33>() + SM<Arch>() + BlockDim<256>() +
                            FillMode<fill_mode::upper>());

#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<Solver>;
    using cuda_data_type = typename example::a_cuda_data_type_t<Solver>;
#else
    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
#endif
    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;
    static_assert(m == n, "potrf is for Hermitian positive-definite matrix matrix only");
    constexpr auto lda_smem = Solver::lda;

    constexpr auto lda        = m;       // this is the leading dimension in global memory for A
    constexpr auto input_size = lda * n; // input global memory size for A

    std::cout << "Use compile-time leading dimension LDA for shared memory = " << lda_smem << std::endl;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size);
    common::fillup_random_diagonal_dominant_matrix_col_major<data_type>(m, n, A.data(), lda, false, -2, 2); // input A is not symmetric

    // To get around cuSolver potrsBatched bug for CUDA <= 12.2, set the diagonal elements of input matrix A to be real
#if (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ <= 2)
    if constexpr (common::is_complex<data_type>()) {
        for (unsigned int row = 0; row < m; row++) {
            A[row + row * lda].y = 0;
        }
    }
#endif

    // Uncomment out the following lines to print out the matrix A
    // printf("A = \n");
    // common::print_matrix(m, n, A.data(), lda);
    // printf("=====\n");

    std::vector<data_type> L(input_size);
    int                    info   = 0;
    data_type*             d_A    = nullptr; /* device copy of A */
    int*                   d_info = nullptr; /* error info */

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(potrf_kernel<Solver>, cudaFuncAttributeMaxDynamicSharedMemorySize, Solver::shared_memory_size));
    //Invokes kernel
    potrf_kernel<Solver><<<1, Solver::block_dim, Solver::shared_memory_size, stream>>>(d_A, d_info);

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    printf("after cuSolverDx potrf kernel: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    // uncomment out the following lines to print out the result factors
    // printf("L = \n");
    // common::print_matrix(m, m, L.data(), lda);

    //=========================
    // cuSolver reference
    //=========================
    // Use dumb B as only factorization is performed
    std::vector<data_type> B;
    common::reference_cusolver_cholesky<data_type, cuda_data_type>(
        A, B, &info, m, 1, 1, (fill_mode_of_v<Solver> == fill_mode::lower));

    // check result
    const auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), A.size());
    std::cout << "Solver: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error << std::endl;

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaDeviceReset());

    if (common::is_error_acceptable<data_type>(total_relative_error)) {
        std::cout << "Success compared with cuSolver API results" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

template<int Arch>
struct simple_potrf_functor {
    int operator()() { return simple_potrf<Arch>(); }
};


int main() { return common::run_example_with_sm<simple_potrf_functor>(); }
