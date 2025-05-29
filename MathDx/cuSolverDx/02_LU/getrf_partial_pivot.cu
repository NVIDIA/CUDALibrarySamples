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

// This example demonstrates using cuSolverDx API to perform LU factorization with pivoting on a single-batch, general MxN matrix.
// The results are compared with the reference values obtained with cuSolver host API.

template<class Solver, typename DataType = typename Solver::a_data_type>
__global__ __launch_bounds__(Solver::max_threads_per_block) void kernel(DataType* A, typename Solver::status_type* info, int* d_ipiv) {

    constexpr auto m        = Solver::m_size;
    constexpr auto n        = Solver::n_size;
    constexpr auto lda_smem = Solver::lda;

    constexpr auto lda_gmem              = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? m : n;
    constexpr auto one_batch_size_a_smem = (cusolverdx::arrangement_of_v_a<Solver> == cusolverdx::col_major) ? lda_smem * n : m * lda_smem;

    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers
    auto [As, ipivs] = cusolverdx::shared_memory::slice<DataType, int>(
        shared_mem,
        alignof(DataType), one_batch_size_a_smem,
        alignof(int)       // the size (number of elements) may be omitted for the last pointer
    );

    // Load data from global memory to shared memory
    common::io<Solver>::load_a(A, lda_gmem, As, lda_smem);

    Solver().execute(As, ipivs, info);

    // Store results back to global memory
    common::io<Solver>::store_a(As, lda_smem, A, lda_gmem);
    int thread_id = threadIdx.x + Solver::block_dim.x * (threadIdx.y + Solver::block_dim.y * threadIdx.z);
    for (int i = thread_id; i < min(m, n); i += Solver::max_threads_per_block) {
        d_ipiv[i] = ipivs[i];
    }
}

template<int Arch>
int simple_getrf() {

    using namespace cusolverdx;
    using Solver =
        decltype(Size<48, 32>() + Precision<float>() + Type<type::complex>() + Function<getrf_partial_pivot>() + Arrangement<arrangement::col_major>() + SM<Arch>() + Block() + BlockDim<33, 1, 1>());
#ifdef CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND
    using data_type      = typename example::a_data_type_t<Solver>;
    using cuda_data_type = typename example::a_cuda_data_type_t<Solver>;
#else
    using data_type      = typename Solver::a_data_type;
    using cuda_data_type = typename Solver::a_cuda_data_type;
#endif

    constexpr unsigned bpb = Solver::batches_per_block;
    std::cout << "Suggested Batches per block = " << bpb << std::endl;
    std::cout << "Suggested BlockDim = " << Solver::suggested_block_dim.x << std::endl;
    std::cout << "BlockDim Used = " << Solver::block_dim.x << std::endl;

    constexpr auto m = Solver::m_size;
    constexpr auto n = Solver::n_size;

    constexpr bool is_col_maj = arrangement_of_v_a<Solver> == arrangement::col_major;
    constexpr auto lda        = is_col_maj ? m : n;
    constexpr auto input_size = is_col_maj ? lda * n : m * lda;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size);
    common::fillup_random_matrix<data_type>(is_col_maj, m, n, A.data(), lda, false, false, 2, 4); // not symmetric and not diagonally dominant

    std::vector<data_type> L(input_size);
    std::vector<int>       ipiv(min(m, n), 0);
    std::vector<int64_t>   ipiv_ref(min(m, n), 0);
    int                    info   = 0;
    data_type*             d_A    = nullptr; /* device copy of A */
    int*                   d_info = nullptr; /* error info */
    int*                   d_ipiv = nullptr;

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

    /* free resources */
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ipiv));

    //=========================
    // cuSolver reference
    //=========================
    std::vector<data_type> dummy_B;
    common::reference_cusolver_lu<data_type, cuda_data_type>(
        A, dummy_B, &info, m, n, 1, 1 /* padded_batches */, true /* is_pivot */, is_col_maj, true /* is_col_major_b */, false /* is_trans_a */, ipiv_ref.data());


    const auto total_relative_error = common::check_error<data_type, data_type>(L.data(), A.data(), A.size());
    std::cout << "GETRF partial pivoting: relative error of A between cuSolverDx and cuSolver results: " << total_relative_error << std::endl;


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
