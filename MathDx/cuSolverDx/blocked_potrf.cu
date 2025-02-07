// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <iostream>
#include <type_traits>

#include <cublasdx.hpp>
#include <cusolverdx.hpp>
#include "common.hpp"

// This is a basic implementation of Cholesky factorization using a single CTA with matrices too large for cuSolverDx's shared memory API.
// The code uses an out-of-core style, left-looking formulation so that it never needs more than 4 blocks in shared memory at a given point.

// There are a few limitations of this implementation:
// * Only real-valued types are supported
// * The matrix size, N, must be a multiple of the block size, NB
// * Only upper-triangular storage is implemented


//////// Helper functions to load and store the diagonal blocks //////
// Only the upper triangular part is transferred
// To help L1 cache utilization, the diagonal blocks bypass the L1 cache

template<unsigned NB, cusolverdx::arrangement Arrange, unsigned NT, class T>
inline __device__ void load_diagonal_block(const T* A, const int lda, T* As, const int ldas) {
    const int tid = threadIdx.x;
    __builtin_assume(tid < NT);

    if constexpr (NT % NB == 0) {
        constexpr unsigned stride_jj = NT / NB;
        const unsigned i = tid % NB;
        const unsigned j = tid / NB;
        for (int jj = 0; jj < NB; jj += stride_jj) {
            bool is_upper_tri = (Arrange == cusolverdx::col_major) ? (i <= j+jj) : (i >= j+jj);
            if (is_upper_tri) {
                As[i + (jj+j)*ldas] = __ldcg(A + i + (jj+j)*lda);
            }
        }
    } else {
        for (int k = tid; k < NB*NB; k += NT) {
            unsigned i = k % NB;
            unsigned j = k / NB;
            bool is_upper_tri = (Arrange == cusolverdx::col_major) ? (i <= j) : (i >= j);
            if (is_upper_tri) {
                As[i + j*ldas] = __ldcg(A + i + j*lda);
            }
        }
    }
    __syncthreads();
}
template<unsigned NB, cusolverdx::arrangement Arrange, unsigned NT, class T>
inline __device__ void store_diagonal_block(const T* As, const int ldas, T* A, const int lda) {
    const int tid = threadIdx.x;
    __builtin_assume(tid < NT);

    __syncthreads();
    if constexpr (NT % NB == 0) {
        constexpr unsigned stride_jj = NT / NB;
        const unsigned i = tid % NB;
        const unsigned j = tid / NB;
        for (int jj = 0; jj < NB; jj += stride_jj) {
            bool is_upper_tri = (Arrange == cusolverdx::col_major) ? (i <= j+jj) : (i >= j+jj);
            if (is_upper_tri) {
                __stcg(A + i + (jj+j)*lda, As[i + (jj+j)*ldas]);
            }
        }
    } else {
        for (int k = tid; k < NB*NB; k += NT) {
            unsigned i = k % NB;
            unsigned j = k / NB;
            bool is_upper_tri = (Arrange == cusolverdx::col_major) ? (i <= j) : (i >= j);
            if (is_upper_tri) {
                __stcg(A + i + j*lda, As[i + j*ldas]);
            }
        }
    }
}

//////// Blocked Cholesky Implementation ////////

template<unsigned NB, cusolverdx::arrangement Arrange, class T>
__device__ T* tile(T* A, unsigned lda, unsigned i, unsigned j) {
    if (Arrange == cusolverdx::col_major) {
        return A + i*NB + j*NB*lda;
    } else {
        return A + i*NB*lda + j*NB;
    }
}

template<unsigned N, unsigned NB, cusolverdx::arrangement Arrange, unsigned NT, class T>
__global__ void potrf_kernel(T* A, unsigned lda, int* info) {
    // Assumes A is stored in the upper triangle

    static_assert(N % NB == 0, "Code currently assumes that N is exactly divisible by the block size");
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "cuSolverDx calls currently hardcoded for real valued type");

    // Get batch
    A    += blockIdx.x * N * lda;
    info += blockIdx.x * N;

    constexpr unsigned lds = NB;

    __shared__ T sA[NB*lds];
    __shared__ T sB[NB*lds];
    __shared__ T sC[NB*lds];
    __shared__ T sD[NB*lds];
    __shared__ int sinfo;

    #ifdef __CUDA_ARCH__
        constexpr unsigned Arch = __CUDA_ARCH__;
    #else
        constexpr unsigned Arch = 700;
    #endif

    using POTRF = decltype(cusolverdx::Function<cusolverdx::function::potrf>() + cusolverdx::FillMode<cusolverdx::fill_mode::upper>() + cusolverdx::Size<NB>() +
                           cusolverdx::Precision<T>() + cusolverdx::Type<cusolverdx::type::real>() + cusolverdx::Arrangement<Arrange>() +
                           cusolverdx::Block() + cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());
    // TRSM is temporarily being exported from cuSolverDx instead of the cuBLASDx
    // Arrangement is flipped to handle transposition
    constexpr auto trans_arrange = (Arrange == cusolverdx::arrangement::col_major) ? cusolverdx::arrangement::row_major : cusolverdx::arrangement::col_major;
    using TRSM = decltype(cusolverdx::Function<cusolverdx::function::trsm>() + cusolverdx::Size<NB, NB, NB>() +
                          cusolverdx::Precision<T>() + cusolverdx::Type<cusolverdx::type::real>() +
                          cusolverdx::Arrangement<trans_arrange, Arrange>() + cusolverdx::FillMode<cusolverdx::fill_mode::lower>() +
                          cusolverdx::Block() + cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());
    // Have to use Arrangement to set transposition, to work around cuBLASDx's lack of support for using TransposeMode with a row-major C
    using GEMM_Arrange = std::conditional_t<Arrange == cusolverdx::col_major,
                                            cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major, cublasdx::col_major>,
                                            cublasdx::Arrangement<cublasdx::col_major, cublasdx::row_major, cublasdx::row_major>>;
    constexpr unsigned int alignment = ((sizeof(T) * NB * NB) % 16 == 0) ? 16 : sizeof(T);
    using GEMM = decltype(cublasdx::Size<NB, NB, NB>() + GEMM_Arrange() + cublasdx::Alignment<alignment, alignment, alignment>() +
                          cublasdx::Precision<T>() + cublasdx::Type<cublasdx::type::real>() +
                          cublasdx::Block() + cublasdx::BlockDim<NT>() + cublasdx::SM<Arch>());

    // left-looking, out of core algorithm

    unsigned n_tiles = N / NB;
    int rinfo = 0;


    for (int k = 0; k < n_tiles; ++k) {
        // Diagonal block
        auto Akk = tile<NB, Arrange>(A, lda, k, k);
        load_diagonal_block<NB, Arrange, NT>(Akk, lda, sA, lds);
        // Schur-complements previous steps
        for (int i = 0; i < k; ++i) {
            auto Aik = tile<NB, Arrange>(A, lda, i, k);
            common::io<POTRF>::load(Aik, lda, sC, lds);
            GEMM().execute(T(-1.0), sC, sC, T(1.0), sA);
            __syncthreads();
        }
        // Factor block
        POTRF().execute(sA, lds, &sinfo);
        store_diagonal_block<NB, Arrange, NT>(sA, lds, Akk, lda);
        if (threadIdx.x == 0 && rinfo == 0 && sinfo != 0) {
            rinfo = sinfo + k*NB;
        }

        // Panel
        for (int j = k+1; j < n_tiles; ++j) {
            auto Akj = tile<NB, Arrange>(A, lda, k, j);
            common::io<POTRF>::load(Akj, lda, sB, lds);

            for (int i = 0; i < k; ++i) {
                auto Aik = tile<NB, Arrange>(A, lda, i, k);
                auto Aij = tile<NB, Arrange>(A, lda, i, j);
                common::io<POTRF>::load(Aik, lda, sC, lds);
                common::io<POTRF>::load(Aij, lda, sD, lds);
                GEMM().execute(T(-1.0), sC, sD, T(1.0), sB);
                __syncthreads();
            }

            TRSM().execute(sA, lds, sB, lds);
            common::io<POTRF>::store(sB, lds, Akj, lda);
            __syncthreads();
        }
    }


    if (threadIdx.x == 0) {
        *info = rinfo;
    }
}

int blocked_potrf() {

    // Only float and double are supported
    using data_type = double;

    constexpr unsigned N       = 512;                   // The matrix size
    constexpr unsigned NB      = 32;                    // The blocking size to use
    constexpr unsigned NT      = 128;                   // The number of threads to use per thread block
    constexpr auto     Arrange = cusolverdx::row_major; // Whether the matrices are column-major or row-major
    const     unsigned batches = 200;

    constexpr auto lda        = N;       // this is the leading dimension in global memory for A
    constexpr auto input_size = lda * N; // input global memory size for A

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(input_size * batches);
    common::fillup_random_diagonal_dominant_matrix<data_type>(Arrange == cusolverdx::col_major, N, N, A.data(), lda, false, -2, 2, batches); // input A is not symmetric

    //printf("A = \n");
    //common::print_matrix(N, N*batches, A.data(), lda);
    //printf("=====\n");

    std::vector<data_type> L(input_size * batches);
    std::vector<int>       info(batches);
    data_type*             d_A    = nullptr; /* device copy of A */
    int*                   d_info = nullptr; /* error info */

    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * info.size()));

    size_t smem_size = 4 * NB * NB * sizeof(data_type) + sizeof(int);
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(potrf_kernel<N, NB, Arrange, NT, data_type>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    //Invokes kernel
    auto run_kernel = [&](cudaStream_t str) {
        potrf_kernel<N, NB, Arrange, NT, data_type><<<batches, NT, smem_size, str>>>(d_A, lda, d_info);
    };
    auto reset = [&] (cudaStream_t str) {
        CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, str));
        CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(str));
    };
    const unsigned int warmup_repeats = 1;
    const unsigned int repeats        = 1;

    reset(stream);
    double ms                     = common::measure::execution(run_kernel, reset, warmup_repeats, repeats, stream) / repeats;
    double seconds_per_giga_batch = ms / 1e3 / batches * 1e9;
    double gb_s                   = input_size * sizeof(data_type) * 2 / seconds_per_giga_batch;
    double gflops                 = common::get_flops_potrf<data_type>(N) / seconds_per_giga_batch;

    common::print_perf("Blocked dx potrf", batches, N, N, 1, gflops, gb_s, ms);

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(L.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int) * info.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    for (int i = 0; i < info.size(); ++i) {
        if (0 > info[i]) {
            std::printf("%d-th parameter is wrong \n", -info[i]);
            exit(1);
        }
    }
    //printf("L = \n");
    //common::print_matrix(N, N*batches, L.data(), lda);

    //=========================
    // cuSolver reference
    //=========================
    std::vector<data_type> B;  // dummy B
    common::reference_cusolver_cholesky<data_type, data_type, true>(
        A, B, info.data(), N, 1, batches, false, Arrange == cusolverdx::col_major, false, false, batches); // is_lower?, is_column major a? is_column major b, do_solve?
    //printf("L ref = \n");
    //common::print_matrix(N, N*batches, A.data(), lda);

    //=========================
    // Compare results
    //=========================
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

int main() { return blocked_potrf(); }
