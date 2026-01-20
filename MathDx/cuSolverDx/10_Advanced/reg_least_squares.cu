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
#include <cublasdx.hpp>
#include <cusolverDn.h>

#include <type_traits>
#include <iostream>

#include "../common/device_io.hpp"
#include "../common/macros.hpp"
#include "../common/measure.hpp"
#include "../common/random.hpp"

// This example solves a batch of regularized least squares problems
//      minimize ||b - Ax||_2^2 + lambda ||x||_2^2
//
// Three approaches are compared
// 1. Using cuBLASDx and cuSolverDx to build a single fused kernel to solve the normal equations
//      (A' A + lambda I) x = A' b
// 2. Using cuSolverDx to do Householder QR on A augmented by lambda I
// 3. using cuBLAS and cuSolver to solve the normal equations

template<unsigned M, unsigned N, unsigned NT, class T, class R>
__global__ __launch_bounds__(NT) void solve_normal_equations(const T* A, unsigned lda, const T* b, T* x, R lambda, int* info) {

    // To maximize parallelism, we compute [G v] = A' * [A b] with a single GEMM, then solve Gx = v
    // The matrices are assumed to be stored in column-major format
    // (M + N) * (N + 1) words of dynamic shared memory are required

    // Setup shared memory
    extern __shared__ __align__(16) unsigned char shared_mem[];

    // Slice shared memory into pointers
    // Note that in memory, we have the augmented matrices [As bs] and [Gs vs]
    auto [As, bs, Gs, vs] = cusolverdx::shared_memory::slice<T, T, T, T>(
        shared_mem,
        alignof(T), M * N,
        alignof(T), M,
        alignof(T), N * N,
        alignof(T)  // the size (number of elements) may be omitted for the last pointer
    );

    //// Index into correct batch
    const unsigned batch = blockIdx.x;
    A += batch * lda * N;
    b += batch * M;
    x += batch * N;
    info += batch;

    //// Setup operators
    #ifdef __CUDA_ARCH__
        constexpr unsigned Arch = __CUDA_ARCH__;
    #else
        constexpr unsigned Arch = 800;
    #endif

    using          prec = std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, commondx::complex<float>>, float, double>;
    constexpr auto type = (std::is_same_v<T, float> || std::is_same_v<T, double>) ? cublasdx::type::real : cublasdx::type::complex;

    using GEMM = decltype(cublasdx::TransposeMode<cublasdx::C, cublasdx::N>() + cublasdx::Size<N, N+1, M>() + cublasdx::LeadingDimension<M, M, N>() +
                          cublasdx::Precision<prec>() + cublasdx::Type<type>() + cublasdx::Block() + cublasdx::BlockDim<NT>() + cublasdx::SM<Arch>());
    using POSV = decltype(cusolverdx::Function<cusolverdx::posv>() + cusolverdx::Size<N, N, 1>() + cusolverdx::FillMode<cusolverdx::lower>() +
                          cusolverdx::Precision<prec>() + cusolverdx::Type<type>() + cusolverdx::Block() + cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());

    //// Load data from global memory
    cusolverdx::copy_2d<NT, M, N, cusolverdx::arrangement::col_major>(A, lda, As, M);
    cusolverdx::copy_2d<NT, M, 1, cusolverdx::arrangement::col_major>(b, M, bs, M);
    __syncthreads();

    //// Normal equations

    // Takes advantage of the fact that b and v follow A and G
    __syncthreads();
    GEMM().execute(T(1.0), As, As, T(0.0), Gs);
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += NT) {
        Gs[i + i * N] += T(lambda);
    }
    __syncthreads();
    POSV().execute(Gs, vs, info);

    //// store solution back to global memory
    __syncthreads();
    cusolverdx::copy_2d<NT, N, 1, cusolverdx::arrangement::col_major>(vs, N, x, N);
}

template<unsigned M, unsigned N, unsigned NT, class T, class R>
__global__ __launch_bounds__(NT) void solve_householder(const T* A, unsigned lda, const T* b, T* x, R lambda, int* info) {

    // QR requires an extra N rows for the regularization terms

    // Setup shared memory
    extern __shared__ __align__(16) unsigned char shared_mem[];
    // Slice shared memory into pointers (arrays)
    auto [As, taus, bs] = cusolverdx::shared_memory::slice<T, T, T>(
        shared_mem,
        alignof(T), (M + N) * N,
        alignof(T), N,
        alignof(T) // the size (number of elements) may be omitted for the last pointer
    );

    //// Index into correct batch
    unsigned batch = blockIdx.x;
    A += batch * lda * N;
    b += batch * M;
    x += batch * N;
    info += batch;

    //// Setup operators
    #ifdef __CUDA_ARCH__
        constexpr unsigned Arch = __CUDA_ARCH__;
    #else
        constexpr unsigned Arch = 800;
    #endif

    using          prec = std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, commondx::complex<float>>, float, double>;
    constexpr auto type = (std::is_same_v<T, float> || std::is_same_v<T, double>) ? cublasdx::type::real : cublasdx::type::complex;

    using GELS = decltype(cusolverdx::Function<cusolverdx::gels>() + cusolverdx::Size<M+N, N, 1>() + cusolverdx::Precision<prec>() + cusolverdx::Type<type>() +
                          cusolverdx::Block() + cusolverdx::BlockDim<NT>() + cusolverdx::SM<Arch>());

    //// Load data from global memory
    cusolverdx::copy_2d<NT, M, N, cusolverdx::arrangement::col_major>(A, lda, As, M+N);
    cusolverdx::copy_2d<NT, M, 1, cusolverdx::arrangement::col_major>(b, M, bs, M+N);
    __syncthreads();

    // Fill the bottom block
    for (int ij = threadIdx.x; ij < N*N; ij += NT) {
        int i = ij % N;
        int j = ij / N;

        As[(M + i) + j * (M+N)] = T(i == j ? lambda : 0);
    }
    for (int i = threadIdx.x; i < N; i += NT) {
        bs[M + i] = 0;
    }
    __syncthreads();

    // Solve least squares problem
    GELS().execute(As, M+N, taus, bs);

    //// store solution back to global memory
    __syncthreads();
    cusolverdx::copy_2d<NT, N, 1, cusolverdx::arrangement::col_major>(bs, M+N, x, N);
}


// Helper kernel to apply the regularization term to the Gram matrix
template<unsigned N, class T, class R>
__global__ void increment_diagonal(T** G, int ldg, R lambda, unsigned batches) {
    // BlockDim.x is assumed to be N

    int start = threadIdx.y + blockDim.y * blockIdx.x;
    int step = blockDim.y * gridDim.x;
    for (int batch = start; batch < batches; batch += step) {
        G[batch][threadIdx.x * (ldg+1)] += T(lambda);
    }
}

// Solves the normal equations using the host API
template<unsigned M, unsigned N, class T, class R>
void cusolver_normal_equations(cublasHandle_t blas_hand, cusolverDnHandle_t solv_hand, cudaStream_t stream, T** A, unsigned lda, T** b, T** x, T** G, unsigned ldg, R lambda, int* info, int batches) {

    static_assert(N <= 256);
    dim3 diag_blockdim = dim3(N, 256/N);
    dim3 diag_griddim = (batches - 1) / diag_blockdim.y + 1;

    if constexpr (std::is_same_v<T, float>) {
        T alpha = 1.0f;
        T beta  = 0.0f;
        CUBLAS_CHECK_AND_EXIT(cublasSgemmBatched(blas_hand, CUBLAS_OP_C, CUBLAS_OP_N, N, N, M,
                    &alpha, A, lda,
                            A, lda,
                    &beta,  G, ldg,
                    batches));
        CUBLAS_CHECK_AND_EXIT(cublasSgemvBatched(blas_hand, CUBLAS_OP_C, M, N,
                    &alpha, A, lda,
                            b, 1,
                    &beta,  x, 1,
                    batches));
        increment_diagonal<N, T><<<diag_griddim, diag_blockdim, 0, stream>>>(G, ldg, lambda, batches);
        CUSOLVER_CHECK_AND_EXIT(cusolverDnSpotrfBatched(solv_hand, CUBLAS_FILL_MODE_LOWER, N, G, ldg, info, batches));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnSpotrsBatched(solv_hand, CUBLAS_FILL_MODE_LOWER, N, 1, G, N, x, N, info, batches));
    } else if constexpr (std::is_same_v<T, double>) {
        T alpha = 1.0;
        T beta  = 0.0;
        CUBLAS_CHECK_AND_EXIT(cublasDgemmBatched(blas_hand, CUBLAS_OP_C, CUBLAS_OP_N, N, N, M,
                    &alpha, A, lda,
                            A, lda,
                    &beta,  G, ldg,
                    batches));
        CUBLAS_CHECK_AND_EXIT(cublasDgemvBatched(blas_hand, CUBLAS_OP_C, M, N,
                    &alpha, A, lda,
                            b, 1,
                    &beta,  x, 1,
                    batches));
        increment_diagonal<N, T><<<diag_griddim, diag_blockdim, 0, stream>>>(G, ldg, lambda, batches);
        CUSOLVER_CHECK_AND_EXIT(cusolverDnDpotrfBatched(solv_hand, CUBLAS_FILL_MODE_LOWER, N, G, ldg, info, batches));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnDpotrsBatched(solv_hand, CUBLAS_FILL_MODE_LOWER, N, 1, G, N, x, N, info, batches));
    } else if constexpr (std::is_same_v<T, commondx::complex<float>>) {
        T alpha = {1.0f, 0.0f};
        T beta  = {0.0f, 0.0f};
        CUBLAS_CHECK_AND_EXIT(cublasCgemmBatched(blas_hand, CUBLAS_OP_C, CUBLAS_OP_N, N, N, M,
                    &alpha, A, lda,
                            A, lda,
                    &beta,  G, ldg,
                    batches));
        CUBLAS_CHECK_AND_EXIT(cublasCgemvBatched(blas_hand, CUBLAS_OP_C, M, N,
                    &alpha, A, lda,
                            b, 1,
                    &beta,  x, 1,
                    batches));
        increment_diagonal<N, T><<<diag_griddim, diag_blockdim, 0, stream>>>(G, ldg, lambda, batches);
        CUSOLVER_CHECK_AND_EXIT(cusolverDnCpotrfBatched(solv_hand, CUBLAS_FILL_MODE_LOWER, N, G, ldg, info, batches));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnCpotrsBatched(solv_hand, CUBLAS_FILL_MODE_LOWER, N, 1, G, N, x, N, info, batches));
    } else if constexpr (std::is_same_v<T, commondx::complex<double>>) {
        T alpha = {1.0f, 0.0f};
        T beta  = {0.0f, 0.0f};
        CUBLAS_CHECK_AND_EXIT(cublasZgemmBatched(blas_hand, CUBLAS_OP_C, CUBLAS_OP_N, N, N, M,
                    &alpha, A, lda,
                            A, lda,
                    &beta,  G, ldg,
                    batches));
        CUBLAS_CHECK_AND_EXIT(cublasZgemvBatched(blas_hand, CUBLAS_OP_C, M, N,
                    &alpha, A, lda,
                            b, 1,
                    &beta,  x, 1,
                    batches));
        increment_diagonal<N, T><<<diag_griddim, diag_blockdim, 0, stream>>>(G, ldg, lambda, batches);
        CUSOLVER_CHECK_AND_EXIT(cusolverDnZpotrfBatched(solv_hand, CUBLAS_FILL_MODE_LOWER, N, G, ldg, info, batches));
        CUSOLVER_CHECK_AND_EXIT(cusolverDnZpotrsBatched(solv_hand, CUBLAS_FILL_MODE_LOWER, N, 1, G, N, x, N, info, batches));
    }
}

template<class T, class R>
double compute_residual_squared(int m, int n, const T* b, const T* A, int lda, const T* x, R lambda) {
    // Compute ||b - Ax||_2^2 + \lambda||x||_2^2
    double nrm2 = 0;
    // residual term
    for (int i = 0; i < m; ++i) {
        T ri = b[i];

        for (int j = 0; j < n; ++j) {
            ri -= A[i + j * lda] * x[j];
        }
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            nrm2 += ri.real() * ri.real() + ri.imag() * ri.imag();
        } else {
            nrm2 += ri * ri;
        }
    }
    // regularization term
    for (int j = 0; j < n; ++j) {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            nrm2 += lambda * (x[j].real() * x[j].real() + x[j].imag() * x[j].imag());
        } else {
            nrm2 += lambda * x[j] * x[j];
        }
    }
    return nrm2;
}

int main() {

    // Configurable parameters
    using data_type            = float; // Use commondx::complex<float> and commondx::complex<double> for complex types
    using real_type            = common::get_precision_t<data_type>;
    constexpr unsigned m       = 100;
    constexpr unsigned n       = 8;
    constexpr unsigned lda     = m;
    constexpr unsigned batches = 10000;
    const real_type lambda     = 0.1;

    // Number of threads to use per thread block for the dx implementations
    constexpr unsigned ne_NT   = 32;
    constexpr unsigned qr_NT   = 32;

    static_assert(m >= n, "Normal equations assume that m >= n");
    static_assert(lda >= m, "Leading dimension of A must be at least m");

    std::cout << "Solving " << batches << " " << m << "x" << n << " regularized least square problems" << std::endl;

    cudaStream_t stream = nullptr;
    CUDA_CHECK_AND_EXIT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<data_type> A(m * n * batches);
    std::vector<data_type> b(m * batches);
    std::vector<data_type> x(n * batches);
    std::vector<data_type> x_qr(n * batches);
    std::vector<int> info(batches);
    common::fillup_random_matrix_col_major<data_type>(m, n * batches, A.data(), lda, false, false, -2, 2);
    common::fillup_random_matrix_col_major<data_type>(m, 1 * batches, b.data(), lda, false, false, -2, 2);


    data_type *d_A, *d_b, *d_x, *d_G;
    int* d_info;
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_b), sizeof(data_type) * b.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_x), sizeof(data_type) * x.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int) * batches));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_G), sizeof(data_type) * n * n * batches)); // Workspace for host implementation

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_b, b.data(), sizeof(data_type) * b.size(), cudaMemcpyHostToDevice, stream));

    // Configure and invoke normal equations kernel
    constexpr unsigned ne_required_smem = (m + n) * (n + 1) * sizeof(data_type);
    const auto ne_kernel = solve_normal_equations<m, n, ne_NT, data_type, real_type>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(ne_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, ne_required_smem));
    const auto run_ne_d_impl = [&](cudaStream_t str) {
        ne_kernel<<<batches, ne_NT, ne_required_smem, str>>>(d_A, lda, d_b, d_x, lambda, d_info);
    };
    run_ne_d_impl(stream);

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(x.data(), d_x, sizeof(data_type) * x.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(info.data(), d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    for (int i = 0; i < info.size(); ++i) {
        if (0 < info[i]) {
            std::printf("Cholesky failed in %dth batch with info %d", i, info[i]);
        }
    }

    // Configure and invoke normal equations kernel
    constexpr unsigned qr_required_smem = ((m + n) * (n + 1) + n) * sizeof(data_type);
    const auto qr_kernel = solve_householder<m, n, qr_NT, data_type, real_type>;
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(qr_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, qr_required_smem));
    const auto run_qr_d_impl = [&](cudaStream_t str) {
        qr_kernel<<<batches, qr_NT, qr_required_smem, str>>>(d_A, lda, d_b, d_x, lambda, d_info);
    };
    run_qr_d_impl(stream);

    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(x_qr.data(), d_x, sizeof(data_type) * x_qr.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));


    //=========================
    // cuSolver reference
    //=========================

    cublasHandle_t blas_hand = NULL;
    CUBLAS_CHECK_AND_EXIT(cublasCreate(&blas_hand));
    CUBLAS_CHECK_AND_EXIT(cublasSetStream(blas_hand, stream));
    cusolverDnHandle_t solv_hand = NULL;
    CUSOLVER_CHECK_AND_EXIT(cusolverDnCreate(&solv_hand));
    CUSOLVER_CHECK_AND_EXIT(cusolverDnSetStream(solv_hand, stream));

    // Setup batched arrays
    std::vector<data_type*> Aarr (batches);
    std::vector<data_type*> barr (batches);
    std::vector<data_type*> xarr (batches);
    std::vector<data_type*> Garr (batches);
    for (int i = 0; i < batches; ++i) {
        Aarr[i] = d_A + i * m * n;
        barr[i] = d_b + i * m;
        xarr[i] = d_x + i * n;
        Garr[i] = d_G + i * n * n;
    }
    data_type **d_Aarr, **d_barr, **d_xarr, **d_Garr;
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_Aarr), sizeof(data_type*) * Aarr.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_barr), sizeof(data_type*) * barr.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_xarr), sizeof(data_type*) * xarr.size()));
    CUDA_CHECK_AND_EXIT(cudaMalloc(reinterpret_cast<void**>(&d_Garr), sizeof(data_type*) * Garr.size()));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_Aarr, Aarr.data(), sizeof(data_type*) * Aarr.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_barr, barr.data(), sizeof(data_type*) * barr.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_xarr, xarr.data(), sizeof(data_type*) * xarr.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(d_Garr, Garr.data(), sizeof(data_type*) * Garr.size(), cudaMemcpyHostToDevice, stream));

    // Run cusolver version for correctness test
    const auto run_ne_h_impl = [&](cudaStream_t str) {
        cusolver_normal_equations<m, n, data_type>(blas_hand, solv_hand, str, d_Aarr, lda, d_barr, d_xarr, d_Garr, n, lambda, d_info, batches);
    };
    run_ne_h_impl(stream);

    std::vector<data_type> x_host(n * batches);
    CUDA_CHECK_AND_EXIT(cudaMemcpyAsync(x_host.data(), d_x, sizeof(data_type) * x.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));

    // Compute ||b - Ax||_2^2 + \lambda||x||_2^2 for each solver
    double ne_d_nrm2 = compute_residual_squared(m, n, b.data(), A.data(), lda, x.data(), lambda);
    double qr_d_nrm2 = compute_residual_squared(m, n, b.data(), A.data(), lda, x_qr.data(), lambda);
    double ne_h_nrm2 = compute_residual_squared(m, n, b.data(), A.data(), lda, x_host.data(), lambda);

    // Compare the performance of the two implementations
    const unsigned int warmup_repeats = 1;
    const unsigned int kernel_repeats = 3;
    double ne_d_ms = common::measure::execution(run_ne_d_impl, warmup_repeats, kernel_repeats, stream) / kernel_repeats;
    double qr_d_ms = common::measure::execution(run_qr_d_impl, warmup_repeats, kernel_repeats, stream) / kernel_repeats;
    double ne_h_ms = common::measure::execution(run_ne_h_impl, warmup_repeats, kernel_repeats, stream) / kernel_repeats;

    constexpr double ne_flop_per_batch = common::get_flops_gemm<data_type>(m, n, n) + n + common::get_flops_potrf<data_type>(n) + common::get_flops_potrs<data_type>(n, 1);
    constexpr double qr_flop_per_batch = common::get_flops_geqrf<data_type>(m+n, n) + common::get_flops_unmqr<data_type>(cusolverdx::side::left, m+n, 1, n) + common::get_flops_trsm<data_type>(n, 1);
    double ne_d_s_per_g_batch = ne_d_ms / 1e3 / batches * 1e9;
    double qr_d_s_per_g_batch = qr_d_ms / 1e3 / batches * 1e9;
    double ne_h_s_per_g_batch = ne_h_ms / 1e3 / batches * 1e9;
    double ne_d_gflop_per_s   = ne_flop_per_batch / ne_d_s_per_g_batch;
    double qr_d_gflop_per_s   = qr_flop_per_batch / qr_d_s_per_g_batch;
    double ne_h_gflop_per_s   = ne_flop_per_batch / ne_h_s_per_g_batch;

    printf("Dx normal equations   %7.2f GFLOP/s, %7.2f ms, objective %e\n", ne_d_gflop_per_s, ne_d_ms, ne_d_nrm2);
    printf("Dx Householder        %7.2f GFLOP/s, %7.2f ms, objective %e\n", qr_d_gflop_per_s, qr_d_ms, qr_d_nrm2);
    printf("Host normal equations %7.2f GFLOP/s, %7.2f ms, objective %e\n", ne_h_gflop_per_s, ne_h_ms, ne_h_nrm2);

    // free device resources
    CUDA_CHECK_AND_EXIT(cudaFree(d_A));
    CUDA_CHECK_AND_EXIT(cudaFree(d_b));
    CUDA_CHECK_AND_EXIT(cudaFree(d_x));
    CUDA_CHECK_AND_EXIT(cudaFree(d_G));
    CUDA_CHECK_AND_EXIT(cudaFree(d_Aarr));
    CUDA_CHECK_AND_EXIT(cudaFree(d_barr));
    CUDA_CHECK_AND_EXIT(cudaFree(d_xarr));
    CUDA_CHECK_AND_EXIT(cudaFree(d_Garr));
    CUDA_CHECK_AND_EXIT(cudaFree(d_info));

    CUBLAS_CHECK_AND_EXIT(cublasDestroy(blas_hand));
    CUSOLVER_CHECK_AND_EXIT(cusolverDnDestroy(solv_hand));
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    CUDA_CHECK_AND_EXIT(cudaDeviceReset());

    return 0;
}
