// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Batched regularized least squares with CUDA Tile C++:
//
//     minimize ||b - A x||^2 + lambda ||x||^2
//
// This mirrors the classic cuSOLVERDx example of the same name
// (example/cusolverdx/10_Advanced/reg_least_squares.cu), including its problem
// sizes, and compares the same three approaches:
//
//   1. A fused tile kernel forms (A^T A + lambda I) x = A^T b with tile matmuls.
//      It factors the normal equations with Cholesky. The Gram matrix never reaches global memory.
//   2. A tile kernel runs Householder QR on [A; sqrt(lambda) I] x ~ [b; 0].
//      This costs more but does not square the condition number.
//   3. The batched cuBLAS and cuSolver host APIs solve the same normal equations.
//      They provide a reference for correctness and throughput.
//
// One block solves one problem: the tile kernels index their batch through the
// partition coordinates of a view that spans the whole batch.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "common.hpp"
#include "common_perf_measure.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

// Sizes taken from the advanced cuSolverDx example, reg_least_squares.cu.
constexpr size_t m = 100, n = 8, nrhs = 1, batches = 10000;
// Tile extents must be powers of two, so the m x n system is loaded as a
// 128 x 8 tile whose trailing rows are zero. Zero rows contribute nothing to
// A^T A or A^T b, and they leave room for the regularization block that the QR
// path writes at rows m .. m + n. Only the tensor extent spanning the batch is
// free of that restriction, which is what lets the batch stay at 10000.
constexpr size_t m_tile = 128;
constexpr float lambda = 0.1f;
// Independent float implementations of the same strictly convex problem, so
// agreement is limited by conditioning rather than by either algorithm.
constexpr float tolerance = 1.e-4f;

using gram_tile_t = ct::tile<float, ct::shape<n, n>>;
using data_tile_t = ct::tile<float, ct::shape<m_tile, n>>;

// Fused solve: [G v] = A^T [A b], G += lambda I, then Cholesky-solve G x = v.
__tile_global__ void normal_equations_kernel(float* a_data,
                                             float* b_data,
                                             float* x_data,
                                             float lambda,
                                             int* info_data) {
    const unsigned batch = ct::bid().x;
    auto a =
        ct::partition_view { ct::tensor_span { a_data, ct::shape<batches * m_tile, n> {} }, ct::shape<m_tile, n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<batches * m_tile, nrhs> {} },
                                  ct::shape<m_tile, nrhs> {} };
    auto x = ct::partition_view { ct::tensor_span { x_data, ct::shape<batches * n, nrhs> {} }, ct::shape<n, nrhs> {} };
    const auto a_tile = a.load(batch, 0);
    const auto a_transposed = ct::transpose(a_tile);

    // lambda on the diagonal and zero elsewhere, from a row-major iota.
    const auto index = ct::iota<ct::tile<int, ct::shape<n, n>>>();
    const auto regularizer =
        ct::select(index / int(n) == index % int(n), ct::full<gram_tile_t>(lambda), ct::zeros<gram_tile_t>());

    const auto g = ct::matmul(a_transposed, a_tile) + regularizer;
    const auto v = ct::matmul(a_transposed, b.load(batch, 0));
    auto [solution, info] = cusolverdx::tiles::cholesky_solve(g, v, cusolverdx::tiles::lower);
    x.store(solution, batch, 0);
    info_data[batch] = static_cast<int>(info);
}

// The same problem as a plain least-squares solve on the augmented system,
// whose normal equations are exactly the ones above. `scale` is sqrt(lambda):
// augmenting with sqrt(lambda) I is what makes ||[A; scale I] x - [b; 0]||^2
// equal the regularized objective.
__tile_global__ void householder_kernel(float* a_data, float* b_data, float* x_data, float scale) {
    const unsigned batch = ct::bid().x;
    auto a =
        ct::partition_view { ct::tensor_span { a_data, ct::shape<batches * m_tile, n> {} }, ct::shape<m_tile, n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<batches * m_tile, nrhs> {} },
                                  ct::shape<m_tile, nrhs> {} };
    auto x = ct::partition_view { ct::tensor_span { x_data, ct::shape<batches * n, nrhs> {} }, ct::shape<n, nrhs> {} };

    // Write scale on the diagonal of the block starting at row m. The rows of
    // A past m are already zero and b is padded with zeros to match, so the
    // augmented system is assembled without touching memory.
    const auto index = ct::iota<ct::tile<int, ct::shape<m_tile, n>>>();
    const auto regularizer =
        ct::select(index / int(n) - int(m) == index % int(n), ct::full<data_tile_t>(scale), ct::zeros<data_tile_t>());

    const auto augmented = a.load(batch, 0) + regularizer;
    auto [factors, tau, solution] = cusolverdx::tiles::least_squares_solve(augmented, b.load(batch, 0));
    x.store(solution, batch, 0);
}

// Applies the regularization to the batch of Gram matrices computed by cuBLAS.
// blockDim.x is n, so each thread owns one diagonal entry.
__global__ void increment_diagonal_kernel(float** g, int ldg, float lambda, int count) {
    const int step = blockDim.y * gridDim.x;
    for (int batch = threadIdx.y + blockDim.y * blockIdx.x; batch < count; batch += step) {
        g[batch][threadIdx.x * (ldg + 1)] += lambda;
    }
}

// Reference path: batched GEMM for the Gram matrices, batched GEMV for A^T b,
// then a batched Cholesky factor and solve. cuBLAS is column-major, so this
// consumes its own copy of A rather than the row-major layout the tile kernels
// read.
void host_normal_equations(cublasHandle_t blas,
                           cusolverDnHandle_t solver,
                           cudaStream_t stream,
                           float** a_array,
                           int lda,
                           float** b_array,
                           float** x_array,
                           float** g_array,
                           int* info,
                           int* status) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmBatched(blas,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    n,
                                    n,
                                    m,
                                    &alpha,
                                    const_cast<const float**>(a_array),
                                    lda,
                                    const_cast<const float**>(a_array),
                                    lda,
                                    &beta,
                                    g_array,
                                    n,
                                    batches));
    CUBLAS_CHECK(cublasSgemvBatched(blas,
                                    CUBLAS_OP_T,
                                    m,
                                    n,
                                    &alpha,
                                    const_cast<const float**>(a_array),
                                    lda,
                                    const_cast<const float**>(b_array),
                                    1,
                                    &beta,
                                    x_array,
                                    1,
                                    batches));
    const dim3 block(n, 256 / n);
    const dim3 grid((batches - 1) / block.y + 1);
    increment_diagonal_kernel<<<grid, block, 0, stream>>>(g_array, n, lambda, batches);
    CUDA_CHECK(cudaGetLastError());
    CUSOLVER_CHECK(cusolverDnSpotrfBatched(solver, CUBLAS_FILL_MODE_LOWER, n, g_array, n, info, batches));
    CUSOLVER_CHECK(
        cusolverDnSpotrsBatched(solver, CUBLAS_FILL_MODE_LOWER, n, nrhs, g_array, n, x_array, n, status, batches));
}

// ||b - A x||^2 + lambda ||x||^2, summed over the batch.
double objective(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& x) {
    double total = 0.0;
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t row = 0; row < m; ++row) {
            double residual = b[batch * m + row];
            for (size_t k = 0; k < n; ++k) {
                residual -= a[batch * m * n + row * n + k] * x[batch * n + k];
            }
            total += residual * residual;
        }
        for (size_t k = 0; k < n; ++k) {
            total += lambda * x[batch * n + k] * x[batch * n + k];
        }
    }
    return total;
}

int main() {
    // Random A and b over [-2, 2], as in the example this mirrors, but from a
    // fixed seed so the run is reproducible.
    std::mt19937 engine(1234);
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    std::vector<float> a(batches * m * n), b(batches * m);
    for (auto& value : a) {
        value = distribution(engine);
    }
    for (auto& value : b) {
        value = distribution(engine);
    }

    // Row-major, zero-padded to m_tile rows, for the tile kernels.
    std::vector<float> a_tiles(batches * m_tile * n, 0.0f), b_tiles(batches * m_tile * nrhs, 0.0f);
    // Column-major with lda = m, for cuBLAS and cuSolver.
    std::vector<float> a_columns(batches * m * n), b_columns(batches * m);
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t row = 0; row < m; ++row) {
            for (size_t col = 0; col < n; ++col) {
                const float value = a[batch * m * n + row * n + col];
                a_tiles[(batch * m_tile + row) * n + col] = value;
                a_columns[batch * m * n + row + col * m] = value;
            }
            b_tiles[(batch * m_tile + row) * nrhs] = b[batch * m + row];
            b_columns[batch * m + row] = b[batch * m + row];
        }
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    device_buffer<float> d_a_tiles(a_tiles.size());
    device_buffer<float> d_b_tiles(b_tiles.size());
    device_buffer<float> d_a_columns(a_columns.size());
    device_buffer<float> d_b_columns(b_columns.size());
    device_buffer<float> d_x(batches * n);
    device_buffer<float> d_x_qr(batches * n);
    device_buffer<float> d_x_host(batches * n);
    device_buffer<float> d_g(batches * n * n);
    device_buffer<int> d_info(batches);
    device_buffer<int> d_host_info(batches);
    device_buffer<int> d_status(1);
    d_a_tiles.copy_from_host(a_tiles.data(), a_tiles.size());
    d_b_tiles.copy_from_host(b_tiles.data(), b_tiles.size());
    d_a_columns.copy_from_host(a_columns.data(), a_columns.size());
    d_b_columns.copy_from_host(b_columns.data(), b_columns.size());

    // Pointer arrays for the batched host APIs.
    std::vector<float*> a_pointers(batches), b_pointers(batches), x_pointers(batches), g_pointers(batches);
    for (size_t batch = 0; batch < batches; ++batch) {
        a_pointers[batch] = d_a_columns.data() + batch * m * n;
        b_pointers[batch] = d_b_columns.data() + batch * m;
        x_pointers[batch] = d_x_host.data() + batch * n;
        g_pointers[batch] = d_g.data() + batch * n * n;
    }
    device_buffer<float*> d_a_pointers(batches);
    device_buffer<float*> d_b_pointers(batches);
    device_buffer<float*> d_x_pointers(batches);
    device_buffer<float*> d_g_pointers(batches);
    d_a_pointers.copy_from_host(a_pointers.data(), a_pointers.size());
    d_b_pointers.copy_from_host(b_pointers.data(), b_pointers.size());
    d_x_pointers.copy_from_host(x_pointers.data(), x_pointers.size());
    d_g_pointers.copy_from_host(g_pointers.data(), g_pointers.size());

    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetStream(blas, stream));
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));

    const float scale = std::sqrt(lambda);
    const auto run_normal_equations = [&](cudaStream_t str) {
        normal_equations_kernel<<<batches, 1, 0, str>>>(
            d_a_tiles.data(), d_b_tiles.data(), d_x.data(), lambda, d_info.data());
    };
    const auto run_householder = [&](cudaStream_t str) {
        householder_kernel<<<batches, 1, 0, str>>>(d_a_tiles.data(), d_b_tiles.data(), d_x_qr.data(), scale);
    };
    const auto run_host = [&](cudaStream_t str) {
        host_normal_equations(blas,
                              solver,
                              str,
                              d_a_pointers.data(),
                              m,
                              d_b_pointers.data(),
                              d_x_pointers.data(),
                              d_g_pointers.data(),
                              d_host_info.data(),
                              d_status.data());
    };

    const double normal_ms = measure(run_normal_equations, stream);
    const double householder_ms = measure(run_householder, stream);
    const double host_ms = measure(run_host, stream);

    std::vector<float> x(batches * n), x_qr(batches * n), x_host(batches * n);
    std::vector<int> info(batches, -1), host_info(batches, -1);
    int status = -1;
    d_x.copy_to_host(x.data(), x.size());
    d_x_qr.copy_to_host(x_qr.data(), x_qr.size());
    d_x_host.copy_to_host(x_host.data(), x_host.size());
    d_info.copy_to_host(info.data(), info.size());
    d_host_info.copy_to_host(host_info.data(), host_info.size());
    d_status.copy_to_host(&status, 1);

    CUBLAS_CHECK(cublasDestroy(blas));
    CUSOLVER_CHECK(cusolverDnDestroy(solver));
    CUDA_CHECK(cudaStreamDestroy(stream));

    bool factored = status == 0;
    for (size_t batch = 0; batch < batches; ++batch) {
        factored = factored && info[batch] == 0 && host_info[batch] == 0;
    }

    // Counted on the useful m x n work, not on the padding the tile kernels carry.
    constexpr double normal_flops =
        flops_gemm(m, n, n) + 2.0 * m * n * nrhs + n + flops_potrf(n) + flops_potrs(n, nrhs);
    constexpr double householder_flops = flops_geqrf(m + n, n) + flops_unmqr_left(m + n, nrhs, n) + flops_trsm(n, nrhs);
    const auto gflops = [](double flops_per_batch, double ms) { return flops_per_batch / (ms / 1e3 / batches * 1e9); };

    std::printf("Solving %zu regularized %zux%zu least squares problems, lambda=%g\n", batches, m, n, lambda);
    std::printf("Dx normal equations   %7.2f GFLOP/s, %7.2f ms, objective %e\n",
                gflops(normal_flops, normal_ms),
                normal_ms,
                objective(a, b, x));
    std::printf("Dx Householder        %7.2f GFLOP/s, %7.2f ms, objective %e\n",
                gflops(householder_flops, householder_ms),
                householder_ms,
                objective(a, b, x_qr));
    std::printf("Host normal equations %7.2f GFLOP/s, %7.2f ms, objective %e\n",
                gflops(normal_flops, host_ms),
                host_ms,
                objective(a, b, x_host));

    // All three minimize the same strictly convex objective, so the reference
    // solution pins down both tile kernels.
    const double normal_error = relative_l2_error(x, x_host);
    const double householder_error = relative_l2_error(x_qr, x_host);
    std::printf("Relative error against the host reference: normal equations=%g, Householder=%g\n",
                normal_error,
                householder_error);
    return factored && normal_error < tolerance && householder_error < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
