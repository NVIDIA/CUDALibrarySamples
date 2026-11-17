// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Blocked Cholesky factorization with CUDA Tile C++, for matrices far larger
// than a single solver call can hold.
//
// This mirrors the classic cuSOLVERDx example of the same name
// (example/cusolverdx/10_Advanced/blocked_potrf.cu), including its problem
// sizes. One block factors one 512x512 matrix in a left-looking order of 32x32 tiles.
// Only a few tiles are live at a time:
//
//     for each diagonal tile k:
//         A[k][k] -= sum over i < k of A[i][k]^T A[i][k]
//         U[k][k]  = cholesky_factor(A[k][k])
//         for each panel tile j > k:
//             A[k][j] -= sum over i < k of A[i][k]^T A[i][j]
//             U[k][j]  = triangular_solve(U[k][k]^T X = A[k][j])
//
// A tile is a value. The working set contains only the named tiles below, and the compiler places them.
// The kernel needs no manual shared-memory slicing, blocked load and store helpers, or barriers.
//
// Only the upper triangle is read and written, and the factor satisfies
// U^T U = A. Results are checked against batched cuSolver.

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "common_perf_measure.hpp"
#include "cublasdx_tile.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

// Sizes taken from the example this mirrors. Both are powers of two, as tile
// extents must be, and n is an exact multiple of the tile size.
using value_type = double;
constexpr size_t n = 512, nb = 32, tiles = n / nb, batches = 400;
constexpr double tolerance = 1.e-12;

// Left-looking blocked Cholesky over the upper triangle, one matrix per block.
// The batch is folded into the row-tile coordinate: matrix `batch` owns tile
// rows [batch * tiles, (batch + 1) * tiles).
__tile_global__ void blocked_potrf_kernel(value_type* a_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<batches * n, n> {} }, ct::shape<nb, nb> {} };
    const unsigned batch = ct::bid().x;
    const unsigned base = batch * tiles;

    auto status = ct::zeros<ct::tile<int, ct::shape<>>>();
    for (unsigned k = 0; k < tiles; ++k) {
        // Diagonal tile, brought up to date with every panel already computed.
        auto akk = a.load(base + k, k);
        for (unsigned i = 0; i < k; ++i) {
            const auto aik = a.load(base + i, k);
            akk = akk - ct::matmul(ct::transpose(aik), aik);
        }
        auto [ukk, info] = cusolverdx::tiles::cholesky_factor(akk, cusolverdx::tiles::upper);
        a.store(ukk, base + k, k);
        status = status + info;

        // Row panel to the right of the diagonal tile.
        for (unsigned j = k + 1; j < tiles; ++j) {
            auto akj = a.load(base + k, j);
            for (unsigned i = 0; i < k; ++i) {
                akj = akj - ct::matmul(ct::transpose(a.load(base + i, k)), a.load(base + i, j));
            }
            a.store(cublasdx::tiles::triangular_solve(ukk,
                                                      akj,
                                                      cublasdx::tiles::upper,
                                                      cublasdx::tiles::non_unit,
                                                      cublasdx::tiles::left,
                                                      cublasdx::tiles::trans),
                    base + k,
                    j);
        }
    }
    info_data[batch] = static_cast<int>(status);
}

// Relative Frobenius-norm difference over the upper triangles only; the rest of
// each buffer is untouched scratch that the two implementations disagree on.
double upper_triangle_error(const std::vector<value_type>& expected, const std::vector<value_type>& actual) {
    double error = 0.0, norm = 0.0;
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t row = 0; row < n; ++row) {
            for (size_t col = row; col < n; ++col) {
                const size_t index = batch * n * n + row * n + col;
                const double delta = expected[index] - actual[index];
                error += delta * delta;
                norm += expected[index] * expected[index];
            }
        }
    }
    return std::sqrt(error / norm);
}

int main() {
    // Symmetric diagonally dominant, hence positive definite. Only the upper
    // triangle is ever read, but making the matrix symmetric keeps the
    // row-major tile view and the column-major cuSolver view identical.
    std::vector<value_type> a(batches * n * n);
    for (size_t batch = 0; batch < batches; ++batch) {
        value_type* matrix = a.data() + batch * n * n;
        for (size_t row = 0; row < n; ++row) {
            matrix[row * n + row] = static_cast<value_type>(n);
            for (size_t col = row + 1; col < n; ++col) {
                const auto value = static_cast<value_type>((row * 7 + col * 13 + batch * 3) % 11) / 11.0 - 0.5;
                matrix[row * n + col] = value;
                matrix[col * n + row] = value;
            }
        }
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const size_t bytes = a.size() * sizeof(value_type);
    device_buffer<value_type> d_a(a.size());
    device_buffer<value_type> d_reference(a.size());
    device_buffer<value_type> d_pristine(a.size());
    device_buffer<int> d_info(batches);
    device_buffer<int> d_reference_info(batches);
    d_pristine.copy_from_host(a.data(), a.size());

    // Pointer array for the batched cuSolver call.
    std::vector<value_type*> reference_pointers(batches);
    for (size_t batch = 0; batch < batches; ++batch) {
        reference_pointers[batch] = d_reference.data() + batch * n * n;
    }
    device_buffer<value_type*> d_reference_pointers(batches);
    d_reference_pointers.copy_from_host(reference_pointers.data(), reference_pointers.size());

    cusolverDnHandle_t solver = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&solver));
    CUSOLVER_CHECK(cusolverDnSetStream(solver, stream));

    const auto reset_tile = [&](cudaStream_t str) {
        CUDA_CHECK(cudaMemcpyAsync(d_a.data(), d_pristine.data(), bytes, cudaMemcpyDeviceToDevice, str));
    };
    const auto run_tile = [&](cudaStream_t str) {
        blocked_potrf_kernel<<<batches, 1, 0, str>>>(d_a.data(), d_info.data());
        CUDA_CHECK(cudaGetLastError());
    };

    // The buffer is symmetric, so cuSolver reading it as column-major sees the
    // same matrix. Its lower triangle is this buffer's upper triangle, so
    // FILL_MODE_LOWER produces exactly the factor the tile kernel writes.
    const auto reset_reference = [&](cudaStream_t str) {
        CUDA_CHECK(cudaMemcpyAsync(d_reference.data(), d_pristine.data(), bytes, cudaMemcpyDeviceToDevice, str));
    };
    const auto run_reference = [&](cudaStream_t str) {
        CUSOLVER_CHECK(cusolverDnSetStream(solver, str));
        CUSOLVER_CHECK(cusolverDnDpotrfBatched(
            solver, CUBLAS_FILL_MODE_LOWER, n, d_reference_pointers.data(), n, d_reference_info.data(), batches));
    };

    const double tile_ms = measure(run_tile, reset_tile, stream);
    const double reference_ms = measure(run_reference, reset_reference, stream);

    std::vector<value_type> factor(a.size()), reference(a.size());
    std::vector<int> info(batches, -1), reference_info(batches, -1);
    d_a.copy_to_host(factor.data(), factor.size());
    d_reference.copy_to_host(reference.data(), reference.size());
    d_info.copy_to_host(info.data(), info.size());
    d_reference_info.copy_to_host(reference_info.data(), reference_info.size());

    CUSOLVER_CHECK(cusolverDnDestroy(solver));
    CUDA_CHECK(cudaStreamDestroy(stream));

    bool factored = true;
    for (size_t batch = 0; batch < batches; ++batch) {
        factored = factored && info[batch] == 0 && reference_info[batch] == 0;
    }

    const auto gflops = [](double flops_per_batch, double ms) { return flops_per_batch / (ms / 1e3 / batches * 1e9); };
    const double error = upper_triangle_error(reference, factor);

    std::printf("Factoring %zu %zux%zu matrices in %zux%zu tiles\n", batches, n, n, nb, nb);
    std::printf("Blocked dx potrf %8.2f GFLOP/s, %8.2f ms\n", gflops(flops_potrf(n), tile_ms), tile_ms);
    std::printf("cuSolver batched %8.2f GFLOP/s, %8.2f ms\n", gflops(flops_potrf(n), reference_ms), reference_ms);
    std::printf("Relative error of U against cuSolver: %g\n", error);
    return factored && error < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
