// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Overdetermined least-squares solve with CUDA Tile C++, expressed both ways:
// against a tall A, and against the transpose of the equivalent wide matrix.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t m = 32, n = 16, nrhs = 4;
constexpr float tolerance = 2.e-4f;

__tile_global__ void least_squares_solve_kernel(float* a_data, float* b_data, float* x_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<m, nrhs> {} }, ct::shape<m, nrhs> {} };
    auto x = ct::partition_view { ct::tensor_span { x_data, ct::shape<n, nrhs> {} }, ct::shape<n, nrhs> {} };
    auto [factors, tau, solution] = cusolverdx::tiles::least_squares_solve(a.load(0, 0), b.load(0, 0));
    x.store(solution, 0, 0);
}

// The same problem given the wide matrix instead: op(A) is again m x n, so this
// consumes the same B and produces the same X.
__tile_global__ void least_squares_solve_transpose_kernel(float* at_data, float* b_data, float* x_data) {
    auto at = ct::partition_view { ct::tensor_span { at_data, ct::shape<n, m> {} }, ct::shape<n, m> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<m, nrhs> {} }, ct::shape<m, nrhs> {} };
    auto x = ct::partition_view { ct::tensor_span { x_data, ct::shape<n, nrhs> {} }, ct::shape<n, nrhs> {} };
    auto [factors, tau, solution] =
        cusolverdx::tiles::least_squares_solve(at.load(0, 0), b.load(0, 0), cusolverdx::tiles::trans);
    x.store(solution, 0, 0);
}

int main() {
    // Dense full-rank tall A, plus its explicit transpose. Building B from a
    // known X makes the overdetermined system consistent, so the least-squares
    // solution is that X.
    std::vector<float> a(m * n), at(n * m), expected(n * nrhs), b(m * nrhs, 0), x(n * nrhs), transposed_x(n * nrhs);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            a[row * n + col] = (row == col ? 1.f : 0.f) + static_cast<float>((row + 2 * col) % 7) / 64.f;
            at[col * m + row] = a[row * n + col];
        }
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<float>(i % 9) / 9.f;
    }
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < nrhs; ++col) {
            for (size_t k = 0; k < n; ++k) {
                b[row * nrhs + col] += a[row * n + k] * expected[k * nrhs + col];
            }
        }
    }

    device_buffer<float> d_a(a.size());
    device_buffer<float> d_at(at.size());
    device_buffer<float> d_b(b.size());
    device_buffer<float> d_x(x.size());
    device_buffer<float> d_transposed_x(transposed_x.size());
    d_a.copy_from_host(a.data(), a.size());
    d_at.copy_from_host(at.data(), at.size());
    d_b.copy_from_host(b.data(), b.size());
    least_squares_solve_kernel<<<1, 1>>>(d_a.data(), d_b.data(), d_x.data());
    least_squares_solve_transpose_kernel<<<1, 1>>>(d_at.data(), d_b.data(), d_transposed_x.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_x.copy_to_host(x.data(), x.size());
    d_transposed_x.copy_to_host(transposed_x.data(), transposed_x.size());

    // Both calls describe the same problem, so they must agree on the solution.
    const double error = relative_l2_error(x, expected);
    const double transposed_error = relative_l2_error(transposed_x, expected);
    const double agreement = relative_l2_error(transposed_x, x);
    std::printf("least_squares_solve: relative solution error=%g, transposed=%g, agreement=%g\n",
                error,
                transposed_error,
                agreement);
    return error < tolerance && transposed_error < tolerance && agreement < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
