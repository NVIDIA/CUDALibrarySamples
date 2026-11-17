// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Factor once, then reuse Cholesky and pivoted-LU factors in later Tile kernels.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t n = 32, nrhs = 4;
constexpr float tolerance = 2.e-4f;

__tile_global__ void cholesky_factor_kernel(float* a_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto [factors, info] = cusolverdx::tiles::cholesky_factor(a.load(0, 0), cusolverdx::tiles::lower);
    a.store(factors, 0, 0);
    *info_data = static_cast<int>(info);
}

__tile_global__ void cholesky_solve_kernel(float* a_data, float* b_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<n, nrhs> {} }, ct::shape<n, nrhs> {} };
    b.store(cusolverdx::tiles::cholesky_solve_using_factor(a.load(0, 0), b.load(0, 0), cusolverdx::tiles::lower), 0, 0);
}

__tile_global__ void lu_factor_kernel(float* a_data, int* pivots_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto pivots = ct::partition_view { ct::tensor_span { pivots_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto [factors, permutation, info] = cusolverdx::tiles::lu_factor(a.load(0, 0));
    a.store(factors, 0, 0);
    pivots.store(permutation, 0);
    *info_data = static_cast<int>(info);
}

__tile_global__ void lu_solve_kernel(float* a_data, int* pivots_data, float* b_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto pivots = ct::partition_view { ct::tensor_span { pivots_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<n, nrhs> {} }, ct::shape<n, nrhs> {} };
    b.store(cusolverdx::tiles::lu_solve_using_factor(a.load(0, 0), pivots.load(0), b.load(0, 0)), 0, 0);
}

int main() {
    // One dense SPD matrix drives both paths: Cholesky needs the definiteness,
    // LU just needs it to be nonsingular. Solve against a known X.
    std::vector<float> expected(n * nrhs), b(n * nrhs);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = float(i % 11) / 11;
    }
    auto a = make_spd_matrix(n);
    const auto original = a;
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < nrhs; ++col) {
            for (size_t t = 0; t < n; ++t) {
                b[row * nrhs + col] += a[row * n + t] * expected[t * nrhs + col];
            }
        }
    }
    device_buffer<float> d_a(a.size());
    device_buffer<float> d_b(b.size());
    device_buffer<int> d_pivots(n);
    device_buffer<int> d_info(1);
    d_a.copy_from_host(a.data(), a.size());
    d_b.copy_from_host(b.data(), b.size());
    cholesky_factor_kernel<<<1, 1>>>(d_a.data(), d_info.data());
    CUDA_CHECK(cudaGetLastError());
    cholesky_solve_kernel<<<1, 1>>>(d_a.data(), d_b.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int cholesky_info = -1;
    d_info.copy_to_host(&cholesky_info, 1);
    std::vector<float> x(n * nrhs);
    d_b.copy_to_host(x.data(), x.size());
    const double cholesky_error = relative_l2_error(x, expected);
    a = original;
    std::fill(b.begin(), b.end(), 0);
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < nrhs; ++col) {
            for (size_t t = 0; t < n; ++t) {
                b[row * nrhs + col] += a[row * n + t] * expected[t * nrhs + col];
            }
        }
    }
    d_a.copy_from_host(a.data(), a.size());
    d_b.copy_from_host(b.data(), b.size());
    lu_factor_kernel<<<1, 1>>>(d_a.data(), d_pivots.data(), d_info.data());
    lu_solve_kernel<<<1, 1>>>(d_a.data(), d_pivots.data(), d_b.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int lu_info = -1;
    d_info.copy_to_host(&lu_info, 1);
    d_b.copy_to_host(x.data(), x.size());
    const double lu_error = relative_l2_error(x, expected);
    std::printf("factor reuse: Cholesky info=%d, error=%g; LU info=%d, error=%g\n",
                cholesky_info,
                cholesky_error,
                lu_info,
                lu_error);
    const bool succeeded = cholesky_info == 0 && lu_info == 0 && cholesky_error < tolerance && lu_error < tolerance;
    return succeeded ? EXIT_SUCCESS : EXIT_FAILURE;
}
