// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Solve A X = B for a positive-definite A with the CUDA Tile C++ cuSOLVERDx API.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t n = 32;
constexpr size_t nrhs = 4;
constexpr float tolerance = 1.e-4f;

__tile_global__ void cholesky_solve_kernel(float* a_data, float* b_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<n, nrhs> {} }, ct::shape<n, nrhs> {} };
    auto [x, info] = cusolverdx::tiles::cholesky_solve(a.load(0, 0), b.load(0, 0), cusolverdx::tiles::lower);
    b.store(x, 0, 0);
    *info_data = static_cast<int>(info);
}

std::vector<float> multiply(const std::vector<float>& a, const std::vector<float>& x) {
    std::vector<float> b(n * nrhs, 0.0f);
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < nrhs; ++col) {
            for (size_t k = 0; k < n; ++k) {
                b[row * nrhs + col] += a[row * n + k] * x[k * nrhs + col];
            }
        }
    }
    return b;
}

int main() {
    // Solve against a known X so the error below measures the solver, not the data.
    const auto a = make_spd_matrix(n);
    std::vector<float> expected(n * nrhs);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = 0.1f + static_cast<float>(i % 11) / 11.0f;
    }
    const auto b = multiply(a, expected);
    device_buffer<float> d_a(a.size());
    device_buffer<float> d_b(b.size());
    device_buffer<int> d_info(1);
    d_a.copy_from_host(a.data(), a.size());
    d_b.copy_from_host(b.data(), b.size());
    cholesky_solve_kernel<<<1, 1>>>(d_a.data(), d_b.data(), d_info.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> x(n * nrhs);
    int info = -1;
    d_b.copy_to_host(x.data(), x.size());
    d_info.copy_to_host(&info, 1);
    const double error = relative_l2_error(x, expected);
    std::printf("cholesky_solve: info=%d, relative solution error=%g\n", info, error);
    return info == 0 && error < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
