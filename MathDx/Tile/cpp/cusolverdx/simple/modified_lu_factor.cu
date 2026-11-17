// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Modified LU with CUDA Tile C++: the no-pivot factorization A - diag(S) = L U.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t m = 16, n = 8;
constexpr float tolerance = 1.e-4f;

__tile_global__ void modified_lu_factor_kernel(float* a_data, float* s_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto s = ct::partition_view { ct::tensor_span { s_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto [factors, shifts] = cusolverdx::tiles::modified_lu_factor(a.load(0, 0));
    a.store(factors, 0, 0);
    s.store(shifts, 0);
}

int main() {
    // Dense tall A whose leading n x n block is diagonally dominant, since this
    // factorization does not pivot.
    std::vector<float> original(m * n), a(m * n), s(n), expected(m * n), reconstructed(m * n, 0);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            original[row * n + col] =
                (row == col ? 4.0f : 0.0f) + static_cast<float>((row + 2 * col) % 5) / 8.0f - 0.25f;
        }
    }
    a = original;

    device_buffer<float> d_a(a.size());
    device_buffer<float> d_s(s.size());
    d_a.copy_from_host(a.data(), a.size());
    modified_lu_factor_kernel<<<1, 1>>>(d_a.data(), d_s.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_a.copy_to_host(a.data(), a.size());
    d_s.copy_to_host(s.data(), s.size());

    // L is unit lower trapezoidal and U upper triangular, both packed into a.
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            for (size_t k = 0; k <= (row < col ? row : col); ++k) {
                const float l = k == row ? 1.0f : a[row * n + k];
                reconstructed[row * n + col] += l * a[k * n + col];
            }
        }
    }
    expected = original;
    for (size_t i = 0; i < n; ++i) {
        expected[i * n + i] -= s[i];
    }

    const double error = relative_l2_error(reconstructed, expected);
    std::printf("modified_lu_factor: relative reconstruction error=%g\n", error);
    return error < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
