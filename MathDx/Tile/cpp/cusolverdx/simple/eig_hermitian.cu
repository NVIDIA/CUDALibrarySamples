// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Symmetric eigensolve with CUDA Tile C++.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t n = 16;
constexpr float tolerance = 3.e-4f;

__tile_global__ void eig_hermitian_kernel(float* a_data, float* w_data, float* v_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto w = ct::partition_view { ct::tensor_span { w_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto v = ct::partition_view { ct::tensor_span { v_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto [values, vectors, info] =
        cusolverdx::tiles::eig_hermitian(a.load(0, 0), cusolverdx::tiles::lower, cusolverdx::tiles::all_vectors);
    w.store(values, 0);
    v.store(vectors, 0, 0);
    *info_data = static_cast<int>(info);
}

float norm(const std::vector<float>& x) {
    float sum = 0;
    for (float value : x) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

int main() {
    // Dense symmetric A: a varying diagonal plus an off-diagonal pattern that
    // depends only on row + col, which keeps it symmetric.
    std::vector<float> a(n * n, 0), w(n), v(n * n), residual(n * n, 0), orth(n * n, 0), identity(n * n, 0);
    for (size_t row = 0; row < n; ++row) {
        identity[row * n + row] = 1;
        for (size_t col = 0; col < n; ++col) {
            a[row * n + col] =
                row == col ? 2.f + static_cast<float>(row) / n : static_cast<float>((row + col) % 5) / 16.f - 0.125f;
        }
    }
    device_buffer<float> d_a(a.size());
    device_buffer<float> d_w(w.size());
    device_buffer<float> d_v(v.size());
    device_buffer<int> d_info(1);
    d_a.copy_from_host(a.data(), a.size());
    eig_hermitian_kernel<<<1, 1>>>(d_a.data(), d_w.data(), d_v.data(), d_info.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int info = -1;
    d_w.copy_to_host(w.data(), w.size());
    d_v.copy_to_host(v.data(), v.size());
    d_info.copy_to_host(&info, 1);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t t = 0; t < n; ++t) {
                residual[i * n + j] += a[i * n + t] * v[t * n + j] - v[i * n + t] * (t == j ? w[j] : 0);
                orth[i * n + j] += v[t * n + i] * v[t * n + j];
            }
        }
    }
    for (size_t i = 0; i < n * n; ++i) {
        orth[i] -= identity[i];
    }
    const float error = norm(residual) / norm(a), orthogonality = norm(orth);
    std::printf("eig_hermitian: info=%d, residual=%g, orthogonality error=%g\n", info, error, orthogonality);
    return info == 0 && error < tolerance && orthogonality < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
