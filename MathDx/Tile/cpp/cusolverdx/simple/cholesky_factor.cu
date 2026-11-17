// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A single-CTA Cholesky factorization using the CUDA Tile C++ cuSOLVERDx API.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t n = 32;
constexpr float tolerance = 1.e-4f;

__tile_global__ void cholesky_factor_kernel(float* a_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto [factor, info] = cusolverdx::tiles::cholesky_factor(a.load(0, 0), cusolverdx::tiles::lower);
    a.store(factor, 0, 0);
    *info_data = static_cast<int>(info);
}

// Relative error of the L L^T reconstruction, using only the lower triangle of the factor.
float reconstruction_error(const std::vector<float>& a, const std::vector<float>& l) {
    float error = 0.0f, norm = 0.0f;
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float reconstructed = 0.0f;
            for (size_t k = 0; k <= (row < col ? row : col); ++k) {
                reconstructed += l[row * n + k] * l[col * n + k];
            }
            const float delta = a[row * n + col] - reconstructed;
            error += delta * delta;
            norm += a[row * n + col] * a[row * n + col];
        }
    }
    return std::sqrt(error / norm);
}

int main() {
    const auto original = make_spd_matrix(n);
    device_buffer<float> d_a(original.size());
    device_buffer<int> d_info(1);
    d_a.copy_from_host(original.data(), original.size());

    cholesky_factor_kernel<<<1, 1>>>(d_a.data(), d_info.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> factor(n * n);
    int info = -1;
    d_a.copy_to_host(factor.data(), factor.size());
    d_info.copy_to_host(&info, 1);

    const float error = reconstruction_error(original, factor);
    std::printf("cholesky_factor: info=%d, relative reconstruction error=%g\n", info, error);
    return info == 0 && error < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
