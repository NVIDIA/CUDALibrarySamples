// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SVD of a small dense matrix with CUDA Tile C++, computed twice from the
// same input: singular values only, and singular values with vectors.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t m = 16, n = 8, r = 8;
constexpr float tolerance = 1.e-4f;

__tile_global__ void svd_values_kernel(float* a_data, float* s_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto s = ct::partition_view { ct::tensor_span { s_data, ct::shape<r> {} }, ct::shape<r> {} };
    auto [values, info] =
        cusolverdx::tiles::svd(a.load(0, 0), cusolverdx::tiles::no_vectors, cusolverdx::tiles::no_vectors);
    s.store(values, 0);
    *info_data = static_cast<int>(info);
}

__tile_global__ void svd_vectors_kernel(float* a_data, float* u_data, float* s_data, float* vt_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto u = ct::partition_view { ct::tensor_span { u_data, ct::shape<m, r> {} }, ct::shape<m, r> {} };
    auto s = ct::partition_view { ct::tensor_span { s_data, ct::shape<r> {} }, ct::shape<r> {} };
    auto vt = ct::partition_view { ct::tensor_span { vt_data, ct::shape<r, n> {} }, ct::shape<r, n> {} };
    auto [left_vectors, values, right_vectors, info] =
        cusolverdx::tiles::svd(a.load(0, 0), cusolverdx::tiles::some_vectors, cusolverdx::tiles::some_vectors);
    u.store(left_vectors, 0, 0);
    s.store(values, 0);
    vt.store(right_vectors, 0, 0);
    *info_data = static_cast<int>(info);
}

int main() {
    // Dense, full-rank A so that the singular vectors are well determined.
    std::vector<float> a(m * n), values(r), u(m * r), s(r), vt(r * n), reconstructed(m * n, 0);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            a[row * n + col] = (row == col ? 4.0f : 0.0f) + static_cast<float>((row + 2 * col) % 7) / 4.0f;
        }
    }

    device_buffer<float> d_a(a.size());
    device_buffer<float> d_values(values.size());
    device_buffer<float> d_u(u.size());
    device_buffer<float> d_s(s.size());
    device_buffer<float> d_vt(vt.size());
    device_buffer<int> d_values_info(1);
    device_buffer<int> d_vectors_info(1);
    d_a.copy_from_host(a.data(), a.size());
    svd_values_kernel<<<1, 1>>>(d_a.data(), d_values.data(), d_values_info.data());
    CUDA_CHECK(cudaGetLastError());
    svd_vectors_kernel<<<1, 1>>>(d_a.data(), d_u.data(), d_s.data(), d_vt.data(), d_vectors_info.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int values_info = -1, vectors_info = -1;
    d_values_info.copy_to_host(&values_info, 1);
    d_vectors_info.copy_to_host(&vectors_info, 1);
    d_values.copy_to_host(values.data(), values.size());
    d_u.copy_to_host(u.data(), u.size());
    d_s.copy_to_host(s.data(), s.size());
    d_vt.copy_to_host(vt.data(), vt.size());

    // Both jobs see the same input, so they must agree on the singular values.
    const double agreement = relative_l2_error(s, values);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < r; ++k) {
                reconstructed[i * n + j] += u[i * r + k] * s[k] * vt[k * n + j];
            }
        }
    }
    const double error = relative_l2_error(reconstructed, a);
    std::printf("SVD: info=%d/%d, singular-value agreement=%g, relative reconstruction error=%g\n",
                values_info,
                vectors_info,
                agreement,
                error);
    return values_info == 0 && vectors_info == 0 && agreement < tolerance && error < tolerance ? EXIT_SUCCESS
                                                                                               : EXIT_FAILURE;
}
