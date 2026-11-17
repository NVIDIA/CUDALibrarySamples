// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Singular values of an upper-bidiagonal matrix with CUDA Tile C++.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t n = 16;
constexpr float tolerance = 1.e-5f;

__tile_global__ void svd_bidiagonal_kernel(float* diag_data, float* off_diag_data, float* s_data, int* info_data) {
    auto diag = ct::partition_view { ct::tensor_span { diag_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto off_diag = ct::partition_view { ct::tensor_span { off_diag_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto s = ct::partition_view { ct::tensor_span { s_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto [values, info] = cusolverdx::tiles::svd_bidiagonal(
        diag.load(0), off_diag.load(0), cusolverdx::tiles::no_vectors, cusolverdx::tiles::no_vectors);
    s.store(values, 0);
    *info_data = static_cast<int>(info);
}

int main() {
    // Upper-bidiagonal B: off_diag[i] is the superdiagonal entry in row i, and
    // the last entry is padding rather than part of B.
    std::vector<float> diag(n), off_diag(n, 0.0f), s(n);
    for (size_t i = 0; i < n; ++i) {
        diag[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i + 1 < n; ++i) {
        off_diag[i] = 0.5f;
    }

    device_buffer<float> d_diag(diag.size());
    device_buffer<float> d_off_diag(off_diag.size());
    device_buffer<float> d_s(s.size());
    device_buffer<int> d_info(1);
    d_diag.copy_from_host(diag.data(), diag.size());
    d_off_diag.copy_from_host(off_diag.data(), off_diag.size());
    svd_bidiagonal_kernel<<<1, 1>>>(d_diag.data(), d_off_diag.data(), d_s.data(), d_info.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int info = -1;
    d_info.copy_to_host(&info, 1);
    d_s.copy_to_host(s.data(), s.size());

    // The squared singular values sum to ||B||_F^2, which pins the spectrum
    // down without needing a reference SVD.
    float frobenius = 0, square_sum = 0;
    bool ordered = true;
    for (size_t i = 0; i < n; ++i) {
        frobenius += diag[i] * diag[i];
        square_sum += s[i] * s[i];
        if (s[i] < 0 || (i && s[i] > s[i - 1])) ordered = false;
    }
    for (size_t i = 0; i + 1 < n; ++i) {
        frobenius += off_diag[i] * off_diag[i];
    }

    const float frobenius_error = std::fabs(frobenius - square_sum) / frobenius;
    std::printf("BDSVD: info=%d, Frobenius-norm error=%g, ordered=%s\n", info, frobenius_error, ordered ? "yes" : "no");
    return info == 0 && frobenius_error < tolerance && ordered ? EXIT_SUCCESS : EXIT_FAILURE;
}
