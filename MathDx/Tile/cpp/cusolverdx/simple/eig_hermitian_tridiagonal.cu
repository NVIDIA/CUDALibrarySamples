// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Eigenvalues of a symmetric tridiagonal matrix with CUDA Tile C++.
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

__tile_global__ void eig_hermitian_tridiagonal_kernel(float* diag_data,
                                                      float* off_diag_data,
                                                      float* w_data,
                                                      int* info_data) {
    auto diag = ct::partition_view { ct::tensor_span { diag_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto off_diag = ct::partition_view { ct::tensor_span { off_diag_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto w = ct::partition_view { ct::tensor_span { w_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto [values, info] =
        cusolverdx::tiles::eig_hermitian_tridiagonal(diag.load(0), off_diag.load(0), cusolverdx::tiles::no_vectors);
    w.store(values, 0);
    *info_data = static_cast<int>(info);
}

int main() {
    // Symmetric tridiagonal T with a varying diagonal and a constant
    // off-diagonal. The last off_diag entry is padding, not part of T.
    std::vector<float> diag(n), off_diag(n, 0.0f), w(n);
    for (size_t i = 0; i < n; ++i) {
        diag[i] = 2.0f + static_cast<float>(i) / static_cast<float>(n);
    }
    for (size_t i = 0; i + 1 < n; ++i) {
        off_diag[i] = 0.25f;
    }

    device_buffer<float> d_diag(diag.size());
    device_buffer<float> d_off_diag(off_diag.size());
    device_buffer<float> d_w(w.size());
    device_buffer<int> d_info(1);
    d_diag.copy_from_host(diag.data(), diag.size());
    d_off_diag.copy_from_host(off_diag.data(), off_diag.size());
    eig_hermitian_tridiagonal_kernel<<<1, 1>>>(d_diag.data(), d_off_diag.data(), d_w.data(), d_info.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int info = -1;
    d_info.copy_to_host(&info, 1);
    d_w.copy_to_host(w.data(), w.size());

    // An eigendecomposition preserves the trace and the Frobenius norm, which
    // pins the spectrum down without needing a reference eigensolver.
    float trace = 0, frobenius = 0, value_sum = 0, value_square_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        trace += diag[i];
        frobenius += diag[i] * diag[i];
        value_sum += w[i];
        value_square_sum += w[i] * w[i];
    }
    for (size_t i = 0; i + 1 < n; ++i) {
        frobenius += 2 * off_diag[i] * off_diag[i];
    }

    const float trace_error = std::fabs(trace - value_sum) / trace;
    const float frobenius_error = std::fabs(frobenius - value_square_sum) / frobenius;
    std::printf("eig_hermitian_tridiagonal: info=%d, trace error=%g, Frobenius-norm error=%g\n",
                info,
                trace_error,
                frobenius_error);
    return info == 0 && trace_error < tolerance && frobenius_error < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
