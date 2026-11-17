// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// No-pivot tridiagonal solve with CUDA Tile C++.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t n = 32, nrhs = 4;
constexpr float tolerance = 2.e-4f;

__tile_global__ void tridiagonal_solve_kernel(float* dl_data,
                                              float* diag_data,
                                              float* du_data,
                                              float* b_data,
                                              int* info_data) {
    auto dl = ct::partition_view { ct::tensor_span { dl_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto diag = ct::partition_view { ct::tensor_span { diag_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto du = ct::partition_view { ct::tensor_span { du_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<n, nrhs> {} }, ct::shape<n, nrhs> {} };
    auto [x, info] = cusolverdx::tiles::tridiagonal_solve(dl.load(0), diag.load(0), du.load(0), b.load(0, 0));
    b.store(x, 0, 0);
    *info_data = static_cast<int>(info);
}

int main() {
    // Diagonally dominant tridiagonal system, given as its three diagonals
    // rather than a dense matrix. Solve against a known X.
    std::vector<float> dl(n, .25f), diag(n, 4), du(n, -.125f), expected(n * nrhs), b(n * nrhs, 0), x(n * nrhs);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = float(i % 13) / 13;
    }
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < nrhs; ++col) {
            b[row * nrhs + col] += diag[row] * expected[row * nrhs + col];
            if (row) b[row * nrhs + col] += dl[row - 1] * expected[(row - 1) * nrhs + col];
            if (row + 1 < n) b[row * nrhs + col] += du[row] * expected[(row + 1) * nrhs + col];
        }
    }
    device_buffer<float> d_dl(n);
    device_buffer<float> d_diag(n);
    device_buffer<float> d_du(n);
    device_buffer<float> d_b(b.size());
    device_buffer<int> d_info(1);
    d_dl.copy_from_host(dl.data(), n);
    d_diag.copy_from_host(diag.data(), n);
    d_du.copy_from_host(du.data(), n);
    d_b.copy_from_host(b.data(), b.size());
    tridiagonal_solve_kernel<<<1, 1>>>(d_dl.data(), d_diag.data(), d_du.data(), d_b.data(), d_info.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_b.copy_to_host(x.data(), x.size());
    int info = -1;
    d_info.copy_to_host(&info, 1);
    const double error = relative_l2_error(x, expected);
    std::printf("tridiagonal_solve: info=%d, relative solution error=%g\n", info, error);
    return info == 0 && error < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
