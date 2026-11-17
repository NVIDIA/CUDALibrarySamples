// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// QR factorization with CUDA Tile C++, covering both ways of using the result:
// forming the thin Q explicitly, and applying Q to another matrix without ever
// forming it.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t m = 32, n = 16, nrhs = 4;
constexpr float tolerance = 2.e-4f;

// Overwrites A with the Householder factors and keeps the scalars in tau; both
// kernels below consume that factorization.
__tile_global__ void qr_factor_kernel(float* a_data, float* tau_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto tau = ct::partition_view { ct::tensor_span { tau_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto [factors, taus] = cusolverdx::tiles::qr_factor(a.load(0, 0));
    a.store(factors, 0, 0);
    tau.store(taus, 0);
}

__tile_global__ void qr_generate_q_kernel(float* a_data, float* tau_data, float* q_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto tau = ct::partition_view { ct::tensor_span { tau_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto q = ct::partition_view { ct::tensor_span { q_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    q.store(cusolverdx::tiles::qr_generate_q(a.load(0, 0), tau.load(0)), 0, 0);
}

__tile_global__ void qr_multiply_by_q_kernel(float* a_data, float* tau_data, float* c_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto tau = ct::partition_view { ct::tensor_span { tau_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto c = ct::partition_view { ct::tensor_span { c_data, ct::shape<m, nrhs> {} }, ct::shape<m, nrhs> {} };
    c.store(
        cusolverdx::tiles::qr_multiply_by_q(a.load(0, 0), tau.load(0), c.load(0, 0), cusolverdx::tiles::left), 0, 0);
}

int main() {
    // Well-conditioned dense A, and a dense C for the implicit-Q application.
    std::vector<float> original(m * n), a(m * n), q(m * n), c(m * nrhs), r(n * n, 0), qtq(n * n, 0), identity(n * n, 0),
        reconstructed(m * n, 0);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            original[row * n + col] = (row == col ? 1.0f : 0.0f) + static_cast<float>((row + 3 * col) % 5) / 32.0f;
        }
    }
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < nrhs; ++col) {
            c[row * nrhs + col] = 1.0f + static_cast<float>((row + 2 * col) % 7) / 8.0f;
        }
    }
    a = original;

    float before = 0;
    for (float value : c) {
        before += value * value;
    }

    device_buffer<float> d_a(a.size());
    device_buffer<float> d_tau(n);
    device_buffer<float> d_q(q.size());
    device_buffer<float> d_c(c.size());
    d_a.copy_from_host(a.data(), a.size());
    d_c.copy_from_host(c.data(), c.size());
    qr_factor_kernel<<<1, 1>>>(d_a.data(), d_tau.data());
    qr_generate_q_kernel<<<1, 1>>>(d_a.data(), d_tau.data(), d_q.data());
    qr_multiply_by_q_kernel<<<1, 1>>>(d_a.data(), d_tau.data(), d_c.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_a.copy_to_host(a.data(), a.size());
    d_q.copy_to_host(q.data(), q.size());
    d_c.copy_to_host(c.data(), c.size());

    for (size_t i = 0; i < n; ++i) {
        identity[i * n + i] = 1;
        for (size_t j = i; j < n; ++j) {
            r[i * n + j] = a[i * n + j];
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < m; ++k) {
                qtq[i * n + j] += q[k * n + i] * q[k * n + j];
            }
        }
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                reconstructed[i * n + j] += q[i * n + k] * r[k * n + j];
            }
        }
    }

    // Q is orthogonal, so applying it must leave the Frobenius norm of C alone.
    float after = 0;
    for (float value : c) {
        after += value * value;
    }

    const double error = relative_l2_error(reconstructed, original);
    const double orthogonality = relative_l2_error(qtq, identity);
    const float preservation = std::fabs(before - after) / before;
    std::printf("QR: relative reconstruction error=%g, orthogonality error=%g, norm-preservation error=%g\n",
                error,
                orthogonality,
                preservation);
    return error < tolerance && orthogonality < tolerance && preservation < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
