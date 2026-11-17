// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// LQ factorization with CUDA Tile C++, covering both ways of using the result:
// forming Q explicitly, and applying Q to another matrix without forming it.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t m = 8, n = 16, nrhs = 4;
constexpr float tolerance = 2.e-4f;

// Overwrites A with the Householder factors and keeps the scalars in tau; both
// kernels below consume that factorization.
__tile_global__ void lq_factor_kernel(float* a_data, float* tau_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto tau = ct::partition_view { ct::tensor_span { tau_data, ct::shape<m> {} }, ct::shape<m> {} };
    auto [factors, taus] = cusolverdx::tiles::lq_factor(a.load(0, 0));
    a.store(factors, 0, 0);
    tau.store(taus, 0);
}

__tile_global__ void lq_generate_q_kernel(float* a_data, float* tau_data, float* q_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto tau = ct::partition_view { ct::tensor_span { tau_data, ct::shape<m> {} }, ct::shape<m> {} };
    auto q = ct::partition_view { ct::tensor_span { q_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    q.store(cusolverdx::tiles::lq_generate_q(a.load(0, 0), tau.load(0)), 0, 0);
}

__tile_global__ void lq_multiply_by_q_kernel(float* a_data, float* tau_data, float* c_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<m, n> {} }, ct::shape<m, n> {} };
    auto tau = ct::partition_view { ct::tensor_span { tau_data, ct::shape<m> {} }, ct::shape<m> {} };
    auto c = ct::partition_view { ct::tensor_span { c_data, ct::shape<n, nrhs> {} }, ct::shape<n, nrhs> {} };
    c.store(
        cusolverdx::tiles::lq_multiply_by_q(a.load(0, 0), tau.load(0), c.load(0, 0), cusolverdx::tiles::left), 0, 0);
}

int main() {
    // Well-conditioned dense wide A, and a dense C for the implicit-Q application.
    std::vector<float> original(m * n), a(m * n), q(m * n), c(n * nrhs), l(m * m, 0), qqt(m * m, 0), identity(m * m, 0),
        reconstructed(m * n, 0);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            original[row * n + col] = (row == col ? 1.0f : 0.0f) + static_cast<float>((row + 3 * col) % 5) / 32.0f;
        }
    }
    for (size_t row = 0; row < n; ++row) {
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
    device_buffer<float> d_tau(m);
    device_buffer<float> d_q(q.size());
    device_buffer<float> d_c(c.size());
    d_a.copy_from_host(a.data(), a.size());
    d_c.copy_from_host(c.data(), c.size());
    lq_factor_kernel<<<1, 1>>>(d_a.data(), d_tau.data());
    lq_generate_q_kernel<<<1, 1>>>(d_a.data(), d_tau.data(), d_q.data());
    lq_multiply_by_q_kernel<<<1, 1>>>(d_a.data(), d_tau.data(), d_c.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_a.copy_to_host(a.data(), a.size());
    d_q.copy_to_host(q.data(), q.size());
    d_c.copy_to_host(c.data(), c.size());

    for (size_t i = 0; i < m; ++i) {
        identity[i * m + i] = 1;
        for (size_t j = 0; j <= i; ++j) {
            l[i * m + j] = a[i * n + j];
        }
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < n; ++k) {
                qqt[i * m + j] += q[i * n + k] * q[j * n + k];
            }
        }
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < m; ++k) {
                reconstructed[i * n + j] += l[i * m + k] * q[k * n + j];
            }
        }
    }

    // Q is orthogonal, so applying it must leave the Frobenius norm of C alone.
    float after = 0;
    for (float value : c) {
        after += value * value;
    }

    const double error = relative_l2_error(reconstructed, original);
    const double orthogonality = relative_l2_error(qqt, identity);
    const float preservation = std::fabs(before - after) / before;
    std::printf("LQ: relative reconstruction error=%g, row-orthogonality error=%g, norm-preservation error=%g\n",
                error,
                orthogonality,
                preservation);
    return error < tolerance && orthogonality < tolerance && preservation < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
