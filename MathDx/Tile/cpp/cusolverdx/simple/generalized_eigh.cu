// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Generalized symmetric-definite eigenvalues, A x = lambda B x, solved two
// ways with CUDA Tile C++: in one call, and by reducing to a standard
// eigenproblem by hand.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common.hpp"
#include "cusolverdx_tile.hpp"

namespace ct = cuda::tiles;

constexpr size_t n = 16;
constexpr float tolerance = 1.e-4f;

__tile_global__ void generalized_eigh_kernel(float* a_data, float* b_data, float* w_data, int* info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto w = ct::partition_view { ct::tensor_span { w_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto [values, info] = cusolverdx::tiles::generalized_eig_hermitian(a.load(0, 0),
                                                                       b.load(0, 0),
                                                                       cusolverdx::tiles::eig_type<1>,
                                                                       cusolverdx::tiles::lower,
                                                                       cusolverdx::tiles::no_vectors);
    w.store(values, 0);
    *info_data = static_cast<int>(info);
}

// The same eigenvalues via the steps the call above performs internally:
// factor B, reduce A to a standard eigenproblem, then solve that.
__tile_global__ void generalized_to_standard_eigh_kernel(float* a_data,
                                                         float* b_data,
                                                         float* w_data,
                                                         int* cholesky_info_data,
                                                         int* eigh_info_data) {
    auto a = ct::partition_view { ct::tensor_span { a_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto b = ct::partition_view { ct::tensor_span { b_data, ct::shape<n, n> {} }, ct::shape<n, n> {} };
    auto w = ct::partition_view { ct::tensor_span { w_data, ct::shape<n> {} }, ct::shape<n> {} };
    auto [factor, cholesky_info] = cusolverdx::tiles::cholesky_factor(b.load(0, 0), cusolverdx::tiles::lower);
    auto reduced = cusolverdx::tiles::generalized_to_standard_eig_hermitian(
        a.load(0, 0), factor, cusolverdx::tiles::eig_type<1>, cusolverdx::tiles::lower);
    auto [values, eigh_info] =
        cusolverdx::tiles::eig_hermitian(reduced, cusolverdx::tiles::lower, cusolverdx::tiles::no_vectors);
    w.store(values, 0);
    *cholesky_info_data = static_cast<int>(cholesky_info);
    *eigh_info_data = static_cast<int>(eigh_info);
}

int main() {
    // Dense symmetric A, and a dense diagonally dominant (so positive-definite) B.
    std::vector<float> a(n * n), b(n * n), w(n), reduced_w(n);
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < n; ++col) {
            const float distance = static_cast<float>(row > col ? row - col : col - row);
            a[row * n + col] = row == col ? static_cast<float>(row + 2) : 1.0f / (1.0f + distance);
            b[row * n + col] = row == col ? 4.0f : 0.5f / (1.0f + distance);
        }
    }

    device_buffer<float> d_a(a.size());
    device_buffer<float> d_b(b.size());
    device_buffer<float> d_w(w.size());
    device_buffer<float> d_reduced_w(reduced_w.size());
    device_buffer<int> d_info(1);
    device_buffer<int> d_cholesky_info(1);
    device_buffer<int> d_eigh_info(1);
    d_a.copy_from_host(a.data(), a.size());
    d_b.copy_from_host(b.data(), b.size());
    generalized_eigh_kernel<<<1, 1>>>(d_a.data(), d_b.data(), d_w.data(), d_info.data());
    CUDA_CHECK(cudaGetLastError());
    generalized_to_standard_eigh_kernel<<<1, 1>>>(
        d_a.data(), d_b.data(), d_reduced_w.data(), d_cholesky_info.data(), d_eigh_info.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int info = -1, cholesky_info = -1, eigh_info = -1;
    d_info.copy_to_host(&info, 1);
    d_cholesky_info.copy_to_host(&cholesky_info, 1);
    d_eigh_info.copy_to_host(&eigh_info, 1);
    d_w.copy_to_host(w.data(), w.size());
    d_reduced_w.copy_to_host(reduced_w.data(), reduced_w.size());

    // Both paths solve the same problem, so they must agree on the eigenvalues.
    const double agreement = relative_l2_error(reduced_w, w);
    std::printf(
        "HEGV: info=%d, HEGST: info=%d/%d, eigenvalue agreement=%g\n", info, cholesky_info, eigh_info, agreement);
    return info == 0 && cholesky_info == 0 && eigh_info == 0 && agreement < tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
