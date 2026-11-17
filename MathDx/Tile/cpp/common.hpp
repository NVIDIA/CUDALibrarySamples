// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Solver-specific matrix generation and naming conventions. Helpers shared
// with the FFT examples live in device_buffer.hpp, error_checks.hpp, and
// error_metrics.hpp. Timing and FLOP counts live in common_perf_measure.hpp.
//
// Naming convention used throughout these examples:
//   - kernels are named <routine>_kernel;
//   - kernel parameters holding raw device pointers are suffixed _data;
//   - partition_view locals are lowercase and named after the matrix they view (a, b, x, ...);
//   - host-side device pointers are prefixed d_;
//   - the status returned by a routine is called info.

#ifndef CUSOLVERDX_TILE_EXAMPLES_COMMON_HPP
#define CUSOLVERDX_TILE_EXAMPLES_COMMON_HPP

#include <cstddef>
#include <vector>

#include "device_buffer.hpp"
#include "error_checks.hpp"
#include "error_metrics.hpp"

using mathdx_examples::device_buffer;
using mathdx_examples::relative_l2_error;

// Dense symmetric positive-definite n x n matrix, built as L L^T from a dense
// lower-triangular factor whose diagonal dominates its off-diagonal entries.
inline std::vector<float> make_spd_matrix(size_t n) {
    std::vector<float> l(n * n, 0.0f), a(n * n, 0.0f);
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col <= row; ++col) {
            l[row * n + col] = row == col ? 2.0f + static_cast<float>(row) / static_cast<float>(n)
                                          : static_cast<float>((row + 2 * col) % 5) / 8.0f - 0.25f;
        }
    }
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < n; ++col) {
            for (size_t k = 0; k < n; ++k) {
                a[row * n + col] += l[row * n + k] * l[col * n + k];
            }
        }
    }
    return a;
}

#endif // CUSOLVERDX_TILE_EXAMPLES_COMMON_HPP
