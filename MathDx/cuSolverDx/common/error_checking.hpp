/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUSOLVERDX_EXAMPLE_COMMON_ERROR_CHECKING_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_ERROR_CHECKING_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include <cusolverdx.hpp>

#include "numeric.hpp"

namespace common {

    struct HostCompactQrChecks {
        double qr_residual;
        double orthogonality;
    };

    template<typename ResultType, typename ReferenceType>
    double check_error(const ResultType* data, const ReferenceType* reference, const std::size_t n, bool print = false, bool verbose = false);

    // Error check for QR factorization: ||Q*R - A|| / ||A|| and ||Q^T*Q - I||
    template<cusolverdx::arrangement Arrange, class T>
    HostCompactQrChecks host_compact_qr_checks(const std::vector<T>& compact_qr,
                                               const std::vector<T>& tau,
                                               const std::vector<T>& a_orig,
                                               unsigned              batches,
                                               unsigned              m,
                                               unsigned              n,
                                               unsigned              lda) {
        const unsigned one_batch_elems = Arrange == cusolverdx::col_major ? lda * n : m * lda;

        auto matrix_offset = [&](unsigned row, unsigned col, unsigned ld) {
            if constexpr (Arrange == cusolverdx::col_major) {
                return row + col * ld;
            } else {
                return row * ld + col;
            }
        };

        std::vector<T> q_exp(m * n, convert<T>(0.0));
        std::vector<T> compact_batch(m * n, convert<T>(0.0));
        std::vector<T> a_orig_batch(m * n, convert<T>(0.0));
        std::vector<T> identity(n * n, convert<T>(0.0));
        for (unsigned i = 0; i < n; ++i) {
            identity[i + i * n] = convert<T>(1.0);
        }

        double worst_qr_residual  = 0.0;
        double worst_orthogonality = 0.0;

        for (unsigned batch = 0; batch < batches; ++batch) {
            const T* compact_batch_src = compact_qr.data() + static_cast<unsigned long long>(batch) * one_batch_elems;
            const T* a_orig_batch_src  = a_orig.data() + static_cast<unsigned long long>(batch) * one_batch_elems;
            const T* tau_batch_src     = tau.data() + static_cast<unsigned long long>(batch) * n;

            for (unsigned col = 0; col < n; ++col) {
                for (unsigned row = 0; row < m; ++row) {
                    compact_batch[row + col * m] = compact_batch_src[matrix_offset(row, col, lda)];
                    a_orig_batch[row + col * m]  = a_orig_batch_src[matrix_offset(row, col, lda)];
                }
            }

            std::fill(q_exp.begin(), q_exp.end(), convert<T>(0.0));
            for (unsigned j = 0; j < n; ++j) {
                q_exp[j + j * m] = convert<T>(1.0);
            }

            for (int j = n - 1; j >= 0; --j) {
                const unsigned uj    = static_cast<unsigned>(j);
                const T        tau_j = tau_batch_src[uj];
                if (tau_j == convert<T>(0.0)) {
                    continue;
                }
                for (unsigned col = uj; col < n; ++col) {
                    T dot = q_exp[uj + col * m];
                    for (unsigned row = uj + 1; row < m; ++row) {
                        dot += conj(compact_batch[row + uj * m]) * q_exp[row + col * m];
                    }
                    q_exp[uj + col * m] -= tau_j * dot;
                    for (unsigned row = uj + 1; row < m; ++row) {
                        q_exp[row + col * m] -= tau_j * dot * compact_batch[row + uj * m];
                    }
                }
            }

            std::vector<T> r(n * n, convert<T>(0.0));
            for (unsigned col = 0; col < n; ++col) {
                for (unsigned row = 0; row <= col; ++row) {
                    r[row + col * n] = compact_batch[row + col * m];
                }
            }

            std::vector<T> qr(m * n, convert<T>(0.0));
            for (unsigned col = 0; col < n; ++col) {
                for (unsigned k = 0; k < n; ++k) {
                    const T r_kc = r[k + col * n];
                    for (unsigned row = 0; row < m; ++row) {
                        qr[row + col * m] += q_exp[row + k * m] * r_kc;
                    }
                }
            }
            const double qr_residual = check_error<T, T>(qr.data(), a_orig_batch.data(), m * n);
            if (!std::isfinite(qr_residual)) {
                return {qr_residual, qr_residual};
            }
            worst_qr_residual = std::max(worst_qr_residual, qr_residual);

            std::vector<T> qtq(n * n, convert<T>(0.0));
            for (unsigned col = 0; col < n; ++col) {
                for (unsigned row = 0; row <= col; ++row) {
                    T dot = convert<T>(0.0);
                    for (unsigned k = 0; k < m; ++k) {
                        dot += conj(q_exp[k + row * m]) * q_exp[k + col * m];
                    }
                    qtq[row + col * n] = dot;
                    qtq[col + row * n] = dot;
                }
            }
            const double orthogonality = check_error<T, T>(qtq.data(), identity.data(), n * n);
            if (!std::isfinite(orthogonality)) {
                return {qr_residual, orthogonality};
            }
            worst_orthogonality = std::max(worst_orthogonality, orthogonality);
        }

        return {worst_qr_residual, worst_orthogonality};
    }

    template<typename T>
    bool is_error_acceptable(double tot_rel_err) {
        constexpr bool is_non_float_non_double_a_b_c =
            (!std::is_same_v<T, float> && !std::is_same_v<T, double>) || (!std::is_same_v<T, cusolverdx::complex<float>> && !std::is_same_v<T, cusolverdx::complex<double>>);

        if (is_non_float_non_double_a_b_c) {
            if (tot_rel_err > 1e-2) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        } else { // A,B,C are either float or double
            if (tot_rel_err > 1e-3) {
                std::cout << tot_rel_err << std::endl;
                return false;
            }
        }
        return std::isfinite(tot_rel_err);
    }

    template<class T>
    void make_diagonal_real(std::vector<T>& matrix, unsigned lda, unsigned m, unsigned batches, bool is_col_major) {
        if constexpr (common::is_complex<T>()) {
            const unsigned row_stride = is_col_major ? 1 : lda;
            const unsigned col_stride = is_col_major ? lda : 1;
            for (unsigned batch = 0; batch < batches; ++batch) {
                for (unsigned row = 0; row < m; ++row) {
                    const auto idx = batch * lda * m + row * row_stride + row * col_stride;
                    matrix[idx]    = T{matrix[idx].real(), 0};
                }
            }
        }
    }

    template<cusolverdx::fill_mode FillMode, cusolverdx::arrangement Arrange, class T>
    std::vector<T> extract_active_triangle(const std::vector<T>& matrix, unsigned m, unsigned batches) {
        std::vector<T> active(matrix.size(), common::convert<T>(0.0));
        auto index = [&](unsigned row, unsigned col) {
            if constexpr (Arrange == cusolverdx::arrangement::col_major) {
                return row + col * m;
            } else {
                return row * m + col;
            }
        };

        for (unsigned batch = 0; batch < batches; ++batch) {
            const T* in_batch  = matrix.data() + batch * m * m;
            T*       out_batch = active.data() + batch * m * m;
            for (unsigned row = 0; row < m; ++row) {
                for (unsigned col = 0; col < m; ++col) {
                    const bool keep = (FillMode == cusolverdx::fill_mode::lower) ? (row >= col) : (row <= col);
                    if (keep) {
                        out_batch[index(row, col)] = in_batch[index(row, col)];
                    }
                }
            }
        }

        return active;
    }
} // namespace common


#endif // CUSOLVERDX_TEST_COMMON_ERROR_CHECKING_HPP
