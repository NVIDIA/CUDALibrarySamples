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

#ifndef CUBLASDX_EXAMPLE_TRSM_COMMON_HPP
#define CUBLASDX_EXAMPLE_TRSM_COMMON_HPP

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../reference/reference_trsm.hpp"
#include "../reference/check_error.hpp"

namespace example {
    namespace detail {

        template<typename T>
        struct elementwise_abs {
            T operator()(T v) { return std::abs(v); }
        };

        template<typename Inner>
        struct elementwise_abs<cublasdx::complex<Inner>> {
            cublasdx::complex<Inner> operator()(cublasdx::complex<Inner> v) {
                return cublasdx::complex<Inner>(std::abs(v.real()), std::abs(v.imag()));
            }
        };

        template<typename T>
        struct repeat {
            T operator()(T v) { return v; }
        };

        template<typename Inner>
        struct repeat<cublasdx::complex<Inner>> {
            cublasdx::complex<Inner> operator()(Inner v) { return cublasdx::complex<Inner>(v, v); }
        };
    } // namespace detail


    template<class DT, cublasdx::arrangement Arrangement>
    auto make_trsm_tensor(DT* pointer, int m, int n, int batches) {
        auto stride =
            cute::conditional_return < Arrangement ==
            cublasdx::col_major > (cute::make_stride(cute::_1 {}, m, m * n), cute::make_stride(n, cute::_1 {}, m * n));
        return cute::make_tensor(pointer, cute::make_layout(cute::make_shape(m, n, batches), stride));
    }
    // Make each diagonal entry of A large enough to ensure diagonal dominance.
    template<class Tensor>
    void make_diagonal_dominant(Tensor&& tensor) {
        using T = typename cute::remove_cvref_t<Tensor>::value_type;
        if constexpr (decltype(cute::rank(tensor))::value == 3) {
            for (unsigned int batch = 0; batch < cute::size<2>(tensor); batch++) {
                make_diagonal_dominant(tensor(cublasdx::slice, cublasdx::slice, batch));
            }
        } else {
            static_assert(decltype(cute::rank(tensor))::value == 2, "Tensor must be a 2D tensor");
            for (unsigned int row = 0; row < cute::size<0>(tensor); row++) {
                T offdiag_sum = detail::repeat<T> {}(5.f);
                for (unsigned int col = 0; col < cute::size<1>(tensor); col++) {
                    if (col != row) {
                        T a_reg = tensor(row, col);
                        offdiag_sum += detail::elementwise_abs<T> {}(a_reg);
                    }
                }

                if (row < cute::size<1>(tensor)) {
                    tensor(row, row) = offdiag_sum;
                }
            }
        }
    }
} // namespace example

#endif // CUBLASDX_EXAMPLE_TRSM_COMMON_HPP
