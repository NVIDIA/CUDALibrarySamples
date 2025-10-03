/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cmath>
#include <iostream>

#include <type_traits>

#include <cusolverdx.hpp>

#include "numeric.hpp"

namespace common {

    template<typename ResultType, typename ReferenceType>
    double check_error(const ResultType* data, const ReferenceType* reference, const std::size_t n, bool print = false, bool verbose = false);

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
} // namespace common


#endif // CUSOLVERDX_TEST_COMMON_ERROR_CHECKING_HPP