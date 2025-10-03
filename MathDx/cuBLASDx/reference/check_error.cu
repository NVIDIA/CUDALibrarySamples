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

#include "check_error.hpp"

#include <cmath>
#include <iostream>

namespace example {
    // Copied from CuTe examples and slightly modified
    template<typename T>
    double calculate_error_impl(const std::vector<T>& data, const std::vector<T>& reference, bool verbose, bool print) {
        using std::abs;
        using std::sqrt;

        // Use either double or complex<double> for error computation
        using value_type = cute::remove_cvref_t<decltype(reference[0])>;
        using error_type = std::conditional_t<is_complex<value_type>(), cublasdx::complex<double>, double>;

        if (print && verbose) {
            printf("Idx:\tVal\tRefVal\tRelError\n");
        }

        double eps = 1e-200;

        double tot_error_sq    = 0;
        double tot_norm_sq     = 0;
        double tot_ind_rel_err = 0;
        double max_ind_rel_err = 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            error_type val = convert<error_type>(data[i]);
            error_type ref = convert<error_type>(reference[i]);

            double aref      = detail::cbabs(ref);
            double diff      = detail::cbabs(ref - error_type(val));
            double rel_error = diff / (aref + eps);

            // Individual relative error
            tot_ind_rel_err += rel_error;

            // Maximum relative error
            max_ind_rel_err = std::max(max_ind_rel_err, rel_error);

            // Total relative error
            tot_error_sq += diff * diff;
            tot_norm_sq += aref * aref;

            if (print && verbose) {
                if constexpr (is_complex<error_type>()) {
                    std::cout << i << ":\t" << '<' << val.real() << ',' << val.imag() << '>' << "\t" << '<'
                              << ref.real() << ',' << ref.imag() << '>' << "\t" << rel_error << "\n";
                } else {
                    std::cout << i << ":\t" << val << "\t" << ref << "\t" << rel_error << "\n";
                }
            }
        }
        if (print)
            printf("Vector reference  norm: [%.5e]\n", sqrt(tot_norm_sq));

        double tot_rel_err = sqrt(tot_error_sq / (tot_norm_sq + eps));
        if (print)
            printf("Vector  relative error: [%.5e]\n", tot_rel_err);

        double ave_rel_err = tot_ind_rel_err / double(data.size());
        if (print)
            printf("Average relative error: [%.5e]\n", ave_rel_err);

        if (print)
            printf("Maximum relative error: [%.5e]\n", max_ind_rel_err);

        return tot_rel_err;
    }

    #define CHECK_ERROR_FOR_PRECISION(Prec)                                  \
    template<>                                                               \
    double calculate_error<Prec, Prec>(const std::vector<Prec>& data,        \
                                       const std::vector<Prec>& reference,   \
                                       bool verbose, bool print) {           \
        return calculate_error_impl(data, reference, verbose, print);        \
    }                                                                        \
    template<>                                                               \
    double calculate_error<cublasdx::complex<Prec>, cublasdx::complex<Prec>> \
        (const std::vector<cublasdx::complex<Prec>>& data,                   \
         const std::vector<cublasdx::complex<Prec>>& reference,              \
         bool verbose, bool print) {                                         \
        return calculate_error_impl(data, reference, verbose, print);        \
    }

    CHECK_ERROR_FOR_PRECISION(double);
    CHECK_ERROR_FOR_PRECISION(int64_t);
    CHECK_ERROR_FOR_PRECISION(uint64_t);
} // namespace test