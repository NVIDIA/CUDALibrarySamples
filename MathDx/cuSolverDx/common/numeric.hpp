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

#ifndef CUSOLVERDX_EXAMPLE_COMMON_NUMERIC_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_NUMERIC_HPP

#include <vector>
#include <algorithm>
#include <complex>
#include <random>
#include <type_traits>

#ifndef CUSOLVERDX_EXAMPLE_NVRTC
#    include <cuda/std/complex>
#    include <cusolverdx.hpp>
#endif // CUSOLVERDX_EXAMPLE_NVRTC

namespace common {

    namespace detail {

        template<class T>
        struct is_complex_helper {
            static constexpr bool value = false;
        };
#ifndef CUSOLVERDX_EXAMPLE_NVRTC
        template<class T>
        struct is_complex_helper<cusolverdx::complex<T>> {
            static constexpr bool value = true;
        };

        template<class T>
        struct is_complex_helper<cuda::std::complex<T>> {
            static constexpr bool value = true;
        };
#endif

        template<class T>
        struct is_complex_helper<std::complex<T>> {
            static constexpr bool value = true;
        };

        template<>
        struct is_complex_helper<cuFloatComplex> {
            static constexpr bool value = true;
        };
        template<>
        struct is_complex_helper<cuDoubleComplex> {
            static constexpr bool value = true;
        };


    } // namespace detail

    template<typename T>
    constexpr bool is_complex() {
        return detail::is_complex_helper<std::remove_cv_t<T>>::value;
    }

    template<typename T, typename Enable = void>
    struct get_precision;

    template<>
    struct get_precision<cuFloatComplex> {
        using type = float;
    };
    template<>
    struct get_precision<cuDoubleComplex> {
        using type = double;
    };

    template<typename T>
    struct get_precision<T, std::enable_if_t<is_complex<T>()>> {
        using type = typename T::value_type;
    };

    template<typename T>
    struct get_precision<T, std::enable_if_t<!is_complex<T>()>> {
        using type = T;
    };

    template<typename T>
    using get_precision_t = typename get_precision<T>::type;


    template<typename T1, typename T2>
    constexpr T1 convert(T2 v) {
        T1 result{}; // add bracket to remove older gcc's warning of "constructor is not user-provided"
        if constexpr (is_complex<T1>() && is_complex<T2>()) {
            result = T1(v);
        } else if constexpr (is_complex<T1>()) {
            result = T1(v, v);
        } else if constexpr (is_complex<T2>()) {
            result = v.real();
        } else {
            result = T1(v);
        }
        return result;
    }

    namespace detail {
        template<typename T>
        double abs(T v) {
            double result;
            if constexpr (is_complex<T>()) {
                auto imag = std::abs(static_cast<double>(v.imag()));
                auto real = std::abs(static_cast<double>(v.real()));
                result    = (real + imag) / 2.0;
            } else {
                result = std::abs(static_cast<double>(v));
            }
            return result;
        }
    } // namespace detail


    template<typename Tin, typename Tout>
    std::vector<Tout> convert(const std::vector<Tin>& input) {
        std::vector<Tout> output(input.size());
        std::transform(
            input.begin(), input.end(), output.begin(), [](const Tin& val) { return static_cast<Tout>(val); });
        return output;
    }
#ifndef CUSOLVERDX_EXAMPLE_NVRTC
    template<typename Prec>
    std::ostream& operator<<(std::ostream& os, cusolverdx::complex<Prec>& v) {
        return os << "(" << v.real() << ", " << v.imag() << "), ";
    }
#endif

} // namespace common


#endif // CUSOLVERDX_EXAMPLE_COMMON_NUMERIC_HPP