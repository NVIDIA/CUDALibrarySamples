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


#pragma once

#include <complex>
#include <type_traits>


inline float conjugate(float x) {
    return x;
}

inline double conjugate(double x) {
    return x;
}

inline std::complex<float> conjugate(const std::complex<float> x) {
    return std::conj(x);
}

inline std::complex<double> conjugate(const std::complex<double> x) {
    return std::conj(x);
}


inline float real(float x) {
    return x;
}

inline double real(double x) {
    return x;
}

inline float real(const std::complex<float>& x) {
    return x.real();
}

inline double real(const std::complex<double>& x) {
    return x.real();
}


inline float imag(float x) {
    return 0.0;
}

inline double imag(double x) {
    return 0.0;
}

inline float imag(const std::complex<float>& x) {
    return x.imag();
}

inline double imag(const std::complex<double>& x) {
    return x.imag();
}


template<typename T>
struct remove_complex {
    using type = T;
};

template<typename T>
struct remove_complex<std::complex<T>> {
    using type = T;
};

template<typename T>
using remove_complex_t = typename remove_complex<T>::type;
