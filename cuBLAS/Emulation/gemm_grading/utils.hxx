/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
