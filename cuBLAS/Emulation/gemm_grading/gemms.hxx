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
#include <iostream>
#include <iomanip>

#include "strassen.hxx"
#include "fix_strassen.hxx"
#include "cublas_gemm.hxx"

// Enum to identify different GEMM algorithms.
enum class GemmAlgorithm {
    NATIVE_STRASSEN,    // Native floating-point Strassen.
    REF_GEMM,           // Standard GEMM provided by linked BLAS library.
    FIXED_STRASSEN,     // Fixed-point Strassen.
    CUDA_NATIVE_GEMM,   // Native CUDA GEMM.
    CUDA_EMULATED_GEMM  // Emulated CUDA GEMM based on Ozaki's scheme.
};



// Generic interface to various realizations of matrix-matrix multiplication.
template<typename T>
void gemm(int64_t n, const T *A, int64_t lda, const T *B, int64_t ldb,
          T *C, int64_t ldc, GemmAlgorithm algo);

