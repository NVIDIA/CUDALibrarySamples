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


#include <cstring>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include "strassen.hxx"  // Include original strassen.hxx for dgemm_naive

/*
 * Fixed-Point Strassen Algorithm (32.32 format)
 * ---------------------------------------------
 *
 * This file implements a fixed-point version of Strassen's matrix multiplication algorithm.
 * The main motivation is to perform matrix multiplication using integer arithmetic, which can be
 * beneficial for hardware without fast floating-point support, or to study numerical properties
 * of fixed-point arithmetic in fast algorithms.
 *
 * Key Features and Steps:
 * ----------------------
 * 1. **Fixed-Point Representation:**
 *    - Uses a 64-bit signed integer (int64_t) to represent numbers in 32.32 fixed-point format:
 *      - The upper 32 bits are the integer part, the lower 32 bits are the fractional part.
 *    - Conversion functions are provided to switch between double and fixed-point.
 *
 * 2. **Scaling:**
 *    - To avoid overflow during computation, input matrices are scaled down by a factor
 *      determined by the maximum absolute values in the input matrices and the matrix size.
 *    - After computation, the result is scaled back to the original range.
 *
 * 3. **Arithmetic Operations:**
 *    - Addition and subtraction are performed directly on the fixed-point integers.
 *    - Multiplication uses 128-bit intermediate results to prevent overflow before shifting
 *      back to the 32.32 format.
 *
 * 4. **Algorithm Structure:**
 *    - The recursive structure of Strassen's algorithm is preserved, but all arithmetic is
 *      performed in fixed-point.
 *    - For small matrices (<=16), a naive fixed-point matrix multiplication is used as the base case.
 *    - Temporary matrices for intermediate results are allocated as needed.
 *
 * 5. **Comparison and Testing:**
 *    - The function `check_fixed_strassen()` compares the fixed-point Strassen result to the naive
 *      floating-point result for a range of matrix sizes, reporting absolute and relative errors.
 *
 * Differences from Floating-Point Strassen:
 * ----------------------------------------
 * - All arithmetic is performed in fixed-point, not floating-point.
 * - Scaling is required to avoid overflow, which is not needed in floating-point.
 * - Conversion between double and fixed-point is necessary at the input and output stages.
 * - The algorithm is otherwise structurally identical to the floating-point version.
 *
 * Limitations:
 * -----------
 * - Fixed-point arithmetic can still overflow for very large matrices or poorly scaled inputs.
 * - Precision is limited by the number of fractional bits (32 in this case).
 * - The implementation is for educational and experimental purposes; for production, further
 *   optimizations and error handling may be needed.
 */

// Fixed-point type using int64_t for 32.32 format.
using fixed_point_t = int64_t;
constexpr int FRACTIONAL_BITS = 32;
constexpr fixed_point_t ONE = 1LL << FRACTIONAL_BITS;

// Complex fixed-point type.
struct complex_fixed_point_t {
    fixed_point_t real;
    fixed_point_t imag;

    complex_fixed_point_t() : real(0), imag(0) {}
    complex_fixed_point_t(const fixed_point_t& r, const fixed_point_t& i) : real(r), imag(i) {}
    complex_fixed_point_t(int val) : real(val), imag(0) {}

    complex_fixed_point_t& operator+=(const complex_fixed_point_t& other) {
        real += other.real;
        imag += other.imag;
        return *this;
    }

    complex_fixed_point_t operator+(const complex_fixed_point_t& other) const {
        return complex_fixed_point_t(real + other.real, imag + other.imag);
    }

    complex_fixed_point_t operator-(const complex_fixed_point_t& other) const {
        return complex_fixed_point_t(real - other.real, imag - other.imag);
    }
};

// Forward declarations.
template<typename T>
void gemm_strassen_fixed_recursive(T* C, const T* A, const T* B,
                                 int m, int n, int k, int ldA, int ldB, int ldC);

// Convert double to fixed-point.
inline fixed_point_t to_fixed(double x) {
    return static_cast<fixed_point_t>(x * ONE);
}

// Convert fixed-point to double.
inline double to_double(fixed_point_t x) {
    return static_cast<double>(x) / ONE;
}

// Convert complex<double> to complex fixed-point.
inline complex_fixed_point_t to_fixed(std::complex<double> x) {
    return {to_fixed(x.real()), to_fixed(x.imag())};
}

// Convert complex fixed-point to complex<double>.
inline std::complex<double> to_double(complex_fixed_point_t x) {
    return std::complex<double>(to_double(x.real), to_double(x.imag));
}

// Fixed-point multiplication with scaling to avoid overflow.
inline fixed_point_t fixed_mul(fixed_point_t a, fixed_point_t b) {
    // Use int128_t for intermediate result to avoid overflow.
    __int128_t temp = static_cast<__int128_t>(a) * static_cast<__int128_t>(b);
    return static_cast<fixed_point_t>(temp >> FRACTIONAL_BITS);
}

// Complex fixed-point multiplication.
inline complex_fixed_point_t fixed_mul(complex_fixed_point_t a, complex_fixed_point_t b) {
    fixed_point_t real = fixed_mul(a.real, b.real) - fixed_mul(a.imag, b.imag);
    fixed_point_t imag = fixed_mul(a.real, b.imag) + fixed_mul(a.imag, b.real);
    return {real, imag};
}

// Regular matrix multiplication for base case (column-major) using fixed-point.
template<typename T>
void gemm_naive_fixed(T* C, const T* A, const T* B, 
                     int m, int n, int k, int ldA, int ldB, int ldC) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            T sum = T(0);
            for (int l = 0; l < k; l++) {
                sum += fixed_mul(A[l * ldA + i], B[j * ldB + l]);
            }
            C[j * ldC + i] = sum;
        }
    }
}

// Matrix addition (column-major) using fixed-point.
template<typename T>
void matrix_add_fixed(T* C, const T* A, const T* B, 
                     int m, int n, int ldA, int ldB, int ldC) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C[j * ldC + i] = A[j * ldA + i] + B[j * ldB + i];
        }
    }
}

// Matrix subtraction (column-major) using fixed-point.
template<typename T>
void matrix_sub_fixed(T* C, const T* A, const T* B, 
                     int m, int n, int ldA, int ldB, int ldC) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C[j * ldC + i] = A[j * ldA + i] - B[j * ldB + i];
        }
    }
}

// Find scaling factor to avoid overflow.
template<typename T>
double find_scaling_factor(const T* A, const T* B, int m, int n, int k, int ldA, int ldB) {
    double max_abs_A = 0.0;
    double max_abs_B = 0.0;
    
    // Find maximum absolute values in A and B.
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            if constexpr (std::is_same<T, std::complex<double>>::value) {
                max_abs_A = std::max(max_abs_A, std::abs(A[j * ldA + i]));
            } else {
                max_abs_A = std::max(max_abs_A, std::abs(A[j * ldA + i]));
            }
        }
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            if constexpr (std::is_same<T, std::complex<double>>::value) {
                max_abs_B = std::max(max_abs_B, std::abs(B[j * ldB + i]));
            } else {
                max_abs_B = std::max(max_abs_B, std::abs(B[j * ldB + i]));
            }
        }
    }
    
    // Calculate scaling factor to avoid overflow.
    double scale = 1.0;
    if (max_abs_A > 0.0 && max_abs_B > 0.0) {
        double max_product = max_abs_A * max_abs_B * k;
        if (max_product > 1.0) {
            scale = 1.0 / std::sqrt(max_product);
        }
    }
    return scale;
}

// Strassen's algorithm implementation using fixed-point arithmetic.
template<typename T>
void gemm_strassen_fixed(T* C, const T* A, const T* B, 
                        int m, int n, int k, int ldA, int ldB, int ldC) {
    // Find scaling factor to avoid overflow.
    double scale = find_scaling_factor(A, B, m, n, k, ldA, ldB);
    
    // Convert input matrices to fixed-point with scaling.
    using fixed_t = typename std::conditional<std::is_same<T, std::complex<double>>::value,
                                            complex_fixed_point_t, fixed_point_t>::type;
    fixed_t* A_fixed = new fixed_t[m * k];
    fixed_t* B_fixed = new fixed_t[k * n];
    fixed_t* C_fixed = new fixed_t[m * n];
    
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < m; i++) {
            if constexpr (std::is_same<T, std::complex<double>>::value) {
                A_fixed[j * m + i] = to_fixed(A[j * ldA + i] * scale);
            } else {
                A_fixed[j * m + i] = to_fixed(A[j * ldA + i] * scale);
            }
        }
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < k; i++) {
            if constexpr (std::is_same<T, std::complex<double>>::value) {
                B_fixed[j * k + i] = to_fixed(B[j * ldB + i] * scale);
            } else {
                B_fixed[j * k + i] = to_fixed(B[j * ldB + i] * scale);
            }
        }
    }
    
    // Call recursive fixed-point Strassen implementation.
    gemm_strassen_fixed_recursive(C_fixed, A_fixed, B_fixed, m, n, k, m, k, m);
    
    // Convert result back to double and apply inverse scaling.
    double inv_scale = 1.0 / (scale * scale);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if constexpr (std::is_same<T, std::complex<double>>::value) {
                C[j * ldC + i] = to_double(C_fixed[j * m + i]) * inv_scale;
            } else {
                C[j * ldC + i] = to_double(C_fixed[j * m + i]) * inv_scale;
            }
        }
    }
    
    delete[] A_fixed;
    delete[] B_fixed;
    delete[] C_fixed;
}

// Recursive part of Strassen's algorithm using fixed-point.
template<typename T>
void gemm_strassen_fixed_recursive(T* C, const T* A, const T* B,
                                 int m, int n, int k, int ldA, int ldB, int ldC) {
    if (m <= 16 || n <= 16 || k <= 16) {
        gemm_naive_fixed(C, A, B, m, n, k, ldA, ldB, ldC);
        return;
    }

    int m2 = m / 2;
    int n2 = n / 2;
    int k2 = k / 2;

    T* P1 = new T[m2 * n2];
    T* P2 = new T[m2 * n2];
    T* P3 = new T[m2 * n2];
    T* P4 = new T[m2 * n2];
    T* P5 = new T[m2 * n2];
    T* P6 = new T[m2 * n2];
    T* P7 = new T[m2 * n2];

    T* tempA = new T[m2 * k2];
    T* tempB = new T[k2 * n2];

    // Submatrix pointers (column-major).
    const T* A11 = A;
    const T* A12 = A + k2 * ldA;
    const T* A21 = A + m2;
    const T* A22 = A + m2 + k2 * ldA;

    const T* B11 = B;
    const T* B12 = B + n2 * ldB;
    const T* B21 = B + k2;
    const T* B22 = B + k2 + n2 * ldB;

    T* C11 = C;
    T* C12 = C + n2 * ldC;
    T* C21 = C + m2;
    T* C22 = C + m2 + n2 * ldC;

    // P1 = (A11 + A22) * (B11 + B22)
    matrix_add_fixed(tempA, A11, A22, m2, k2, ldA, ldA, m2);
    matrix_add_fixed(tempB, B11, B22, k2, n2, ldB, ldB, k2);
    gemm_strassen_fixed_recursive(P1, tempA, tempB, m2, n2, k2, m2, k2, m2);

    // P2 = (A21 + A22) * B11
    matrix_add_fixed(tempA, A21, A22, m2, k2, ldA, ldA, m2);
    gemm_strassen_fixed_recursive(P2, tempA, B11, m2, n2, k2, m2, ldB, m2);

    // P3 = A11 * (B12 - B22)
    matrix_sub_fixed(tempB, B12, B22, k2, n2, ldB, ldB, k2);
    gemm_strassen_fixed_recursive(P3, A11, tempB, m2, n2, k2, ldA, k2, m2);

    // P4 = A22 * (B21 - B11)
    matrix_sub_fixed(tempB, B21, B11, k2, n2, ldB, ldB, k2);
    gemm_strassen_fixed_recursive(P4, A22, tempB, m2, n2, k2, ldA, k2, m2);

    // P5 = (A11 + A12) * B22
    matrix_add_fixed(tempA, A11, A12, m2, k2, ldA, ldA, m2);
    gemm_strassen_fixed_recursive(P5, tempA, B22, m2, n2, k2, m2, ldB, m2);

    // P6 = (A21 - A11) * (B11 + B12)
    matrix_sub_fixed(tempA, A21, A11, m2, k2, ldA, ldA, m2);
    matrix_add_fixed(tempB, B11, B12, k2, n2, ldB, ldB, k2);
    gemm_strassen_fixed_recursive(P6, tempA, tempB, m2, n2, k2, m2, k2, m2);

    // P7 = (A12 - A22) * (B21 + B22)
    matrix_sub_fixed(tempA, A12, A22, m2, k2, ldA, ldA, m2);
    matrix_add_fixed(tempB, B21, B22, k2, n2, ldB, ldB, k2);
    gemm_strassen_fixed_recursive(P7, tempA, tempB, m2, n2, k2, m2, k2, m2);

    // C11 = P1 + P4 - P5 + P7
    matrix_add_fixed(C11, P1, P4, m2, n2, m2, m2, ldC);
    matrix_sub_fixed(C11, C11, P5, m2, n2, ldC, m2, ldC);
    matrix_add_fixed(C11, C11, P7, m2, n2, ldC, m2, ldC);

    // C12 = P3 + P5
    matrix_add_fixed(C12, P3, P5, m2, n2, m2, m2, ldC);

    // C21 = P2 + P4
    matrix_add_fixed(C21, P2, P4, m2, n2, m2, m2, ldC);

    // C22 = P1 + P3 - P2 + P6
    matrix_add_fixed(C22, P1, P3, m2, n2, m2, m2, ldC);
    matrix_sub_fixed(C22, C22, P2, m2, n2, ldC, m2, ldC);
    matrix_add_fixed(C22, C22, P6, m2, n2, ldC, m2, ldC);

    delete[] P1;
    delete[] P2;
    delete[] P3;
    delete[] P4;
    delete[] P5;
    delete[] P6;
    delete[] P7;
    delete[] tempA;
    delete[] tempB;
}

// Check fixed-point Strassen's implementation against naive implementation.
template<typename T>
void check_fixed_strassen() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<typename std::conditional<std::is_same<T, std::complex<double>>::value, double, T>::type> dist(-1.0, 1.0);

    // Test sizes: 2^1 to 2^9 (2 to 512)
    for (int size = 2; size <= 512; size *= 2) {
        // Allocate matrices.
        T* A = new T[size * size];
        T* B = new T[size * size];
        T* C_strassen = new T[size * size];
        T* C_naive = new T[size * size];

        // Fill matrices with random values.
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                if constexpr (std::is_same<T, std::complex<double>>::value) {
                    A[j * size + i] = T(dist(gen), dist(gen));
                    B[j * size + i] = T(dist(gen), dist(gen));
                } else {
                    A[j * size + i] = dist(gen);
                    B[j * size + i] = dist(gen);
                }
            }
        }

        // Initialize result matrices to zero.
        std::memset(C_strassen, 0, size * size * sizeof(T));
        std::memset(C_naive, 0, size * size * sizeof(T));

        // Run both implementations.
        gemm_strassen_fixed(C_strassen, A, B, size, size, size, size, size, size);
        gemm_naive(C_naive, A, B, size, size, size, size, size, size);

        // Calculate residuals.
        double abs_residual = 0.0;
        double rel_residual = 0.0;
        double norm_C = 0.0;

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                double diff;
                if constexpr (std::is_same<T, std::complex<double>>::value) {
                    diff = std::abs(C_strassen[j * size + i] - C_naive[j * size + i]);
                    norm_C = std::max(norm_C, std::abs(C_naive[j * size + i]));
                } else {
                    diff = std::abs(C_strassen[j * size + i] - C_naive[j * size + i]);
                    norm_C = std::max(norm_C, std::abs(C_naive[j * size + i]));
                }
                abs_residual = std::max(abs_residual, diff);
            }
        }
        rel_residual = abs_residual / (norm_C + 1e-16); // Add small epsilon to avoid division by zero

        // Print results.
        std::cout << "Size: " << std::setw(4) << size 
                  << " | Abs Residual: " << std::scientific << std::setprecision(2) << abs_residual
                  << " | Rel Residual: " << std::scientific << std::setprecision(2) << rel_residual
                  << std::endl;

        // Clean up.
        delete[] A;
        delete[] B;
        delete[] C_strassen;
        delete[] C_naive;
    }
}

