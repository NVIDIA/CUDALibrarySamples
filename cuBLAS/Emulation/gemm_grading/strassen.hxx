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


#ifndef STRASSEN_HXX
#define STRASSEN_HXX

#include <cstring>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <limits>

// Regular matrix multiplication for base case (column-major).
template<typename T>
void gemm_naive(T* C, const T* A, const T* B, int m, int n, int k, int ldA, int ldB, int ldC) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            T sum = T(0);
            for (int l = 0; l < k; l++) {
                sum += A[l * ldA + i] * B[j * ldB + l];
            }
            C[j * ldC + i] = sum;
        }
    }
}

// Matrix addition (column-major).
template<typename T>
void matrix_add(T* C, const T* A, const T* B, int m, int n, int ldA, int ldB, int ldC) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C[j * ldC + i] = A[j * ldA + i] + B[j * ldB + i];
        }
    }
}

// Matrix subtraction (column-major).
template<typename T>
void matrix_sub(T* C, const T* A, const T* B, int m, int n, int ldA, int ldB, int ldC) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            C[j * ldC + i] = A[j * ldA + i] - B[j * ldB + i];
        }
    }
}

// Strassen's algorithm implementation (column-major).
template<typename T>
void gemm_strassen(T* C, const T* A, const T* B, int m, int n, int k, int ldA, int ldB, int ldC) {
    if (m <= 16 || n <= 16 || k <= 16) {
        gemm_naive(C, A, B, m, n, k, ldA, ldB, ldC);
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
    matrix_add(tempA, A11, A22, m2, k2, ldA, ldA, m2);
    matrix_add(tempB, B11, B22, k2, n2, ldB, ldB, k2);
    gemm_strassen(P1, tempA, tempB, m2, n2, k2, m2, k2, m2);

    // P2 = (A21 + A22) * B11
    matrix_add(tempA, A21, A22, m2, k2, ldA, ldA, m2);
    gemm_strassen(P2, tempA, B11, m2, n2, k2, m2, ldB, m2);

    // P3 = A11 * (B12 - B22)
    matrix_sub(tempB, B12, B22, k2, n2, ldB, ldB, k2);
    gemm_strassen(P3, A11, tempB, m2, n2, k2, ldA, k2, m2);

    // P4 = A22 * (B21 - B11)
    matrix_sub(tempB, B21, B11, k2, n2, ldB, ldB, k2);
    gemm_strassen(P4, A22, tempB, m2, n2, k2, ldA, k2, m2);

    // P5 = (A11 + A12) * B22
    matrix_add(tempA, A11, A12, m2, k2, ldA, ldA, m2);
    gemm_strassen(P5, tempA, B22, m2, n2, k2, m2, ldB, m2);

    // P6 = (A21 - A11) * (B11 + B12)
    matrix_sub(tempA, A21, A11, m2, k2, ldA, ldA, m2);
    matrix_add(tempB, B11, B12, k2, n2, ldB, ldB, k2);
    gemm_strassen(P6, tempA, tempB, m2, n2, k2, m2, k2, m2);

    // P7 = (A12 - A22) * (B21 + B22)
    matrix_sub(tempA, A12, A22, m2, k2, ldA, ldA, m2);
    matrix_add(tempB, B21, B22, k2, n2, ldB, ldB, k2);
    gemm_strassen(P7, tempA, tempB, m2, n2, k2, m2, k2, m2);

    // C11 = P1 + P4 - P5 + P7
    matrix_add(C11, P1, P4, m2, n2, m2, m2, ldC);
    matrix_sub(C11, C11, P5, m2, n2, ldC, m2, ldC);
    matrix_add(C11, C11, P7, m2, n2, ldC, m2, ldC);

    // C12 = P3 + P5
    matrix_add(C12, P3, P5, m2, n2, m2, m2, ldC);

    // C21 = P2 + P4
    matrix_add(C21, P2, P4, m2, n2, m2, m2, ldC);

    // C22 = P1 + P3 - P2 + P6
    matrix_add(C22, P1, P3, m2, n2, m2, m2, ldC);
    matrix_sub(C22, C22, P2, m2, n2, ldC, m2, ldC);
    matrix_add(C22, C22, P6, m2, n2, ldC, m2, ldC);

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

// Check Strassen's implementation against naive implementation.
template<typename T>
void check_strassen() {
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
        gemm_strassen(C_strassen, A, B, size, size, size, size, size, size);
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
                  << " | Abs Residual: " << std::scientific << std::setprecision(2) << std::setw(10) << abs_residual
                  << " | Rel Residual: " << std::scientific << std::setprecision(2) << std::setw(10) << rel_residual
                  << std::endl;

        // Clean up.
        delete[] A;
        delete[] B;
        delete[] C_strassen;
        delete[] C_naive;
    }
}

#endif // STRASSEN_HXX 
