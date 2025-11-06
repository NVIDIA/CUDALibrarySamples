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


#include <algorithm>
#include <complex>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>

#include "utils.hxx"
#include "gemms.hxx"
#include <iomanip>
#include <cmath>

// ANSI color codes.
#define ANSI_COLOR_GREEN "\033[32m"
#define ANSI_COLOR_RED "\033[31m"
#define ANSI_COLOR_RESET "\033[0m"

// Debug print macro.
#ifdef DEBUG_PRINTS
    #define DEBUG_PRINT(x) std::cout << x
#else
    #define DEBUG_PRINT(x)
#endif

//
// References:
//
// [1] Jim Demmel, Xiaoye Li, Julien Langou, Weslley Pereira, Mark Gates, Cindy Rubio Gonzalez (2024),
//     "How to grade the accuracy of an implementation of the BLAS," slides available at:
//     https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Grading_BLAS.pdf
//
// [2] Jim Demmel et al. (2025), "More aggressive (sparse) BLAS testing, to identify aggressive optimizations."
//     Private communication. Unpublished manuscript, referenced with author approval.
//
// [3] Harun Bayraktar (2025): Precision Redefined: Unlocking and Delivering the
//     Full Power of Modern GPUs for Scientific Computing, PASC25
//     https://linklings.s3.amazonaws.com/organizations/pasc/pasc25/submissions/stype110/U83sS-msa270s2.pdf
//

// Acknowledgments:
// We gratefully acknowledge Prof. James W. Demmel for his generous support, for sharing
// the manuscript [2], and for his review and feedback on this work prior to publication.
//


// Select k unique indices from 0, 1, ..., n-1 at random.
std::vector<int64_t> selectIndices(int64_t n, int64_t k, std::mt19937 &gen) {
    std::vector<int64_t> indices(n);
    std::vector<int64_t> selectedIndices(k);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    std::copy(indices.begin(), indices.begin() + k, selectedIndices.begin());
    std::sort(selectedIndices.begin(), selectedIndices.end());
    return selectedIndices;
}

template<typename T>
void print_matrix(const std::vector<T> &A, int64_t n, int64_t lda, const std::string &label) {
#ifdef DEBUG_PRINTS
    std::cout << "\n" << label << "\n";
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            if constexpr (std::is_same_v<T, std::complex<double>>) {
                std::cout << A[i + j * lda].real() << "+" << A[i + j * lda].imag() << "i ";
            } else {
                std::cout << A[i + j * lda] << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
#endif
}



/*
 * Distinguish Strassen-like GEMM and O(n**3) GEMM.
 *
 * @param[in] n: dimension of the matrices. n >= 2.
 * @param[in] k: number of exact zero entries in (A*B). 1 <= k <= n-1.
 * @param[in] gen: random number generator
 * @param[in] verbose: print matrix generation information and A, B
 * @return true if the algorithm appears to be conventional O(n**3) GEMM.
 * 
 * This routine implements Test 2 described in [2] and corresponds
 * to Test 1b in [1].
 */
template<typename T>
bool test2(int64_t n, int64_t k, std::mt19937 &gen, bool verbose, GemmAlgorithm algo) {
    using Real = remove_complex_t<T>;

    // Input argument check.
    if (n < 2) {
        std::cerr << "[test2] n must be >= 2" << std::endl;
        return true;
    }

    // Clamp k to valid range.
    k = std::max(int64_t(1), std::min(k, n-1));

    // Allocate matrices A, B.
    int64_t lda = n;
    int64_t ldb = n;
    std::vector<T> A(lda * n);
    std::vector<T> B(ldb * n);

    // A = rand(n,n) + 1i*rand(n,n), B = rand(n,n) + 1i*rand(n,n).
    std::normal_distribution<Real> dist(0.0, 1.0); // mean = 0, stddev = 1
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) {
            if constexpr (std::is_same_v<T, std::complex<Real>>) {
                A[i + j * lda] = T(dist(gen), dist(gen));
                B[i + j * ldb] = T(dist(gen), dist(gen));
            } else {
                A[i + j * lda] = dist(gen);
                B[i + j * ldb] = dist(gen);
            }
        }
    }

    // Select a subset of 1 < k < n rows of A labeled (i_1, ...,i_k)
    // and k columns of B, labeled j_1, ..., j_k, where (AB)(i_k, j_k) = 0.
    std::vector<int64_t> selectedRowIndices = selectIndices(n, k, gen);
    std::vector<int64_t> selectedColIndices = selectIndices(n, k, gen);

    if (verbose) {
        std::cout << "\nSelected row indices: ";
        for (int64_t idx : selectedRowIndices) {
            std::cout << idx << " ";
        }
        std::cout << "\nSelected column indices: ";
        for (int64_t idx : selectedColIndices) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
    }

    // Fill A, B as complementary sparse matrices.
    //
    // For each m in [1:k], pick a random proper subset S_m of [1:n] and zero out
    // entries A(i_m, r) for all r in S_m and zero out B(s, j_m) for all s not in S_m.
    for (int64_t m = 0; m < k; m++) {
        int64_t i_m = selectedRowIndices[m];
        int64_t j_m = selectedColIndices[m];

        // Select a proper subset S_m of [1:n].
        std::vector<int64_t> allIndices(n);
        std::iota(allIndices.begin(), allIndices.end(), 0);
        std::shuffle(allIndices.begin(), allIndices.end(), gen);
        // The first splitPoint elements are the indices of A that will be zeroed out,
        // the remaining elements are the indices of B that will be zeroed out.
        int64_t splitPoint = std::uniform_int_distribution<int64_t>(1, n-1)(gen);
        std::vector<int64_t> zeroIndicesA(allIndices.begin(), allIndices.begin() + splitPoint);
        std::vector<int64_t> zeroIndicesB(allIndices.begin() + splitPoint, allIndices.end());
        std::sort(zeroIndicesA.begin(), zeroIndicesA.end());
        std::sort(zeroIndicesB.begin(), zeroIndicesB.end());

        // Zero out A(i_m, r) for all r in S_m and zero out B(s, j_m) for all s not in S_m.
        for (int64_t j : zeroIndicesA) {
            A[i_m + j * lda] = T(0);
        }
        for (int64_t i : zeroIndicesB) {
            B[i + j_m * ldb] = T(0);
        }
    }

    if (verbose) {
        print_matrix(A, n, lda, "Matrix A:");
        print_matrix(B, n, ldb, "Matrix B:");
    }

    int64_t ldc = n;
    std::vector<T> C(n * ldc);
    gemm(n, A.data(), lda, B.data(), ldb, C.data(), ldc, algo);

    // Check if C(i_m, j_m) = 0 for all m in [1:k].
    for (int64_t m = 0; m < k; m++) {
        int64_t i_m = selectedRowIndices[m];
        int64_t j_m = selectedColIndices[m];
        if (C[i_m + j_m * ldc] != 0.0) {
#ifdef DEBUG_PRINTS
            std::cout << "[test 2] Strassen-like GEMM found: C(" << i_m << ", " << j_m << ") = " 
                      << C[i_m + j_m * ldc] << " != 0.0" << std::endl;
#endif
            return false;
        }
    }
    return true;
}

/*
 * Distinguish fixed-point DGEMM and conventional O(n**3) DGEMM.
 *
 * @param[in] n: dimension of the matrices. n >= 1.
 * @param[in] minExponent: minimum exponent for scaling factors.
 *                         log2(sqrt(underflow)) + log2(n) + 2 <= minExponent.
 * @param[in] maxExponent: maximum exponent for scaling factors.
 *                         minExponent <= maxExponent <= log2(sqrt(overflow)) - log2(n) - 2.
 * @param[in] gen: random number generator
 * @param[in] verbose: print matrix generation information and generated matrices A, B
 * @return true if the algorithm appears to use conventional floating point.
 * 
 * This routine implements Test 4 described in [2] and corresponds
 * to Test 2b in [1].
 */
template<typename T>
bool test4(int64_t n, int minExponent, int maxExponent, std::mt19937 &gen, bool verbose, GemmAlgorithm algo) {
    using Real = remove_complex_t<T>;

    // Get machine constants.
    const Real overflow = std::numeric_limits<Real>::max();
    const Real safmax = sqrt(overflow);
    const Real safmin = 1.0/overflow;
    const Real eps = std::numeric_limits<Real>::epsilon() * 0.5; // LAPACK eps = half IEEE eps
    const Real tol = 10.0 * eps;

    // Input argument check.
    if (n < 1) {
        std::cerr << "[test4] n must be >= 1" << std::endl;
        return true;
    }
    int safmaxExponent;
    std::frexp(safmax, &safmaxExponent);
    if (maxExponent > safmaxExponent - std::log2(n) - 2) {
        std::cerr << "[test4] maxExponent must be <= log2(sqrt(overflow)) - log2(n) - 2"
                  << " to guarantee an overflow-free computation" << std::endl;
        return true;
    }
    if (minExponent < -safmaxExponent + std::log2(n) + 2) {
        std::cerr << "[test4] minExponent must be >= log2(sqrt(underflow)) + log2(n) + 2"
                  << " to guarantee an overflow-free computation" << std::endl;
        return true;
    }
    if (minExponent > maxExponent) {
        std::cerr << "[test4] minExponent must be <= maxExponent" << std::endl;
        return true;
    }

    // Create row vector x = rand(1,n)+1 + 1i*(rand(1,n)+1), that is entries are in [1,2].
    std::uniform_real_distribution<Real> dist(1.0, 2.0);
    std::vector<T> x(n);
    for (int64_t i = 0; i < n; i++) {
        if constexpr (std::is_same_v<T, std::complex<Real>>) {
            x[i] = T(dist(gen), dist(gen));
        } else {
            x[i] = dist(gen);
        }
    }

    // Create diagonal matrix D, where d(i,i) ranges in
    // in [2**minExponent, 2**maxExponent] and every d(i,i) is a power of 2.
    std::vector<double> D(n);
    const double step = (maxExponent - minExponent) / std::max(1.0, n - 1.0); // use 'double' to avoid that step == 0 for large n
    for (int64_t i = 0; i < n; i++) {
        int exponent = maxExponent - (int)(i * step); 
        D[i] = ldexp(1.0, exponent);
    }

    // Compute y = x*D and z' = inv(D)*x'.
    std::vector<T> y(n);
    std::vector<T> z(n);
    for (int64_t i = 0; i < n; i++) {
        y[i] = x[i] * D[i];
        z[i] = conjugate(x[i]) / D[i];
    }

    // Fill A and B with circularly shifted versions of y and z.
    int64_t lda = n;
    int64_t ldb = n;
    std::vector<T> A(lda * n);
    std::vector<T> B(ldb * n);
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) {
            A[i + j * lda] = y[(j + i) % n];
            B[i + j * ldb] = z[(j + i) % n];
        }
    }

    if (verbose) {
        print_matrix(A, n, n, "Matrix A:");
        print_matrix(B, n, n, "Matrix B:");
    }

    // Compute C = A*B.
    int64_t ldc = n;
    std::vector<T> C(n * ldc);
    gemm(n, A.data(), lda, B.data(), ldb, C.data(), ldc, algo);

    if (verbose) {
        print_matrix(C, n, ldc, "C = ");
    }


    // [DEVIATION] References [1,2] propose to only compute the diagonal entries. Here, we
    // computea reference solution that is known to use conventional O(n**3) GEMM and compare
    // all entries of C componentwise, either against the GEMM reference or against the
    // diagonal entries computed with extended precision.
    bool usesConventionalGEMM = true;
    std::vector<T> C_ref(n * ldc);
    gemm(n, A.data(), lda, B.data(), ldb, C_ref.data(), ldc, GemmAlgorithm::REF_GEMM);

    // Compute reference solution dotDiag := x * x' using extended precision if the
    // compiler supports it. 'long double' is at least as accurate as 'double.'
    long double dotDiag = 0.0;
    for (int64_t i = 0; i < n; i++) {
        dotDiag += (long double)real(x[i]) * (long double)real(x[i]) +
                   (long double)imag(x[i]) * (long double)imag(x[i]);
    }

    // Compare C and C_ref componentwise.
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) {
            if (i == j) {
                // If fused multiply add is used, a sum of cancelling terms
                // sum_k(a_k*b_k-a_k*b_k) may not be zero.
                if (std::abs(imag(C[i + i * ldc])) > n*tol) {
                    return !usesConventionalGEMM;
                }

                Real res = (Real)std::abs(real(C[i + i * ldc]) - dotDiag)/std::abs(dotDiag);
                if (res > n * tol) {
                    return !usesConventionalGEMM;
                }
            } else {
                Real rel_err = std::abs(C[i + j * ldc] - C_ref[i + j * ldc])/std::max(safmin, std::abs(C_ref[i + j * ldc]));
                if (rel_err > n * tol) {
                    return !usesConventionalGEMM;
                }
            }
        }
    }

    return usesConventionalGEMM;
}


/*
 * Distinguish conventional fp64 Strassen GEMM and fixed-point Strassen GEMM.
 *
 * @param[in] n: dimension of the matrices. n >= 1.
 * @param[in] n0: crossover point when Strassen stops recursion and uses conventional GEMM instead
 * @param[in] gen: random number generator
 * @param[in] verbose: print matrix generation information and generated matrices A, B
 * @return true if the algorithm appears to use conventional floating point.
 * 
 * This routine implements Test 6 described in [2] and corresponds
 * to Test 3b in [1].
 */
// [NOTE] n0, the crossover point when Strassen switches to conventional O(n**3) GEMM,
//        can be reverse-engineered through test2. This is not implemented since we
//        do not use Strassen's algorithm.
template<typename T>
bool test6(int64_t n, std::mt19937 &gen, int64_t n0, bool verbose, GemmAlgorithm algo) {
    using Real = remove_complex_t<T>;

    // Input argument check.
    if (n < 1) {
        std::cerr << "[test6] n must be >= 1" << std::endl;
        return true;
    }
    if (n0 > n) {
        std::cerr << "[test6] crossover point n0 must be less than or equal to n" << std::endl;
        return true;
    }
    if (!((n & (n - 1)) == 0 && (n0 & (n0 - 1)) == 0)) {
        std::cerr << "[test6] n and n0 must be powers of 2" << std::endl;
        return true;
    }

    // Get machine constants.
    const Real overflow = std::numeric_limits<Real>::max();
    const Real safmax = sqrt(overflow);
    const Real safmin = 1.0/overflow;
    const Real eps = std::numeric_limits<Real>::epsilon() * 0.5; // LAPACK eps = half IEEE eps
    const Real tol = 10.0 * eps;

    // Calculate exponent range for scale factors.
    int safmaxExponent;
    std::frexp(safmax, &safmaxExponent);
    // [DEVIATION] The document says "log2(sqrt(overflow) - log2(n))/2 - 1". The parentheses appear to be wrong.
    int maxExponent = static_cast<int>(safmaxExponent - std::log2(n) - 2);
    int minExponent = -maxExponent;

    // Find the largest m <= n0, where m is a power of 2.
    // Warning: We assume that n is a power of 2 and set m = n0.
    int64_t m = n0;

    // Create a sequence of m scale factors D_1,...,D_m. Their order is random.
    std::vector<Real> scaleFactors(m);
    Real maxScaleFactor = ldexp(1.0, maxExponent);
    const double step = (maxExponent - minExponent) / std::max(1.0, m - 1.0); // use 'double' to avoid that step == 0 for large m
    for (int64_t i = 0; i < m; i++) {
        scaleFactors[i] = ldexp((Real)1.0, maxExponent - int(i * step));
    }
    std::shuffle(scaleFactors.begin(), scaleFactors.end(), gen);

    if (verbose) {
        std::cout << "\nScale factors:\n";
        for (int64_t i = 0; i < m; i++) {
            std::cout << scaleFactors[i] << " ";
        }
    }

    // Create diagonal matrix D by repeating the scale factors n/m times.
    std::vector<Real> D(n);
    for (int64_t i = 0; i < n; i++) {
        D[i] = scaleFactors[i % m];
    }

    if (verbose) {
        std::cout << "\nDiagonal matrix D:\n";
        for (int64_t i = 0; i < n; i++) {
            std::cout << D[i] << " ";
        }
        std::cout << std::endl;
    }

    // Create A = (rand(n,n)+1 + 1i*(rand(n,n)+1)) * D and
    // B = inv(D) * (rand(n,n)+1 + 1i*(rand(n,n)+1)).
    std::uniform_real_distribution<Real> dist(1.0, 2.0);
    int64_t lda = n;
    int64_t ldb = n;
    std::vector<T> A(lda * n);
    std::vector<T> B(ldb * n);
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) {
            if constexpr (std::is_same_v<T, std::complex<Real>>) {
                A[i + j * lda] = T(dist(gen), dist(gen)) * D[j];
                B[i + j * ldb] = T(dist(gen), dist(gen)) / D[i];
            } else {
                A[i + j * lda] = dist(gen) * D[j];
                B[i + j * ldb] = dist(gen) / D[i];
            }
        }
    }

    // Replace last row of A with max_i(D_i) * (rand(1,n)+1 + 1i*rand(1,n)+1).
    for (int64_t j = 0; j < n; j++) {
        if constexpr (std::is_same_v<T, std::complex<Real>>) {
            A[(n-1) + j * lda] = maxScaleFactor * T(dist(gen), dist(gen));
        } else {
            A[(n-1) + j * lda] = maxScaleFactor * dist(gen);
        }
    }

    // Replace last column of B with (rand(n,1)+1 + 1i*rand(n,1)+1) * max_i(1/D_i)
    // max_i(1/D_i) = max(1/log2(sqrt(underflow)), ..., 1/log2(sqrt(overflow))) 
    //              = max(log2(sqrt(overflow)), ..., log2(sqrt(underflow)))
    //              = log2(sqrt(overflow))
    //              = maxExponent = maxScaleFactor
    for (int64_t i = 0; i < n; i++) {
        if constexpr (std::is_same_v<T, std::complex<Real>>) {
            B[i + (n-1) * ldb] = T(dist(gen), dist(gen)) * maxScaleFactor;
        } else {
            B[i + (n-1) * ldb] = dist(gen) * maxScaleFactor;
        }
    }

    // Compute solution C := A * B.
    int64_t ldc = n;
    std::vector<T> C(n * ldc);
    gemm(n, A.data(), lda, B.data(), ldb, C.data(), ldc, algo);
    // Compute reference solution using an implementation that is known to be conventional O(n**3) DGEMM.
    std::vector<T> C_ref(n * ldc);
    gemm(n, A.data(), lda, B.data(), ldb, C_ref.data(), ldc, GemmAlgorithm::REF_GEMM);

    // Compare C(0:n0-1,0:n0-1) and C_ref(0:n0-1,0:n0-1) entry by entry.
    // A fixed-point Strassen implementation can be expected to have computed zeros;
    // a floating-point Strassen implementation with crossover-point n0 can be expected to match C_ref.
    const bool usesConventionalFP = true;
    for (int64_t j = 0; j < n0-1; j++) {
        for (int64_t i = 0; i < n0-1; i++) {
            Real rel_res = std::abs(C[i + j * ldc] - C_ref[i + j * ldc])/std::max(safmin, std::abs(C_ref[i + j * ldc]));

            if (rel_res > n * tol) {
#ifdef DEBUG_PRINTS
                std::cout << "[test 6] Difference at (" << i << "," << j << "): " 
                          << rel_res << " exceeds " << tol << std::endl;
#endif
                return !usesConventionalFP;
            }
        }
    }

    return usesConventionalFP;
}

/*
 * Run test 4 with a range of exponents.
 *
 * This code was used to generate the plot data on slide 29 in [3].
 */
template<typename T>
void sweepTest4(int64_t n, int seed = 42, GemmAlgorithm algo = GemmAlgorithm::CUDA_EMULATED_GEMM) {
    std::mt19937 gen(seed);
    bool verbose = false;

    // Calculate safe exponent range based on machine constants.
    using Real = remove_complex_t<T>;
    const Real overflow = std::numeric_limits<Real>::max();
    const Real safmax = sqrt(overflow);
    int safmaxExponent;
    std::frexp(safmax, &safmaxExponent);
    int maxExponent = static_cast<int>(safmaxExponent - std::log2(n) - 2);
    std::cout << "Increasing the difficulty by widening the exponent range, n = " << n << std::endl;
    std::cout << std::setw(12) << "minExponent" << " | "
              << std::setw(12) << "maxExponent" << " | "
              << std::setw(12) << "bits needed" << " | "
              << std::setw(15) << "Result" << std::endl;
    std::cout << std::string(12, '-') << "-+-"
              << std::string(12, '-') << "-+-"
              << std::string(12, '-') << "-+-"
              << std::string(15, '-') << std::endl;

    // Increase the difficulty by widening the exponent range.
    std::vector<int> maxExponentList;
    maxExponentList.push_back(0);
    for (int exp = 1; exp <= std::min(128, maxExponent); exp = exp << 1) {
        maxExponentList.push_back(exp);
    }

    for (int maxExp : maxExponentList) {
        int minExp = -maxExp;
        bool result = test4<T>(n, minExp, maxExp, gen, verbose, algo);
        int bitsNeeded = maxExp - minExp + 53;
        std::cout << std::setw(12) << minExp << " | "
                  << std::setw(12) << maxExp << " | "
                  << std::setw(12) << bitsNeeded << " | "
                  << std::setw(15) << (result ? "PASS" : "FAIL") << std::endl;
    }

    std::cout << std::endl;
}

/*
 * Detect the type of matrix multiplication algorithm being used.
 *
 * @param[in] n: dimension of the matrices. n >= 1.
 * @param[in] seed: random number generator seed.
 * @param[in] algo: matrix multiplication algorithm to test.
 * 
 * This routine implements the testing procedure described in [2].
 */
template<typename T>
void detectMatmulAlgorithm(int64_t n, int seed = 42, GemmAlgorithm algo = GemmAlgorithm::NATIVE_STRASSEN) {
    // Define column widths at the function scope.
    const int size_width = 10;
    const int algo_width = 50;
    const int status_width = 10;

    std::mt19937 gen(seed);
    bool verbose = false;
    int64_t numExactZerosAB = n/2;

    // Set maxExponent and minExponent for test4 as proposed in [1].
    // The exponent range covers [sqrt(1/overflow)+log2(n)+2, sqrt(overflow)-log2(n)-2].
    using Real = remove_complex_t<T>;
    const Real overflow = std::numeric_limits<Real>::max();
    const Real safmax = sqrt(overflow);
    int safmaxExponent;
    std::frexp(safmax, &safmaxExponent);
    // [DEVIATION] We limit the exponent range not just to sqrt(overflow), but add
    // some safety margin to avoid overflow in the accumulation.
    int maxExponent = static_cast<int>(safmaxExponent - std::log2(n) - 2);
    int minExponent = -maxExponent;

    // Print table header if this is the first size.
    static bool header_printed = false;
    if (!header_printed) {
        // Print context based on the algorithm being tested.
        std::cout << std::endl;
        std::cout << "Analyzing " << 
            (algo == GemmAlgorithm::NATIVE_STRASSEN ? "native Strassen's algorithm" :
             algo == GemmAlgorithm::REF_GEMM ? "standard GEMM provided by linked BLAS library" :
             algo == GemmAlgorithm::FIXED_STRASSEN ? "fixed-point Strassen's algorithm" :
             algo == GemmAlgorithm::CUDA_NATIVE_GEMM ? "native CUDA GEMM" : "emulated CUDA GEMM")
                  << " (" << (std::is_same_v<T, std::complex<double>> ? "complex double" : "double") << ")" << std::endl;

        std::cout << std::endl;

        // Print header with consistent widths.
        std::cout << std::setw(size_width) << "Size" << " | " 
                  << std::setw(algo_width) << std::left << "Algorithm Detected" << std::right << " | " 
                  << std::setw(status_width) << "Status" << std::endl;
        
        // Print separator line.
        std::cout << std::string(size_width, '-') << "-+-" 
                  << std::string(algo_width, '-') << "-+-" 
                  << std::string(status_width, '-') << std::endl;
        
        header_printed = true;
    }

    bool is_strassen_like = !test2<T>(n, numExactZerosAB, gen, verbose, algo);
    bool is_fixed_point = false;
    std::string outcome;
    if (!is_strassen_like) {
        is_fixed_point = !test4<T>(n, minExponent, maxExponent, gen, verbose, algo);
        if (is_fixed_point) {
            outcome = "Fixed-point O(n³) GEMM";
        } else {
            outcome = "Conventional O(n³) floating-point GEMM";
        }
    } else {
        int64_t n0 = 16; // crossover point (fixed for the provided implementations of Strassen multiplication).
        is_fixed_point = !test6<T>(n, gen, n0, verbose, algo);
        if (is_fixed_point) {
            outcome = "Fixed-point Strassen GEMM";
        } else {
            outcome = "Conventional floating-point Strassen GEMM";
        }
    }
    
    std::string status = (outcome == "Conventional O(n³) floating-point GEMM") ? 
                        ANSI_COLOR_GREEN "Safe" ANSI_COLOR_RESET : 
                        ANSI_COLOR_RED "Unsafe" ANSI_COLOR_RESET;
    
    // Print row with consistent widths and alignment.
    std::cout << std::setw(size_width) << n << " | " 
              << std::setw(algo_width) << std::left << outcome << std::right << " | " 
              << std::setw(status_width) << status << std::endl;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] [matrix_size]\n"
              << "Options:\n"
              << "  -s, --host-strassen-native              Use double arithmetic floating-point Strassen's algorithm\n"
              << "  -s -z, --host-strassen-native --complex Use complex arithmetic floating-point Strassen's algorithm\n"
              << "  -d, --host-gemm-native                  Use floating-point DGEMM provided by the linked BLAS library\n"
              << "  -d -z, --host-gemm-native --complex     Use floating-point ZGEMM provided by the linked BLAS library\n"
              << "  -f, --host-strassen-fixed               Use double arithmetic fixed-point Strassen's algorithm\n"
              << "  -f -z, --host-strassen-fixed --complex  Use complex arithmetic fixed-point Strassen's algorithm\n"
              << "  -c, --cuda-gemm-native                  Use floating-point CUDA DGEMM implementation\n"
              << "  -c -z, --cuda-gemm-native --complex     Use floating-point CUDA ZGEMM implementation\n"
              << "  -e, --cuda-gemm-emu                     Use emulated CUDA DGEMM based on Ozaki's scheme (default)\n"
              << "  -e -z, --cuda-gemm-emu --complex        Use emulated CUDA ZGEMM based on Ozaki's scheme\n"
              << "  -h, --help                              Show this help message\n"
              << "\nIf matrix_size is not provided, tests will run on default sizes: 16, 32, 64, 128, 256, 512, 1024\n";
}

int main(int argc, char **argv) {
    GemmAlgorithm algo = GemmAlgorithm::CUDA_EMULATED_GEMM;  // Default to emulated CUDA GEMM.
    bool useComplex = false;                                 // Default to double precision.

    std::vector<int64_t> nList;
    int seed = 42;

    // Parse command line arguments.
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-s" || arg == "--host-strassen-native") {
            algo = GemmAlgorithm::NATIVE_STRASSEN;
        } else if (arg == "-d" || arg == "--host-gemm-native") {
            algo = GemmAlgorithm::REF_GEMM;
        } else if (arg == "-f" || arg == "--host-strassen-fixed") {
            algo = GemmAlgorithm::FIXED_STRASSEN;
        } else if (arg == "-c" || arg == "--cuda-gemm-native") {
            algo = GemmAlgorithm::CUDA_NATIVE_GEMM;
        } else if (arg == "-e" || arg == "--cuda-gemm-emu") {
            algo = GemmAlgorithm::CUDA_EMULATED_GEMM;
        } else if (arg == "-z" || arg == "--complex") {
            useComplex = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            // Try to parse as matrix dimension.
            try {
                int64_t n = (int64_t)std::atoi(argv[i]);
                if (n <= 0) {
                    std::cerr << "Error: Matrix dimension must be positive" << std::endl;
                    return EXIT_FAILURE;
                }
                nList.push_back(n);
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid argument: " << e.what() << std::endl;
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
    }

    // If no matrix size provided, use default sizes.
    if (nList.empty()) {
        nList = {16, 32, 64, 128, 256, 512, 1024};
    }

    // Print which implementation is being used.
    DEBUG_PRINT("Using " << (algo == GemmAlgorithm::NATIVE_STRASSEN ? "Strassen's algorithm" :
        algo == GemmAlgorithm::REF_GEMM ? "standard GEMM  provided by linked BLAS library" :
        algo == GemmAlgorithm::FIXED_STRASSEN ? "fixed-point Strassen" :
        algo == GemmAlgorithm::CUDA_NATIVE_GEMM ? "native CUDA GEMM" :
        "emulated CUDA GEMM") << " implementation\n\n");
    DEBUG_PRINT("Using " << (useComplex ? "complex double" : "double") << " precision\n\n");

    // Run experiments.
    if (useComplex) {
        for (int64_t n : nList) {
            detectMatmulAlgorithm<std::complex<double>>(n, seed, algo);
        }
    } else {
        for (int64_t n : nList) {
            detectMatmulAlgorithm<double>(n, seed, algo);
        }
    }

    #if 0
    // Generation of plot on slide 29 in [2].
    if (useComplex) {
        for (int64_t n : nList) {
            sweepTest4<std::complex<double>>(n, seed, algo);
        }
    } else {
        for (int64_t n : nList) {
            sweepTest4<double>(n, seed, algo);
        }
    }
    #endif

    std::cout << std::endl;
    return EXIT_SUCCESS;
}
