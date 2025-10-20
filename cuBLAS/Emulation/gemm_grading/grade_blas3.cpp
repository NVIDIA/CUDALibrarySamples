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

#include <cstring>
#include <algorithm>
#include <random>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <vector>

#include "utils.hxx"
#include "gemms.hxx"

//
// References:
//
// [1] Jim Demmel, Xiaoye Li, Julien Langou, Weslley Pereira, Mark Gates, Cindy Rubio Gonzalez (2024),
//     "How to grade the accuracy of an implementation of the BLAS," slides available at:
//     https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Grading_BLAS.pdf
//

// Acknowledgments:
// We gratefully acknowledge Prof. James W. Demmel for his generous support, for sharing
// the manuscript [2], and for his review and feedback on this work prior to publication.
//

/* Criterion 2 (Grade "A") defined in [1] is the most stringent criterion to evalute
 * the accuracy of an implementation of matrix multiplication. It considers the
 * componentwise relative error bound
 *     |fl((A*B)(i,j)) - (A*B)(i,j)| <= f(n)*eps*(|A|*|B|)(i,j)
 * For grade "A" compliance, f(n) must not exceed linear growth.
 * 
 * @param[in] m: number of rows of the matrix A and of the matrices C and CC.
 * @param[in] n: number of columns of the matrix B and of the matrices C and CC.
 * @param[in] k: number of columns of the matrix A and the number of rows of the matrix B.
 * @param[in] alpha_ptr: pointer to the scalar alpha.
 * @param[in] A: m-by-k matrix A.
 * @param[in] lda: leading dimension of A.
 * @param[in] B: k-by-n matrix B.
 * @param[in] ldb: leading dimension of B.
 * @param[in] beta_ptr: pointer to the scalar beta.
 * @param[in] C: m-by-n matrix C.
 * @param[in] ldc: leading dimension of C.
 * @param[in] CC: m-by-n matrix holding the computed solution that shall be checked.
 * @param[in] ldcc: leading dimension of CC.
 * @return The maximum and average componentwise relative error.
 */
template<typename T>
std::pair<double,double> test_criterion2(
    // Input problem: alpha * A * B + beta * C
    int64_t m, int64_t n, int64_t k, const T *alpha_ptr,
    const T *A, int64_t lda, const T* B, int64_t ldb,
    const T *beta_ptr, const T *C, int64_t ldc,
    // Computed solution that shall be checked: CC := alpha * A * B + beta * C
    const T *CC, int64_t ldcc) {
    const double eps = std::numeric_limits<double>::epsilon();

    T cj[m]; // a column of the result matrix
    double abscj[m];

    const T alpha = *alpha_ptr;
    const T beta = *beta_ptr;

    double max_err = 0.0;
    double avg_err = 0.0;

    // Compute one column of C at a time.
    for (int64_t j = 0; j < n; j++) {
        memset(cj, 0, sizeof(T) * m);
        memset(abscj, 0, sizeof(double) * m);

        // Compute (A*B)(i,j)
        for (int64_t l = 0; l < k; l++) {
            for (int64_t i = 0; i < m; i++) {
                cj[i] += A[i + l * lda] * B[l + j * ldb];
                abscj[i] += std::abs(A[i + l * lda]) * std::abs(B[l + j * ldb]);
            }
        }

        for (int64_t i = 0; i < m; i++) {
            cj[i] = alpha * cj[i] + beta * C[i + j * ldc];
            abscj[i] = std::abs(alpha) * abscj[i] + std::abs(beta) * std::abs(C[i + j * ldc]);
        }

        double err = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double e = std::abs(cj[i] - CC[i + j * ldcc])/eps;
            if (abscj[i] > 0.0) {
                e /= abscj[i];
            }
            err = std::max(err, e);
            avg_err += e;
        }

        max_err = std::max(max_err, err);
    } // for j

    return {max_err, (avg_err/n)/m};
}

// A = rand(n,n) + 1i*rand(n,n)
template<typename T>
void rand(int64_t n, T *A, int64_t lda, std::mt19937 &gen) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) {
            if constexpr (std::is_same_v<T, std::complex<double>>) {
                A[i + j * lda] = T(dist(gen), dist(gen));
            } else {
                A[i + j * lda] = dist(gen);
            }
        }
    }
}

// A = 0.
template<typename T>
void zero(int64_t n, T *A, int64_t lda, std::mt19937 &gen) {
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < n; i++) {
            A[i + j * lda] = T(0);
        }
    }
}

/*
 * Simple linear regression.
 *
 * @param[in] x, y: The data pairs (x[i], y[i]).
 * @return The slope b of the data fitted to the model y = a + b x.
 */
double fitLinearModel(const std::vector<double> &x, const std::vector<double> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y must have the same size");
    }

    const size_t n = x.size();

    // Compute the mean of x and y.
    long double sumx = 0.0, sumy = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sumx += (long double)(x[i]);
        sumy += (long double)(y[i]);
    }
    const long double meanx = sumx / n;
    const long double meany = sumy / n;

    // Compute
    //  * SSxx = Σ (x[i] - meanx)^2
    //  * SSxy = Σ (x[i] - meanx) * (y[i] - meany)
    long double SSxx = 0.0, SSxy = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const long double dx = (long double)(x[i]) - meanx;
        const long double dy = (long double)(y[i]) - meany;
        SSxx += dx * dx;
        SSxy += dx * dy;
    }

    const double slope = static_cast<double>(SSxy / SSxx);

    return slope;
}


/*
 * Grades square matrix multiplication.
 *
 * @param[in] test_sizes: the list of matrix sizes to test.
 * @param[in] seed: the seed for the random number generator.
 * @param[in] algo: the algorithm to use.
 * @param[in] init_A_fcn: the function to initialize the matrix A.
 * @param[in] init_B_fcn: the function to initialize the matrix B.
 * @param[in] init_C_fcn: the function to initialize the matrix C.
 */
template<typename T>
void grade(std::vector<int64_t> &test_sizes,
           int seed = 42,
           GemmAlgorithm algo = GemmAlgorithm::CUDA_EMULATED_GEMM,
           void (*init_A_fcn)(int64_t n, T* A, int64_t lda, std::mt19937&) = rand,
           void (*init_B_fcn)(int64_t n, T* B, int64_t ldb, std::mt19937&) = rand,
           void (*init_C_fcn)(int64_t n, T* C, int64_t ldc, std::mt19937&) = zero) {
    std::mt19937 gen(seed);

    // Find the largest matrix size.
    int64_t max_n = *std::max_element(test_sizes.begin(), test_sizes.end());

    // Number of samples run for each matrix size.
    int64_t num_samples = 1;

    // Data points (x[i],y[i]).
    std::vector<double> x;
    std::vector<double> y;

    // Allocate matrices.
    int64_t lda = max_n;
    int64_t ldb = max_n;
    int64_t ldc = max_n;
    std::vector<T> A(lda * max_n);
    std::vector<T> B(ldb * max_n);
    std::vector<T> C(ldc * max_n);
    std::vector<T> CC(ldc * max_n);

    T alpha = 1.0;
    T beta = 0.0;

    // Initialize matrices.
    for (int64_t n : test_sizes) {
        for (int64_t i = 0; i < num_samples; i++) {
            init_A_fcn(n, A.data(), lda, gen);
            init_B_fcn(n, B.data(), ldb, gen);
            init_C_fcn(n, C.data(), ldc, gen);

            // Compute CC = A * B.
            gemm(n, A.data(), lda, B.data(), ldb, CC.data(), ldc, algo);

            auto [max_error, avg_error] = test_criterion2(
                n, n, n, &alpha, A.data(), lda, B.data(), ldb,
                &beta, C.data(), ldc, CC.data(), ldc);

            std::cout << "n = " << n << " max error = " << max_error << " avg error = " << avg_error << std::endl;

            // Record pair (x,y) = (log(n), log(max_error)).
            if (max_error > 0.0 && n > 0) {
                x.push_back(std::log((double)n));
                y.push_back(std::log(max_error));
            }
        }
    }

    double slope =fitLinearModel(x, y);
    std::cout << "For grade A compliance, errors must not exceed linear growth.\n"
              << "Under the assumption that the error grows linearly over the tested\n"
              << "range, the fitted model in log-log scale, log(error) = a + b * log(n), has slope " << slope << ".\n"
              << "For grade A compliance, the slope in log-log scale must be <= 1.\n"
              << "We highlight that this is not a proof; a formal claim requires proper statistical testing.\n";
}

void print_usage(const char* program_name) {
    std::cout <<"Usage: " << program_name << " [options] [matrix_size]\n"
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
    GemmAlgorithm algo = GemmAlgorithm::CUDA_EMULATED_GEMM;  // Default to emulated CUDA GEMM
    bool useComplex = false;                                 // Default to double precision

    std::vector<int64_t> nList;
    int seed = 42;

    // Parse command line arguments
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
        } else {
            // Try to parse as matrix dimension
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

    // If no matrix size provided, use default sizes
    if (nList.empty()) {
        nList = {16, 32, 64, 128, 256, 512, 1024};
    }

    // Print which implementation is being used
    DEBUG_PRINT("Using " << (algo == GemmAlgorithm::NATIVE_STRASSEN ? "native Strassen's algorithm" :
                             algo == GemmAlgorithm::REF_GEMM ? "standard GEMM provided by linked BLAS library" :
                             algo == GemmAlgorithm::FIXED_STRASSEN ? "fixed-point Strassen's algorithm" :
                             algo == GemmAlgorithm::CUDA_NATIVE_GEMM ? "native CUDA GEMM" :
                            "emulated CUDA GEMM") << "\n\n");
    DEBUG_PRINT("Using " << (useComplex ? "complex double" : "double") << " precision\n\n");

    // Grade the implementation
    if (useComplex) {
        grade<std::complex<double>>(nList, seed, algo);
    } else {
        grade<double>(nList, seed, algo);
    }

    std::cout << std::endl;
    return EXIT_SUCCESS;
}
