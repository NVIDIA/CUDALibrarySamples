#ifndef CUSOLVERDX_EXAMPLE_COMMON_RANDOM_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_RANDOM_HPP

#include <cassert>
#include "numeric.hpp"

namespace common {

    template<typename T>
    std::enable_if_t<!is_complex<T>(), std::vector<T>> generate_random_data(const float min, const float max, const size_t size) {
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        std::vector<T> ret(size);
        for (auto& v : ret) {
            v = convert<T>(dist(gen));
        }
        return ret;
    }

    template<typename T>
    std::enable_if_t<is_complex<T>(), std::vector<T>> generate_random_data(const float min, const float max, const size_t size) {
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        std::vector<T> ret(size);
        for (auto& v : ret) {
            using scalar_type = typename get_precision<T>::type;
            scalar_type r     = static_cast<scalar_type>(dist(gen));
            scalar_type i     = static_cast<scalar_type>(dist(gen));
            v.x               = r;
            v.y               = i;
        }
        return ret;
    }

    // fill up host array
    template<typename T>
    std::enable_if_t<is_complex<T>(), void> fillup_random_matrix(const bool         is_column_major,
                                                                 const unsigned int m, // rows
                                                                 const unsigned int n, // columns
                                                                 T*                 A,
                                                                 const unsigned int lda,
                                                                 const bool         symm,
                                                                 const bool         diag_dom,
                                                                 const float        min,
                                                                 const float        max,
                                                                 const unsigned int batches = 1) {

        using precision = typename get_precision<T>::type;
        std::random_device                        rd;
        std::default_random_engine                gen(rd());
        std::uniform_real_distribution<precision> distribution(min, max);

        std::vector<T> h_data = generate_random_data<T>(min, max, m * n * batches);

        const unsigned int row_stride = is_column_major ? 1 : lda;
        const unsigned int col_stride = is_column_major ? lda : 1;

        for (unsigned int batch = 0; batch < batches; batch++) {
            // copy the random m*n values from vector to the matrix A
            for (unsigned int col = 0; col < n; col++) {
                for (int row = 0; row < m; row++) {
                    auto idx                                               = row + col * m;
                    A[row * row_stride + col * col_stride + batch * m * n] = h_data[idx + batch * m * n];
                }
            }

            if (symm) {
                // A is Hermitian
                for (unsigned int col = 0; col < n; col++) {
                    for (unsigned int row = 0; row < col; row++) {
                        // upper triangular matrix
                        T Areg = A[row * row_stride + col * col_stride + batch * m * n];
                        // conjugate and copy to lower triangular matrix
                        A[row * row_stride + col * col_stride + batch * m * n]   = Areg;
                        A[row * row_stride + col * col_stride + batch * m * n].y = -A[row * row_stride + col * col_stride + batch * m * n].y;
                    }
                }
            }
            if (diag_dom) {
                // reset diag(A) such that A is diagonal dominant
                for (unsigned int row = 0; row < m; row++) {
                    cuDoubleComplex offdiag_sum = {0.0, 0.0};
                    for (unsigned int col = 0; col < n; col++) {
                        if (col != row) {
                            T Areg = A[row * row_stride + col * col_stride + batch * m * n];
                            offdiag_sum.x += std::abs(Areg.x);
                            offdiag_sum.y += std::abs(Areg.y);
                        }
                    }
                    if (row < n) {
                        A[row * row_stride + row * col_stride + batch * m * n].x = offdiag_sum.x + 5.0;
                        A[row * row_stride + row * col_stride + batch * m * n].y = offdiag_sum.y + 5.0;
                    }
                }
            }
        }
    }

    // fill up host array
    template<typename T>
    std::enable_if_t<!is_complex<T>(), void> fillup_random_matrix(const bool         is_column_major,
                                                                  const unsigned int m, // rows
                                                                  const unsigned int n, // columns
                                                                  T*                 A,
                                                                  const unsigned int lda,
                                                                  const bool         symm,
                                                                  const bool         diag_dom,
                                                                  const float        min,
                                                                  const float        max,
                                                                  const unsigned int batches = 1) {

        using precision = typename get_precision<T>::type;
        std::random_device                        rd;
        std::default_random_engine                gen(rd());
        std::uniform_real_distribution<precision> distribution(min, max);

        std::vector<T> h_data = generate_random_data<T>(min, max, m * n * batches);

        const unsigned int row_stride = is_column_major ? 1 : lda;
        const unsigned int col_stride = is_column_major ? lda : 1;

        for (unsigned int batch = 0; batch < batches; batch++) {
            // copy the random m*n values from vector to the matrix A
            for (unsigned int col = 0; col < n; col++) {
                for (int row = 0; row < m; row++) {
                    auto idx                                               = row + col * m;
                    A[row * row_stride + col * col_stride + batch * m * n] = h_data[idx + batch * m * n];
                }
            }

            if (symm) {
                // A is Hermitian
                for (unsigned int col = 0; col < n; col++) {
                    for (unsigned int row = 0; row < col; row++) {
                        // upper triangular matrix
                        T Areg = A[row * row_stride + col * col_stride + batch * m * n];
                        // conjugate and copy to lower triangular matrix
                        A[col * row_stride + row * col_stride + batch * m * n] = Areg;
                    }
                }
            }
            if (diag_dom) {
                // reset diag(A) such that A is diagonal dominant
                for (unsigned int row = 0; row < m; row++) {
                    double offdiag_sum {0.0};
                    for (unsigned int col = 0; col < n; col++) {
                        if (col != row) {
                            T Areg = A[row * row_stride + col * col_stride + batch * m * n];
                            offdiag_sum += std::abs(Areg);
                        }
                    }
                    if (row < n) {
                        A[row * row_stride + row * col_stride + batch * m * n] = offdiag_sum + 5.0;
                    }
                }
            }
        }
    }

    template<typename T>
    void fillup_random_matrix_col_major(const unsigned int m, // rows
                                        const unsigned int n, // columns
                                        T*                 A,
                                        const unsigned int lda,
                                        const bool         symm,
                                        const bool         diag_dom,
                                        const float        min,
                                        const float        max,
                                        const unsigned int batches = 1) {
        fillup_random_matrix(true, m, n, A, lda, symm, diag_dom, min, max, batches);
    }

    template<typename T>
    void fillup_random_diagonal_dominant_matrix(const bool         is_column_major,
                                                const unsigned int m, // rows
                                                const unsigned int n, // columns
                                                T*                 A,
                                                const unsigned int lda,
                                                const bool         symm,
                                                const float        min,
                                                const float        max,
                                                const unsigned int batches = 1) {
        fillup_random_matrix(is_column_major, m, n, A, lda, symm, true, min, max, batches);
    }

    template<typename T>
    void fillup_random_diagonal_dominant_matrix_col_major(const unsigned int m, // rows
                                                          const unsigned int n, // columns
                                                          T*                 A,
                                                          const unsigned int lda,
                                                          const bool         symm,
                                                          const float        min,
                                                          const float        max,
                                                          const unsigned int batches = 1) {
        fillup_random_matrix(true, m, n, A, lda, symm, true, min, max, batches);
    }

    template<typename T>
    void transpose_matrix(std::vector<T>& A, const unsigned int dim_fast, const unsigned int dim_slow, const unsigned batches) {
        assert(A.size() == dim_fast * dim_slow * batches);

        std::vector<T> A_temp(A);
        for (auto i = 0; i < batches; i++) {
            for (auto j = 0; j < dim_slow; j++) {
                for (auto k = 0; k < dim_fast; k++) {
                    A[i * dim_fast * dim_slow + j * dim_fast + k] = A_temp[i * dim_slow * dim_fast + k * dim_slow + j];
                }
            }
        }
    }


} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_RANDOM_HPP
