/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cublas_api.h>
#include <cusolverDn.h>
#include <library_types.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUSPARSE_CHECK(err)                                                                        \
    do {                                                                                           \
        cusparseStatus_t err_ = (err);                                                             \
        if (err_ != CUSPARSE_STATUS_SUCCESS) {                                                     \
            printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusparse error");                                            \
        }                                                                                          \
    } while (0)

// memory alignment
#define ALIGN_TO(A, B) (((A + B - 1) / B) * B)

// device memory pitch alignment
static const size_t device_alignment = 32;

// type traits
template <typename T> struct traits;

template <> struct traits<float> {
    // scalar type
    typedef float T;
    typedef T S;

    static constexpr T zero = 0.f;
    static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_32F;
#endif

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, S f) { return v * f; }
};

template <> struct traits<double> {
    // scalar type
    typedef double T;
    typedef T S;

    static constexpr T zero = 0.;
    static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_64F;
#endif

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, S f) { return v * f; }
};

template <> struct traits<cuFloatComplex> {
    // scalar type
    typedef float S;
    typedef cuFloatComplex T;

    static constexpr T zero = {0.f, 0.f};
    static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_32F;
#endif

    inline static S abs(T val) { return cuCabsf(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return make_cuFloatComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCaddf(a, b); }
    inline static T add(T a, S b) { return cuCaddf(a, make_cuFloatComplex(b, 0.f)); }

    inline static T mul(T v, S f) { return make_cuFloatComplex(v.x * f, v.y * f); }
};

template <> struct traits<cuDoubleComplex> {
    // scalar type
    typedef double S;
    typedef cuDoubleComplex T;

    static constexpr T zero = {0., 0.};
    static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_64F;
#endif

    inline static S abs(T val) { return cuCabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return make_cuDoubleComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCadd(a, b); }
    inline static T add(T a, S b) { return cuCadd(a, make_cuDoubleComplex(b, 0.)); }

    inline static T mul(T v, S f) { return make_cuDoubleComplex(v.x * f, v.y * f); }
};

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const cuComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const cuDoubleComplex *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
        }
        std::printf("\n");
    }
}

template <typename T>
void generate_random_matrix(cusolver_int_t m, cusolver_int_t n, T **A, int *lda) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<typename traits<T>::S> dis(-1.0, 1.0);
    auto rand_gen = std::bind(dis, gen);

    *lda = n;

    size_t matrix_mem_size = static_cast<size_t>(*lda * m * sizeof(T));
    // suppress gcc 7 size warning
    if (matrix_mem_size <= PTRDIFF_MAX)
        *A = (T *)malloc(matrix_mem_size);
    else
        throw std::runtime_error("Memory allocation size is too large");

    if (*A == NULL)
        throw std::runtime_error("Unable to allocate host matrix");

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            T *A_col = (*A) + *lda * j;
            A_col[i] = traits<T>::rand(rand_gen);
        }
    }
}

// Makes matrix A of size mxn and leading dimension lda diagonal dominant
template <typename T>
void make_diag_dominant_matrix(cusolver_int_t m, cusolver_int_t n, T *A, int lda) {
    for (int j = 0; j < std::min(m, n); ++j) {
        T *A_col = A + lda * j;
        auto col_sum = traits<typename traits<T>::S>::zero;
        for (int i = 0; i < m; ++i) {
            col_sum += traits<T>::abs(A_col[i]);
        }
        A_col[j] = traits<T>::add(A_col[j], col_sum);
    }
}

// Returns cudaDataType value as defined in library_types.h for the string containing type name
cudaDataType get_cuda_library_type(std::string type_string) {
    if (type_string.compare("CUDA_R_16F") == 0)
        return CUDA_R_16F;
    else if (type_string.compare("CUDA_C_16F") == 0)
        return CUDA_C_16F;
    else if (type_string.compare("CUDA_R_32F") == 0)
        return CUDA_R_32F;
    else if (type_string.compare("CUDA_C_32F") == 0)
        return CUDA_C_32F;
    else if (type_string.compare("CUDA_R_64F") == 0)
        return CUDA_R_64F;
    else if (type_string.compare("CUDA_C_64F") == 0)
        return CUDA_C_64F;
    else if (type_string.compare("CUDA_R_8I") == 0)
        return CUDA_R_8I;
    else if (type_string.compare("CUDA_C_8I") == 0)
        return CUDA_C_8I;
    else if (type_string.compare("CUDA_R_8U") == 0)
        return CUDA_R_8U;
    else if (type_string.compare("CUDA_C_8U") == 0)
        return CUDA_C_8U;
    else if (type_string.compare("CUDA_R_32I") == 0)
        return CUDA_R_32I;
    else if (type_string.compare("CUDA_C_32I") == 0)
        return CUDA_C_32I;
    else if (type_string.compare("CUDA_R_32U") == 0)
        return CUDA_R_32U;
    else if (type_string.compare("CUDA_C_32U") == 0)
        return CUDA_C_32U;
    else
        throw std::runtime_error("Unknown CUDA datatype");
}

// Returns cusolverIRSRefinement_t value as defined in cusolver_common.h for the string containing
// solver name
cusolverIRSRefinement_t get_cusolver_refinement_solver(std::string solver_string) {
    if (solver_string.compare("CUSOLVER_IRS_REFINE_NONE") == 0)
        return CUSOLVER_IRS_REFINE_NONE;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_CLASSICAL") == 0)
        return CUSOLVER_IRS_REFINE_CLASSICAL;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_GMRES;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_CLASSICAL_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_CLASSICAL_GMRES;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_GMRES_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_GMRES_GMRES;
    else
        printf("Unknown solver parameter: \"%s\"\n", solver_string.c_str());

    return CUSOLVER_IRS_REFINE_NOT_SET;
}