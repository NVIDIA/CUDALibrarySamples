/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#if !defined(__cplusplus)
#error "This file can only by include in C++ file because it overload function names"
#endif

#include <cuComplex.h>

static inline __host__ __device__ int div_up(int m, int n) {
    return (m + n - 1) / n;
};

/*---------------------------------------------------------------------------------*/
/* Real part , Imaginary part */
static __inline__ __device__ __host__ float cuReal(float x) {
    return (x);
}

static __inline__ __device__ __host__ double cuReal(double x) {
    return (x);
}

static __inline__ __device__ __host__ float cuReal(cuComplex x) {
    return (cuCrealf(x));
}
static __inline__ __device__ __host__ double cuReal(cuDoubleComplex x) {
    return (cuCreal(x));
}

static __inline__ __device__ __host__ float cuImag(float x) {
    return (0.0f);
}

static __inline__ __device__ __host__ double cuImag(double x) {
    return (0.0);
}

static __inline__ __device__ __host__ float cuImag(cuComplex x) {
    return (cuCimagf(x));
}
static __inline__ __device__ __host__ double cuImag(cuDoubleComplex x) {
    return (cuCimag(x));
}

/* Absolute Value */
static __inline__ __device__ __host__ float cuAbs(float x) {
    return (fabsf(x));
}

static __inline__ __device__ __host__ double cuAbs(double x) {
    return (fabs(x));
}

static __inline__ __device__ __host__ float cuAbs(cuComplex x) {
    return (cuCabsf(x));
}

static __inline__ __device__ __host__ double cuAbs(cuDoubleComplex x) {
    return (cuCabs(x));
}

/*---------------------------------------------------------------------------------*/
/* Conjugate */
static __inline__ __device__ __host__ float cuConj(float x) {
    return (x);
}

static __inline__ __device__ __host__ double cuConj(double x) {
    return (x);
}

static __inline__ __device__ __host__ cuComplex cuConj(cuComplex x) {
    return (cuConjf(x));
}
/*---------------------------------------------------------------------------------*/

#if __CUDA_ARCH__ < 600
template <typename T>
__inline__ __device__ T atomicAdd(T *address, const T value) {
    return atomicAdd(address, value);
}

template <>
__inline__ __device__ double atomicAdd(double *address, const double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int  old            = *address_as_ull, assumed;

    do {
        assumed = old;
        old     = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val + __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
        // NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

template <>
__inline__ __device__ float atomicAdd(float *address, const float val) {
    unsigned int *address_as_ull = (unsigned int *)address;
    unsigned int  old            = *address_as_ull, assumed;

    do {
        assumed = old;
        old     = atomicCAS(address_as_ull, assumed,
                            __float_as_int(val + __int_as_float(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
        // NaN)
    } while (assumed != old);

    return __int_as_float(old);
}
#endif
