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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <iostream>

#ifndef DEBUG_PRINT
#ifdef DEBUG_PRINTS
#define DEBUG_PRINT(x) std::cout << x
#else
#define DEBUG_PRINT(x)
#endif
#endif

// Helper function to check CUDA errors.
static void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        DEBUG_PRINT("CUDA error: " << msg << " - " << cudaGetErrorString(error) << std::endl);
        exit(EXIT_FAILURE);
    }
}

// Helper function to check cuBLAS errors.
static void checkCublasError(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        DEBUG_PRINT("cuBLAS error: " << msg << std::endl);
        exit(EXIT_FAILURE);
    }
}

inline cublasHandle_t getCublasHandle() {
    static cublasHandle_t handle{};
    static bool initialized = false;
    if (!initialized) {
      checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");
      std::atexit([]{ (void)cublasDestroy(handle); });
      initialized = true;
    }
    return handle;
}

// Allocate A, B, and C on the device.
template <typename T>
static inline void allocateMatricesOnDevice(
    int n, int lda, int ldb, int ldc, T *&d_A, T *&d_B, T *&d_C,
    size_t &size_A, size_t &size_B, size_t &size_C) {
    size_A = n * lda * sizeof(T);
    size_B = n * ldb * sizeof(T);
    size_C = n * ldc * sizeof(T);

    checkCudaError(cudaMalloc(&d_A, size_A), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc(&d_B, size_B), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc(&d_C, size_C), "Failed to allocate device memory for C");
}

// Compute C := A * B with cuBLAS.
template<typename T>
void cublas_native_gemm(int n, const T *A, int lda, const T *B, int ldb, T *C, int ldc) {
    DEBUG_PRINT("Using cuBLAS algorithm (no emulation)" << std::endl);

    cublasHandle_t handle = getCublasHandle();

    // Allocate device memory.
    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = 0, size_B = 0, size_C = 0;
    allocateMatricesOnDevice<T>(n, lda, ldb, ldc, d_A, d_B, d_C, size_A, size_B, size_C);

    // Copy input matrices to device.
    checkCudaError(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice), "Failed to copy B to device");

    // Compute C := A * B.
    const T alpha = T(1.0);
    const T beta = T(0.0);

    cudaDataType dtype = std::is_same_v<T, std::complex<double>> ? CUDA_C_64F : CUDA_R_64F;
    checkCublasError(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        reinterpret_cast<const void*>(&alpha),
        (const void*)d_A, dtype, lda,
        (const void*)d_B, dtype, ldb,
        reinterpret_cast<const void*>(&beta),
        (void*)d_C, dtype, ldc,
        CUBLAS_COMPUTE_64F,
        CUBLAS_GEMM_DEFAULT),
      "Failed to execute cublasGemmEx");

    // Copy result back to host.
    checkCudaError(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost), "Failed to copy result back to host");

    // Clean up.
    checkCudaError(cudaFree(d_A), "Failed to free device memory for A");
    checkCudaError(cudaFree(d_B), "Failed to free device memory for B");
    checkCudaError(cudaFree(d_C), "Failed to free device memory for C");
}

// Compute C := A * B with cuBLAS.
template<typename T>
void cublas_emulated_gemm(int n, const T *A, int lda, const T *B, int ldb, T *C, int ldc) {

    DEBUG_PRINT("Using cuBLAS algorithm (emulation)" << std::endl);

    cublasHandle_t handle = getCublasHandle();

    // Detailed examples of how to configure emulated DGEMM can be found in
    // the example Emulation/dgemm_dynamic and Emulation/dgemm_fixed.

    cudaEmulationStrategy_t strategy = CUDA_EMULATION_STRATEGY_EAGER;
    cudaEmulationSpecialValuesSupport svHandling = CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;
    int maxMantissaBitCount = 200;

    checkCublasError(cublasSetFixedPointEmulationMaxMantissaBitCount(handle, maxMantissaBitCount),
        "Failed to set the number of mantissa bits");

    checkCublasError(cublasSetEmulationStrategy(handle, strategy),
            "Failed to set the emulation strategy");

    checkCublasError(cublasSetEmulationSpecialValuesSupport(handle, svHandling),
        "Failed to set the value handling");

    // Allocate device memory.
    T *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = 0, size_B = 0, size_C = 0;
    allocateMatricesOnDevice<T>(n, lda, ldb, ldc, d_A, d_B, d_C, size_A, size_B, size_C);

    // Copy input matrices to device.
    checkCudaError(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice), "Failed to copy B to device");

    // Compute C := A * B.
    const T alpha = T(1.0);
    const T beta = T(0.0);

    cudaDataType dtype = std::is_same_v<T, std::complex<double>> ? CUDA_C_64F : CUDA_R_64F;
    checkCublasError(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        reinterpret_cast<const void*>(&alpha),
        (const void*)d_A, dtype, lda,
        (const void*)d_B, dtype, ldb,
        reinterpret_cast<const void*>(&beta),
        (void*)d_C, dtype, ldc,
        CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
        CUBLAS_GEMM_DEFAULT),
      "Failed to execute cublasGemmEx");

    // Copy result back to host.
    checkCudaError(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost), "Failed to copy result back to host");

    // Clean up.
    checkCudaError(cudaFree(d_A), "Failed to free device memory for A");
    checkCudaError(cudaFree(d_B), "Failed to free device memory for B");
    checkCudaError(cudaFree(d_C), "Failed to free device memory for C");
}
