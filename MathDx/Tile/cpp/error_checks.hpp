// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cusolverDn.h>

#include <cstdio>
#include <cstdlib>

namespace mathdx_examples {

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) return;

    std::fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, expression, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

inline void check_cufft(cufftResult status, const char* expression, const char* file, int line) {
    if (status == CUFFT_SUCCESS) return;

    std::fprintf(stderr, "%s:%d: %s failed: %d\n", file, line, expression, static_cast<int>(status));
    std::exit(EXIT_FAILURE);
}

inline void check_cublas(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status == CUBLAS_STATUS_SUCCESS) return;

    std::fprintf(stderr, "%s:%d: %s failed: %d\n", file, line, expression, static_cast<int>(status));
    std::exit(EXIT_FAILURE);
}

inline void check_cusolver(cusolverStatus_t status, const char* expression, const char* file, int line) {
    if (status == CUSOLVER_STATUS_SUCCESS) return;

    std::fprintf(stderr, "%s:%d: %s failed: %d\n", file, line, expression, static_cast<int>(status));
    std::exit(EXIT_FAILURE);
}

} // namespace mathdx_examples

#define CUDA_CHECK(...) ::mathdx_examples::check_cuda((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__)
#define CUFFT_CHECK(...) ::mathdx_examples::check_cufft((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__)
#define CUBLAS_CHECK(...) ::mathdx_examples::check_cublas((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__)
#define CUSOLVER_CHECK(...) ::mathdx_examples::check_cusolver((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__)
