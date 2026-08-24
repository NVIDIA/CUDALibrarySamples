/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef LIBMATHDX_COMMON_MACROS_HPP
#define LIBMATHDX_COMMON_MACROS_HPP

#include <cstdio>
#include <cstdlib>

#ifndef ASSERT
#define ASSERT(ans)                                                                                                    \
    do {                                                                                                               \
        bool res = (ans);                                                                                              \
        if (!res) {                                                                                                    \
            fprintf(stderr, "ASSERT %s failed on %s:%d\n", #ans, __FILE__, __LINE__);                                  \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0);
#endif // ASSERT

#ifndef NVRTC_CHECK
#define NVRTC_CHECK(ans)                                                                                               \
    do {                                                                                                               \
        nvrtcResult result = (ans);                                                                                    \
        if (result != NVRTC_SUCCESS) {                                                                                 \
            fprintf(stderr, "NVRTC error %s at %s:%d\n", nvrtcGetErrorString(result), __FILE__, __LINE__);             \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif // NVRTC_CHECK

#ifndef NVJITLINK_CHECK
#define NVJITLINK_CHECK(handle, ans)                                                                                   \
    do {                                                                                                               \
        nvJitLinkResult result = (ans);                                                                                \
        if (result != NVJITLINK_SUCCESS) {                                                                             \
            fprintf(stderr, "nvJitLink error: %d on %s:%d\n", (int)result, __FILE__, __LINE__);                        \
            size_t lsize;                                                                                              \
            result = nvJitLinkGetErrorLogSize(handle, &lsize);                                                         \
            if (result == NVJITLINK_SUCCESS && lsize > 0) {                                                            \
                std::vector<char> log(lsize);                                                                          \
                result = nvJitLinkGetErrorLog(handle, log.data());                                                     \
                if (result == NVJITLINK_SUCCESS) {                                                                     \
                    fprintf(stderr, "%s\n", log.data());                                                               \
                }                                                                                                      \
            }                                                                                                          \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif // NVJITLINK_CHECK

#ifndef CUDA_CHECK
#define CUDA_CHECK(ans)                                                                                                \
    do {                                                                                                               \
        cudaError_t status = (ans);                                                                                    \
        if (status != cudaSuccess) {                                                                                   \
            fprintf(stderr, "CUDA error %s on %s:%d\n", cudaGetErrorString(status), __FILE__, __LINE__);               \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif // CUDA_CHECK

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(ans)                                                                                              \
    do {                                                                                                               \
        cublasStatus_t status = (ans);                                                                                 \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            fprintf(stderr, "cuBLAS error %d on %s:%d\n", (int)status, __FILE__, __LINE__);                            \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif // CUBLAS_CHECK

#ifndef CU_CHECK
#define CU_CHECK(ans)                                                                                                  \
    do {                                                                                                               \
        CUresult status = (ans);                                                                                       \
        if (status != CUDA_SUCCESS) {                                                                                  \
            const char* ptr { nullptr };                                                                               \
            cuGetErrorString(status, &ptr);                                                                            \
            fprintf(stderr, "CUDA error %s on %s:%d\n", ptr, __FILE__, __LINE__);                                      \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif // CU_CHECK

#ifndef CURAND_CHECK
#define CURAND_CHECK(ans)                                                                                              \
    do {                                                                                                               \
        curandStatus_t status = (ans);                                                                                 \
        if (status != CURAND_STATUS_SUCCESS) {                                                                         \
            fprintf(stderr, "CURAND error %d on %s:%d\n", (int)status, __FILE__, __LINE__);                            \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif // CURAND_CHECK

#ifndef LIBMATHDX_CHECK
#define LIBMATHDX_CHECK(ans)                                                                                           \
    do {                                                                                                               \
        commondxStatusType status = (ans);                                                                             \
        if (status != COMMONDX_SUCCESS) {                                                                              \
            fprintf(stderr, "libmathdx error %d on %s:%d\n", status, __FILE__, __LINE__);                              \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif // LIBMATHDX_CHECK

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#define LIB_STD_17
#endif

#endif // LIBMATHDX_COMMON_MACROS_HPP
