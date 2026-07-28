/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUBLASDX_EXAMPLE_COMMON_NVRTC_HPP
#define CUBLASDX_EXAMPLE_COMMON_NVRTC_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
#define CUBLASDX_EXAMPLE_NVRTC
#include "../common/common.hpp"

#define NVRTC_SAFE_CALL(x)                                                                            \
    do {                                                                                              \
        nvrtcResult result = x;                                                                       \
        if (result != NVRTC_SUCCESS) {                                                                \
            std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

#define NVJITLINK_SAFE_CALL(h, x)                                                \
    do {                                                                         \
        nvJitLinkResult call_result = x;                                         \
        if (call_result != NVJITLINK_SUCCESS) {                                  \
            std::cerr << "\nerror: " #x " failed with error " << call_result << '\n'; \
            size_t lsize;                                                        \
            nvJitLinkResult log_result = nvJitLinkGetErrorLogSize(h, &lsize);    \
            if (log_result == NVJITLINK_SUCCESS && lsize > 0) {                  \
                std::vector<char> log(lsize);                                    \
                log_result = nvJitLinkGetErrorLog(h, log.data());                \
                if (log_result == NVJITLINK_SUCCESS) {                           \
                    std::cerr << "error: " << log.data() << '\n';                \
                }                                                                \
            }                                                                    \
            std::exit(call_result);                                              \
        }                                                                        \
    } while (0)

#ifndef CU_CHECK_AND_EXIT
#    define CU_CHECK_AND_EXIT(error)                                                  \
        {                                                                             \
            auto status = static_cast<CUresult>(error);                               \
            if (status != CUDA_SUCCESS) {                                             \
                const char* pstr;                                                     \
                cuGetErrorString(status, &pstr);                                      \
                std::cout << pstr << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                    \
            }                                                                         \
        }
#endif // CU_CHECK_AND_EXIT

namespace example {
    namespace nvrtc {
        template<class Type>
        inline Type get_global_from_module(CUmodule module, const char* name) {
            CUdeviceptr value_ptr;
            size_t      value_size;
            CU_CHECK_AND_EXIT(cuModuleGetGlobal(&value_ptr, &value_size, module, name));
            Type value_host;
            CU_CHECK_AND_EXIT(cuMemcpyDtoH(&value_host, value_ptr, value_size));
            return value_host;
        }

        inline std::vector<std::string> get_cublasdx_include_dirs() {
            std::vector<std::string> cublasdx_include_dirs_array;

            auto append_multiple_dirs = [](auto& container, const std::string& semicolon_separated_dirs) {
                if (semicolon_separated_dirs.empty())
                    return;

                std::stringstream ss(semicolon_separated_dirs);
                std::string       dir;
                while (std::getline(ss, dir, ';')) {
                    if (!dir.empty()) { // Skip empty directories
                        container.push_back("--include-path=" + dir);
                    }
                }
            };

            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_COMMONDX_INCLUDE_DIR");
                if (env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                } else {
#ifdef COMMONDX_INCLUDE_DIR
                    { cublasdx_include_dirs_array.push_back("--include-path=" + std::string(COMMONDX_INCLUDE_DIR)); }
#endif
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUTLASS_INCLUDE_DIR");
                if (env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                } else {
#ifdef CUTLASS_INCLUDE_DIR
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(CUTLASS_INCLUDE_DIR));
#endif
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUBLASDX_INCLUDE_DIR");
                if (env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                } else {
#ifdef CUBLASDX_INCLUDE_DIRS
                    append_multiple_dirs(cublasdx_include_dirs_array, std::string(CUBLASDX_INCLUDE_DIRS));
#endif
                }
            }
            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_CUDA_INCLUDE_DIR");
                if (env_ptr != nullptr) {
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr));
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(env_ptr) + "/cccl");
                } else {
#ifdef CUDA_INCLUDE_DIR
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(CUDA_INCLUDE_DIR));
                    cublasdx_include_dirs_array.push_back("--include-path=" + std::string(CUDA_INCLUDE_DIR) + "/cccl");
#endif
                }
            }

            {
                const char* env_ptr = std::getenv("CUBLASDX_EXAMPLE_USER_DIRECTORIES");
                if (env_ptr != nullptr) {
                    append_multiple_dirs(cublasdx_include_dirs_array, std::string(env_ptr));
                }
            }
            return cublasdx_include_dirs_array;
        }

        template<class T = float>
        inline std::vector<T> generate_random_data(const std::size_t length,
                                                   const double      min = 1.0,
                                                   const double      max = 1.0) {
            std::vector<T> data(length);
            std::random_device                     rd;
            std::mt19937                           gen(rd());
            std::uniform_real_distribution<double> dist(min, max);
            std::generate(data.begin(), data.end(), [&]() { return static_cast<T>(dist(gen)); });
            return data;
        }

        inline void make_lower_col_major_diagonal_dominant(std::vector<float>& a, const unsigned m) {
            for (unsigned row = 0; row < m; row++) {
                float offdiag_sum = 5.f;
                for (unsigned col = 0; col < m; col++) {
                    if (col != row) {
                        offdiag_sum += std::abs(a[row + col * m]);
                    }
                }
                a[row + row * m] = offdiag_sum;
            }
        }

        inline double relative_l2_error(const std::vector<float>& data, const std::vector<float>& reference) {
            double error_sq = 0.0;
            double norm_sq  = 0.0;
            for (std::size_t i = 0; i < data.size(); i++) {
                const double diff = std::abs(static_cast<double>(data[i]) - static_cast<double>(reference[i]));
                error_sq += diff * diff;
                norm_sq += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
            }
            return std::sqrt(error_sq / (norm_sq + 1e-200));
        }

        inline bool check_float_result(const std::vector<float>& data,
                                       const std::vector<float>& reference,
                                       // Simple, naive example check; not a rigorously derived error bound.
                                       const double              max_relative_l2_error = 1e-3) {
            const double error = relative_l2_error(data, reference);
            if (error > max_relative_l2_error) {
                std::cout << error << std::endl;
                return false;
            }
            return true;
        }

        inline unsigned get_device_architecture(int device) {
            int major = 0;
            int minor = 0;
            CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
            CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
            return major * 10 + minor;
        }

        inline std::string get_device_architecture_option(int device) {
            // --gpus-architecture=compute_... will generate PTX, which means NVRTC must be at least as recent as the CUDA driver;
            // --gpus-architecture=sm_... will generate SASS, which will always run on any CUDA driver from the current major
            std::string gpu_architecture_option =
                "--gpu-architecture=sm_" + std::to_string(get_device_architecture(device));
            return gpu_architecture_option;
        }

        inline void print_program_log(const nvrtcProgram prog) {
            size_t log_size;
            NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
            char* log = new char[log_size];
            NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
            std::cout << log << '\n';
            delete[] log;
        }
    } // namespace nvrtc

    // nvJitLink uses "-arch=sm_XX", while NVRTC uses "--gpu-architecture=sm_XX".
    namespace nvjitlink {
        inline std::string get_device_architecture_option(int device) {
            return "-arch=sm_" + std::to_string(nvrtc::get_device_architecture(device));
        }
    } // namespace nvjitlink
} // namespace example

#endif // CUBLASDX_EXAMPLE_COMMON_NVRTC_HPP
