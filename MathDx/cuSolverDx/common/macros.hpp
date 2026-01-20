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

#ifndef CUSOLVERDX_EXAMPLE_COMMON_MACROS_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_MACROS_HPP

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

#ifndef CUSOLVER_CHECK_AND_EXIT
#    define CUSOLVER_CHECK_AND_EXIT(error)                                              \
        {                                                                               \
            auto status = static_cast<cusolverStatus_t>(error);                         \
            if (status != CUSOLVER_STATUS_SUCCESS) {                                    \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUSOLVER_CHECK

#ifndef CUSPARSE_CHECK_AND_EXIT
#    define CUSPARSE_CHECK_AND_EXIT(error)                                              \
        {                                                                               \
            auto status = static_cast<cusparseStatus_t>(error);                         \
            if (status != CUSPARSE_STATUS_SUCCESS) {                                    \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUSPARSE_CHECK_AND_EXIT

#ifndef CUBLAS_CHECK_AND_EXIT
#    define CUBLAS_CHECK_AND_EXIT(error)                                              \
        {                                                                               \
            auto status = static_cast<cublasStatus_t>(error);                         \
            if (status != CUBLAS_STATUS_SUCCESS) {                                    \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUSOLVER_CHECK

#endif // CUSOLVERDX_EXAMPLE_COMMON_MACROS_HPP
