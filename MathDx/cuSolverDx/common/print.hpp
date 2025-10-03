/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUSOLVERDX_EXAMPLE_COMMON_PRINT_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_PRINT_HPP

#include <string>
#include <stdio.h>

#include <cusolverdx.hpp>
#include <numeric>

namespace common {

    template<typename T>
    void print_matrix(const unsigned int M, const unsigned int N, bool is_col_major, const T* A, const unsigned int lda, const int nbatches = 1) {
        if (is_col_major) {
            if constexpr (!is_complex<T>()) {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N * nbatches; j++) {
                        std::printf("%0.2f ", A[j * lda + i]);
                    }
                    printf("\n");
                }
            } else {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N * nbatches; j++) {
                        std::printf("%0.2f + %0.2fj ", A[j * lda + i].x, A[j * lda + i].y);
                    }
                    printf("\n");
                }
            }
        } else { // row major
            if constexpr (!is_complex<T>()) {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N * nbatches; j++) {
                        std::printf("%0.2f ", A[i * lda + j]);
                    }
                    printf("\n");
                }
            } else {
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N * nbatches; j++) {
                        std::printf("%0.2f + %0.2fj ", A[i * lda + j].x, A[i * lda + j].y);
                    }
                    printf("\n");
                }
            }
        }
    }

    template<typename T, const unsigned int M, const unsigned int N, const unsigned int LDA = M, bool is_col_major = true>
    void print_matrix(const T* A, const int nbatches = 1) {
        print_matrix(M, N, is_col_major, A, LDA, nbatches);
    }


    __forceinline__ __device__ bool block0() {
        return (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y) == 0;
    }

    __forceinline__ __device__ bool thread0() {
        return (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) == 0;
    }

    inline std::string type_string(cusolverdx::type t) {
        return (t == cusolverdx::type::real) ? "real" : "complex";
    }

    template<typename T>
    std::enable_if_t<not is_complex<T>(), std::string> precision_string() {
        if (std::is_same_v<T, __half>) {
            return "__half";
        } else if (std::is_same_v<T, float>) {
            return "float";
        } else if (std::is_same_v<T, double>) {
            return "double";
        }

        return "unknown precision";
    }

    template<typename T>
    std::enable_if_t<is_complex<T>(), std::string> precision_string() {
        return precision_string<typename T::value_type>();
    }

    template<typename CUSOLVERDX>
    __forceinline__ __device__ void print(CUSOLVERDX) {
        if (thread0() and block0()) {
            printf("\nOPERATORS =======================================================\n");
            printf("\n");
            printf("BlockDim: \n");
            if constexpr (cusolverdx::detail::has_operator_v<cusolverdx::operator_type::block_dim, CUSOLVERDX>) {
                auto bd = cusolverdx::block_dim_of_v<CUSOLVERDX>;
                printf("Value: %u %u %u \n\n", bd.x, bd.y, bd.z);
            } else {
                printf("Value: Absent\n\n");
            }

            printf("Block: \n");
            if constexpr (cusolverdx::detail::has_operator_v<cusolverdx::operator_type::block, CUSOLVERDX>) {
                printf("Value: Present \n\n");
            } else {
                printf("Value: Absent\n\n");
            }

            printf("Function: \n");
            if constexpr (cusolverdx::detail::has_operator_v<cusolverdx::operator_type::function, CUSOLVERDX>) {
                auto opt = cusolverdx::function_of_v<CUSOLVERDX>;
                auto msg = opt == cusolverdx::function::potrf ? "potrf" : "potrs";
                printf("Value: %s \n\n", msg);
            } else {
                printf("Value: Absent\n\n");
            }

            printf("SolverPrecision: \n");
            if constexpr (cusolverdx::detail::has_operator_v<cusolverdx::operator_type::precision, CUSOLVERDX>) {
                using prec = cusolverdx::precision_of<CUSOLVERDX>;

                auto prec_str = [](auto v) {
                    if (COMMONDX_STL_NAMESPACE::is_same_v<decltype(v), __half>) {
                        return "__half";
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<decltype(v), float>) {
                        return "float";
                    } else if (COMMONDX_STL_NAMESPACE::is_same_v<decltype(v), double>) {
                        return "double";
                    }

                    return "unknown precision";
                };
                printf("Value: %s %s %s \n\n",
                       prec_str(typename prec::x_type {}),
                       prec_str(typename prec::y_type {}),
                       prec_str(typename prec::z_type {}));
            } else {
                printf("Value: Absent\n\n");
            }


            printf("SolverSize: \n");
            if constexpr (cusolverdx::detail::has_operator_v<cusolverdx::operator_type::size, CUSOLVERDX>) {
                auto [x, y, z] = cusolverdx::size_of_v<CUSOLVERDX>;
                printf("Value: %u %u %u \n\n", x, y, z);
            } else {
                printf("Value: Absent\n\n");
            }

            printf("SM: \n");
            if constexpr (cusolverdx::detail::has_operator_v<cusolverdx::operator_type::sm, CUSOLVERDX>) {
                auto sm_v = cusolverdx::sm_of_v<CUSOLVERDX>;
                printf("Value: %u \n\n", sm_v);
            } else {
                printf("Value: Absent\n\n");
            }

            printf("Type: \n");
            if constexpr (cusolverdx::detail::has_operator_v<cusolverdx::operator_type::type, CUSOLVERDX>) {
                auto opt = cusolverdx::type_of_v<CUSOLVERDX>;
                auto msg = opt == cusolverdx::type::real ? "real" : "complex";
                printf("Value: %s \n\n", msg);
            } else {
                printf("Value: Absent\n\n");
            }

            printf("Is type complete? \n");
            auto msg = CUSOLVERDX::is_complete() ? "Yes" : "No";
            printf("Value: %s \n \n", msg);

            printf("END =============================================================\n\n");
        }
    }

} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_PRINT_HPP