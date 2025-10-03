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

#ifndef CUBLASDX_EXAMPLE_REFERENCE_HPP
#define CUBLASDX_EXAMPLE_REFERENCE_HPP

#include <iostream>
#include <type_traits>
#include "../common/common.hpp"

#include "cublaslt_runner.hpp"
#include "naive_reference.hpp"
#include "check_error.hpp"

namespace example {

    template<class BLAS,
             class Alpha,
             class AValueType,
             class BValueType,
             class Beta,
             class CValueType,
             class ALoadOp              = cublasdx::identity,
             class BLoadOp              = cublasdx::identity,
             class CLoadOp              = cublasdx::identity,
             class CStoreOp             = cublasdx::identity,
             class ComputeReferenceType = detail::get_reference_value_type_t<typename BLAS::c_value_type>>
    std::vector<ComputeReferenceType>
    reference_gemm(Alpha                                       alpha,
                   const example::device_vector<AValueType>&   data_a,
                   unsigned int                                lda,
                   const example::device_vector<BValueType>&   data_b,
                   unsigned int                                ldb,
                   Beta                                        beta,
                   const example::device_vector<CValueType>&   data_c,
                   unsigned int                                ldc,
                   cudaStream_t                                stream = 0,
                   const ALoadOp&                              a_load_op  = {},
                   const BLoadOp&                              b_load_op  = {},
                   const CLoadOp&                              c_load_op  = {},
                   const CStoreOp&                             c_store_op = {}) {
        using compute_value_type = ComputeReferenceType;

        constexpr bool use_cublas = commondx::is_floating_point_v<compute_value_type>;

        #if (THRUST_VERSION >= 101600)
        auto execution_policy = thrust::cuda::par_nosync.on(stream);
        #else
        auto execution_policy = thrust::cuda::par.on(stream);
        #endif

        // cuBLAS will perform this transformation on its own, thus there is no reason to do it
        static constexpr bool additional_conj_a = cublasdx::transpose_mode_of_a<BLAS> == cublasdx::transpose_mode::conj_transposed;
        static constexpr bool additional_conj_b = cublasdx::transpose_mode_of_b<BLAS> == cublasdx::transpose_mode::conj_transposed;

        auto additonal_a_transform = cute::conditional_return<additional_conj_a>(cublasdx::conjugate{}, cublasdx::identity{});
        auto additonal_b_transform = cute::conditional_return<additional_conj_b>(cublasdx::conjugate{}, cublasdx::identity{});

        // first [conjugate] then transform
        auto full_a_transform = cublasdx::detail::compose_functors(additonal_a_transform, a_load_op);
        auto full_b_transform = cublasdx::detail::compose_functors(additonal_b_transform, b_load_op);

        auto a_load_transform_then_convert = cublasdx::detail::compose_functors(full_a_transform, converter<compute_value_type>{});
        auto b_load_transform_then_convert = cublasdx::detail::compose_functors(full_b_transform, converter<compute_value_type>{});
        auto c_load_transform_then_convert = cublasdx::detail::compose_functors(c_load_op, converter<compute_value_type>{});
        [[maybe_unused]] auto c_store_transform_then_convert = cublasdx::detail::compose_functors(c_store_op, converter<compute_value_type>{});

        static constexpr unsigned int                  m = cublasdx::size_of<BLAS>::m;
        static constexpr unsigned int                  n = cublasdx::size_of<BLAS>::n;
        [[maybe_unused]] static constexpr unsigned int k = cublasdx::size_of<BLAS>::k;

        example::device_vector<compute_value_type> ref_data_a(data_a.size());
        example::device_vector<compute_value_type> ref_data_b(data_b.size());
        example::device_vector<compute_value_type> ref_data_c(data_c.size());

        thrust::transform(execution_policy, data_a.cbegin(), data_a.cend(), ref_data_a.begin(), a_load_transform_then_convert);
        thrust::transform(execution_policy, data_b.cbegin(), data_b.cend(), ref_data_b.begin(), b_load_transform_then_convert);
        thrust::transform(execution_policy, data_c.cbegin(), data_c.cend(), ref_data_c.begin(), c_load_transform_then_convert);

        using arr = cublasdx::arrangement_of<BLAS>;
        auto gemm_shape = cute::make_tuple(m, n, k);
        auto gemm_arr   = cute::make_tuple(arr::a, arr::b, arr::c);
        auto gemm_ld    = cute::make_tuple(lda, ldb, ldc);

        if constexpr(use_cublas) {
            cublaslt_runner<compute_value_type>(gemm_shape, gemm_arr, gemm_ld)
                .execute(convert<compute_value_type>(alpha),
                         ref_data_a.data(),
                         ref_data_b.data(),
                         convert<compute_value_type>(beta),
                         ref_data_c.data(),
                         stream);
        } else {
            reference_gemm_naive_device(gemm_shape, gemm_arr, gemm_ld,
                                        convert<compute_value_type>(alpha),
                                        ref_data_a,
                                        ref_data_b,
                                        convert<compute_value_type>(beta),
                                        ref_data_c);
        }

        if constexpr (!std::is_same_v<CStoreOp, cublasdx::identity>) {
            thrust::transform(execution_policy, ref_data_c.cbegin(), ref_data_c.cend(), ref_data_c.begin(), c_store_transform_then_convert);
        }

        return ref_data_c;
    }

    template<class BLAS,
             class Alpha,
             class AValueType, class AAllocType,
             class BValueType, class BAllocType,
             class Beta,
             class CValueType, class CAllocType,
             class ALoadOp             = cublasdx::identity,
             class BLoadOp             = cublasdx::identity,
             class CLoadOp             = cublasdx::identity,
             class CStoreOp            = cublasdx::identity,
             class ComputeReferenceType = detail::get_reference_value_type_t<typename BLAS::c_value_type>>
    std::vector<ComputeReferenceType>
    reference_gemm(Alpha                                       alpha,
                   const std::vector<AValueType, AAllocType>&  data_a,
                   unsigned int                                lda,
                   const std::vector<BValueType, BAllocType>&  data_b,
                   unsigned int                                ldb,
                   Beta                                        beta,
                   const std::vector<CValueType, CAllocType>&  data_c,
                   unsigned int                                ldc,
                   cudaStream_t                                stream = 0,
                   const ALoadOp&                              a_load_op  = {},
                   const BLoadOp&                              b_load_op  = {},
                   const CLoadOp&                              c_load_op  = {},
                   const CStoreOp&                             c_store_op = {}) {
        example::device_vector<AValueType> device_a = data_a;
        example::device_vector<BValueType> device_b = data_b;
        example::device_vector<CValueType> device_c = data_c;

        return reference_gemm<BLAS>(
            alpha,
            device_a,
            lda,
            device_b,
            ldb,
            beta,
            device_c,
            ldc,
            stream,
            a_load_op,
            b_load_op,
            c_load_op,
            c_store_op
        );
    }

    // Generate dynamic LeadingDimensions for cuBLAS
    template<class BLAS,
             template<class, class>   class DataVector,
             class Alpha,
             class AValueType, class AAllocType,
             class BValueType, class BAllocType,
             class Beta,
             class CValueType, class CAllocType,
             class ALoadOp             = cublasdx::identity,
             class BLoadOp             = cublasdx::identity,
             class CLoadOp             = cublasdx::identity,
             class CStoreOp            = cublasdx::identity,
             class ComputeReferenceType = detail::get_reference_value_type_t<typename BLAS::c_value_type>>
    const std::vector<ComputeReferenceType>
    reference_gemm(Alpha                                     alpha,
                   const DataVector<AValueType, AAllocType>& data_a,
                   const DataVector<BValueType, BAllocType>& data_b,
                   Beta                                      beta,
                   const DataVector<CValueType, CAllocType>& data_c,
                   cudaStream_t                              stream = 0,
                   const ALoadOp&                            a_load_op  = {},
                   const BLoadOp&                            b_load_op  = {},
                   const CLoadOp&                            c_load_op  = {},
                   const CStoreOp&                           c_store_op = {}) {
        const auto [m, n, k] = cublasdx::size_of<BLAS>::value;
        const auto lda = cublasdx::arrangement_of<BLAS>::a == cublasdx::arrangement::col_major ? m : k;
        const auto ldb = cublasdx::arrangement_of<BLAS>::b == cublasdx::arrangement::col_major ? k : n;
        const auto ldc = cublasdx::arrangement_of<BLAS>::c == cublasdx::arrangement::col_major ? m : n;

        return reference_gemm<BLAS>(
            alpha, data_a, lda, data_b, ldb, beta, data_c, ldc, stream, a_load_op, b_load_op, c_load_op, c_store_op);
    }
}

#endif // CUBLASDX_EXAMPLE_REFERENCE_HPP