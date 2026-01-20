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

#ifndef CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP
#define CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP

#include <type_traits>
#include "../common/common.hpp"

namespace example {
    using unsigned_tuple = cute::tuple<unsigned, unsigned, unsigned>;
    using arr_tuple      = cute::tuple<cublasdx::arrangement, cublasdx::arrangement, cublasdx::arrangement>;

    template<typename ValueType>
    void reference_gemm_naive_device(unsigned_tuple const&           gemm_shape,
                                     arr_tuple const&                gemm_arr,
                                     unsigned_tuple const&           gemm_ld,
                                     ValueType const&                alpha,
                                     device_vector<ValueType> const& A,
                                     device_vector<ValueType> const& B,
                                     ValueType const&                beta,
                                     device_vector<ValueType>&       C);
} // namespace example

#endif // CUBLASDX_EXAMPLE_NAIVE_REFERENCE_HPP
