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

#ifndef CUSOLVERDX_EXAMPLE_COMMON_HPP_
#define CUSOLVERDX_EXAMPLE_COMMON_HPP_

#include <type_traits>
#include <vector>
#include <random>
#include <iostream>

#ifndef CUSOLVERDX_EXAMPLE_NVRTC
#    include <cuda/std/complex>
#    include <cusolverdx.hpp>
#endif

// the nvcc bug in CUDA 12.2-12.4, fixed in 12.5
#ifdef __NVCC__
#    if (__CUDACC_VER_MAJOR__ == 12 && (__CUDACC_VER_MINOR__ >= 2 && __CUDACC_VER_MINOR__ <= 5))
#        define CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND 1
#    endif
#endif

namespace example {
    // Used when CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND is defined
    template<typename T>
    using a_data_type_t = typename T::a_data_type;

    template<typename T>
    using a_cuda_data_type_t = typename T::a_cuda_data_type;

    template<typename T>
    using a_precision_type_t = typename T::a_precision;

} // namespace example

#endif
