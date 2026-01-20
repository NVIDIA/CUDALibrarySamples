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

#ifndef CUSOLVERDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP

#include "macros.hpp"
#include <cusolverdx.hpp>
#include <cstdio>

namespace common {
    // This function enables creating architecture agnostic examples
    // and functions while avoid compilation overhead, by compiling
    // only the enabled branches and then based on runtime CUDA compute
    // capability dispatching with appropriate argument.
    //
    // Functor is example function which takes static integer type as
    // its argument. Then the example can read this value and use it
    // for its SM<Val>() operator.
    template<template<int> class Functor>
    inline int run_example_with_sm() {
        // Get CUDA device compute capability
        const unsigned cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
// All SM supported by cuSOLVERDx
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_70
            case 700: return Functor<700>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_72
            case 720: return Functor<720>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_75
            case 750: return Functor<750>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_80
            case 800: return Functor<800>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_86
            case 860: return Functor<860>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_87
            case 870: return Functor<870>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_89
            case 890: return Functor<890>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_90
            case 900: return Functor<900>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_100
            case 1000: return Functor<1000>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_101
            case 1010: return Functor<1010>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_120
            case 1200: return Functor<1200>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_110
            case 1100: return Functor<1100>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_103
            case 1030: return Functor<1030>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_121
            case 1210: return Functor<1210>()();
#endif

            default: {
                printf("Examples not configured to support SM %u.  Use the CUSOLVERDX_CUDA_ARCHITECTURES CMake variable to configure the SM support.\n",
                       cuda_device_arch);
                // Fail
                return 1;
            }
        }
    }
} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP
