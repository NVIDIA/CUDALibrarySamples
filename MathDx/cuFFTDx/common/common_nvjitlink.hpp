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


#ifndef CUFFTDX_EXAMPLE_COMMON_NVJITLINK_HPP
#define CUFFTDX_EXAMPLE_COMMON_NVJITLINK_HPP

#include "common_nvrtc.hpp"

#define NVJITLINK_SAFE_CALL(h,x)                                 \
    do {                                                         \
        nvJitLinkResult result = x;                              \
        if (result != NVJITLINK_SUCCESS) {                       \
            std::cerr << "\nerror: " #x " failed with error "    \
                        << result << '\n';                       \
            size_t lsize;                                        \
            result = nvJitLinkGetErrorLogSize(h, &lsize);        \
            if (result == NVJITLINK_SUCCESS && lsize > 0) {      \
                std::vector<char> log(lsize);                    \
                result = nvJitLinkGetErrorLog(h, log.data());    \
                if (result == NVJITLINK_SUCCESS) {               \
                    std::cerr << "error: " << log.data() << '\n';\
                }                                                \
            }                                                    \
            exit(1);                                             \
        }                                                        \
    } while(0)

namespace example {
    namespace nvjitlink {
        inline std::string get_device_architecture_option(int device) {
            std::string gpu_architecture_option = "-arch=sm_" + std::to_string(nvrtc::get_device_architecture(device));
            return gpu_architecture_option;
        }
    }
}
#endif // CUFFTDX_EXAMPLE_COMMON_NVJITLINK_HPP