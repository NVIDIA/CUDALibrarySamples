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

#ifndef LIBMATHDX_EXAMPLES_COMMON_ARCH_HPP
#define LIBMATHDX_EXAMPLES_COMMON_ARCH_HPP

#include <cuda_runtime.h>
#include <libcommondx.h>
#include <nvrtc.h>

#include <array>
#include <string>
#include <vector>

#include "macros.hpp"

namespace examples {

inline std::string modifier_to_cpp(commondxArchModifier_t modifier) {
    switch (modifier) {
        case COMMONDX_ARCH_MODIFIER_GENERIC:
            return "";
        case COMMONDX_ARCH_MODIFIER_ARCH_SPECIFIC:
            return "a";
        case COMMONDX_ARCH_MODIFIER_FAMILY_SPECIFIC:
            return "f";
    }
    ASSERT(false);
    return "UNKNOWN_COMMONDX_ARCH_MODIFIER";
}

// A (major, minor) compute capability tuple
// CC's are ordered. 8.0 < 8.6 < 9.0 < 10.0 < 12.0 ...
struct cc_t {
    const long long int major;
    const long long int minor;
    inline bool operator<(const cc_t& other) const {
        return major < other.major || (major == other.major && minor < other.minor);
    }
    inline bool operator==(const cc_t& other) const { return major == other.major && minor == other.minor; }
    inline bool operator>(const cc_t& other) const { return !(*this == other || *this < other); }
    inline bool operator>=(const cc_t& other) const { return *this == other || *this > other; }
    inline bool operator<=(const cc_t& other) const { return *this == other || *this < other; }
};

// A compute-capability + accelerated arch modifier (generic, arch specific, family specific)
struct arch_t {
    const cc_t cc;
    const commondxArchModifier_t modifier;
    arch_t(cc_t cc, commondxArchModifier_t modifier) : cc(cc), modifier(modifier) {};
    arch_t(long long int major, long long int minor) : cc { major, minor }, modifier(COMMONDX_ARCH_MODIFIER_GENERIC) {};
    arch_t(long long int major, long long int minor, commondxArchModifier_t modifier) :
        cc { major, minor }, modifier(modifier) {};
    inline std::array<long long int, 2> to_array() const {
        return std::array<long long int, 2> { this->operator_sm(), modifier };
    };
    inline long long int operator_sm() const { return 100 * cc.major + 10 * cc.minor; };
    inline std::string str() const { return std::to_string(10 * cc.major + cc.minor) + modifier_to_cpp(modifier); };
    inline bool operator==(const arch_t& other) const { return cc == other.cc && modifier == other.modifier; }
    inline bool operator!=(const arch_t& other) const { return !(*this == other); }
};

inline cc_t get_current_cc() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
    return cc_t { deviceProp.major, deviceProp.minor };
}

inline cc_t max_nvrtc_target_cc() {
    int numArchs = 0;
    NVRTC_CHECK(nvrtcGetNumSupportedArchs(&numArchs));
    ASSERT(numArchs > 0);
    std::vector<int> supportedArchs(numArchs, 0);
    NVRTC_CHECK(nvrtcGetSupportedArchs(supportedArchs.data()));
    return cc_t { supportedArchs.back() / 10, supportedArchs.back() % 10 };
}

// TL;DR: we should never compile for something NVRTC can't target, and we should always cap to the max NVRTC arch
inline cc_t get_target_cc() {
    auto current = get_current_cc();
    auto max = max_nvrtc_target_cc();
    if (current > max) {
        return max;
    } else {
        return current;
    }
}

inline arch_t get_target_sm() {
    return { get_target_cc(), COMMONDX_ARCH_MODIFIER_GENERIC };
}

// TL;DR: We should always use SM<...> for an SM which is less than what we compile to
inline cc_t get_dx_cc() {
    auto target = get_target_cc();
    auto max = cc_t { 12, 0 };
    if (target > max) {
        return max;
    } else {
        return target;
    }
}

inline arch_t get_dx_sm() {
    return { get_dx_cc(), COMMONDX_ARCH_MODIFIER_GENERIC };
}

inline arch_t maybe_accelerated_dx(cc_t in) {
    if (in == get_current_cc() && (in == cc_t { 9, 0 } || in == cc_t { 10, 0 } || in == cc_t { 10, 1 } ||
                                   in == cc_t { 10, 3 } || in == cc_t { 11, 0 })) {
        return { in, COMMONDX_ARCH_MODIFIER_ARCH_SPECIFIC };
    } else {
        return { in, COMMONDX_ARCH_MODIFIER_GENERIC };
    }
}

inline arch_t maybe_accelerated_target(cc_t in) {
    if (in == get_current_cc() &&
        (in == cc_t { 9, 0 } || in == cc_t { 10, 0 } || in == cc_t { 10, 1 } || in == cc_t { 10, 3 } ||
         in == cc_t { 11, 0 } || in == cc_t { 12, 0 } || in == cc_t { 12, 1 })) {
        return { in, COMMONDX_ARCH_MODIFIER_ARCH_SPECIFIC };
    } else {
        return { in, COMMONDX_ARCH_MODIFIER_GENERIC };
    }
}

} // namespace examples

#endif // LIBMATHDX_EXAMPLES_COMMON_ARCH_HPP
