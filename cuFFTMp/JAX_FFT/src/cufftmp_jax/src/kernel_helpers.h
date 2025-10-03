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

#ifndef _CUFFTMP_JAX_KERNEL_HELPERS_H_
#define _CUFFTMP_JAX_KERNEL_HELPERS_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

/**
 * Boilerplate to copy descriptors from Python to C++
 */

namespace cufftmp_jax {

// See https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && 
                        std::is_trivially_copyable<From>::value &&
                        std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
    static_assert(
        std::is_trivially_constructible<To>::value,
        "This implementation additionally requires destination type to be trivially constructible"
    );

    To dst;
    memcpy(&dst, &src, sizeof(To));
    return dst;
}

template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
    return std::string(bit_cast<const char*>(&descriptor), sizeof(T));
}

template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
    if (opaque_len != sizeof(T)) {
        throw std::runtime_error("Invalid opaque object size with opaque_len = " + std::to_string(opaque_len) + ", and sizeof(T) = " + std::to_string(sizeof(T)));
    }
    return bit_cast<const T*>(opaque);
}

}  // namespace cufftmp_jax

#endif