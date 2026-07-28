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

// Enable runtime database and get_all_implementations API
#define CUFFTDX_ENABLE_RUNTIME_DATABASE

#include <cufftdx.hpp>
#include "lto_database.hpp.inc"

#include "../common/common.hpp"

template<unsigned int Arch>
int lto_query_database_helper() {
    static constexpr unsigned int fft_size = 128;
    constexpr unsigned int arch = static_cast<unsigned int>(Arch);

    // Query runtime database for all LTO (ltoir) implementations for this size/arch
    auto all_implementations = cufftdx::experimental::utils::get_all_implementations(fft_size,
                                                                                     cufftdx::fft_direction::forward,
                                                                                     cufftdx::fft_type::c2c,
                                                                                     arch,
                                                                                     cufftdx::utils::execution_type::block,
                                                                                     cufftdx::precision::f32,
                                                                                     0,
                                                                                     0,
                                                                                     {0, 0, 0},
                                                                                     cufftdx::complex_layout::natural,
                                                                                     cufftdx::real_mode::normal,
                                                                                     cufftdx::experimental::query_code_type::ltoir_offline);

    if (all_implementations.empty()) {
        std::cout << "No offline LTO implementations found for size=" << fft_size << " arch=" << arch
                  << "; empty query result is expected when this cuFFT database does not support offline LTO query." << std::endl;
        return 0;
    }

    std::cout << "get_all_implementations(ltoir): size=" << fft_size << " arch=" << arch
              << " found " << all_implementations.size() << " implementation(s):" << std::endl;

    for (unsigned int i = 0; i < all_implementations.size(); ++i) {
        const auto& impl = all_implementations[i];
        std::cout << "  [" << i << "]"
                  << " block_dim=(" << impl.block_dim_x << "," << impl.block_dim_y << "," << impl.block_dim_z << ")"
                  << " ept=" << impl.elements_per_thread
                  << " fpb=" << impl.ffts_per_block
                  << " shared_mem=" << impl.shared_memory_size
                  << " storage_size=" << impl.storage_size
                  << std::endl;
    }

    std::cout << "Success" << std::endl;
    return 0;
}

template<unsigned int Arch>
struct lto_query_database_helper_functor {
    int operator()() { return lto_query_database_helper<Arch>(); }
};

int main() {
    return example::sm_runner<lto_query_database_helper_functor>();
}
