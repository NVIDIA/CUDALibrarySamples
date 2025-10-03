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

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include "block_fft_performance.hpp"

template<unsigned int Arch>
void block_fft_performance() {
    using namespace cufftdx;

    using fft_base = decltype(Block() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                              Precision<float>() + SM<Arch>());

    static constexpr unsigned int elements_per_thread = 8;
    static constexpr unsigned int fft_size            = 512;
    static constexpr unsigned int ffts_per_block      = 1;

    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))
    benchmark_block_fft<fft_base, fft_size, elements_per_thread, ffts_per_block>(stream, true);
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
}

template<unsigned int Arch>
struct block_fft_performance_functor {
    void operator()() { return block_fft_performance<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<block_fft_performance_functor>();
}