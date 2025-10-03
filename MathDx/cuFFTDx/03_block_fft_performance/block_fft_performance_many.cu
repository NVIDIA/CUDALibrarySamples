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

template<unsigned int      Arch,
         unsigned int      FFTSize,
         cufftdx::fft_type FFTType,
         class PrecisionType,
         cufftdx::fft_direction FFTDirection      = cufftdx::fft_direction::forward,
         bool                   UseSuggested      = true,
         unsigned int           ElementsPerThread = 8,
         unsigned int           FFTsPerBlock      = 1>
void block_fft_performance(const cudaStream_t& stream, bool verbose) {
    using namespace cufftdx;

    using FFT_base = decltype(Block() + Type<FFTType>() + Precision<PrecisionType>() + SM<Arch>());

    using FFT_with_direction =
        std::conditional_t<FFTType == fft_type::c2c, decltype(FFT_base() + Direction<FFTDirection>()), FFT_base>;

    benchmark_block_fft<FFT_with_direction, FFTSize, ElementsPerThread, FFTsPerBlock, UseSuggested>(stream, verbose);

    if (verbose)
        std::cout << std::endl;
}

template<unsigned int Arch>
struct block_fft_performance_functor {
    void operator()() {
        using namespace cufftdx;

        cudaStream_t stream;
        CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))

        bool default_verbose = false;


        // To specify EPT and FPB values, set UsedSuggested to false.
        // FFTDirection is used if and only if FFTType is C2C.
        // Below is an example of a test run with specified EPT and FPB values.

        block_fft_performance<Arch, 137, fft_type::c2c, float, fft_direction::forward, false, 8, 1>(stream,
                                                                                                    default_verbose);

        block_fft_performance<Arch, 137, fft_type::c2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 251, fft_type::c2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 512, fft_type::c2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 1024, fft_type::c2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 2048, fft_type::c2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 4096, fft_type::c2c, float>(stream, default_verbose);

        block_fft_performance<Arch, 137, fft_type::c2c, float, fft_direction::inverse>(stream, default_verbose);
        block_fft_performance<Arch, 251, fft_type::c2c, float, fft_direction::inverse>(stream, default_verbose);
        block_fft_performance<Arch, 512, fft_type::c2c, float, fft_direction::inverse>(stream, default_verbose);
        block_fft_performance<Arch, 1024, fft_type::c2c, float, fft_direction::inverse>(stream, default_verbose);
        block_fft_performance<Arch, 2048, fft_type::c2c, float, fft_direction::inverse>(stream, default_verbose);
        block_fft_performance<Arch, 4096, fft_type::c2c, float, fft_direction::inverse>(stream, default_verbose);

        block_fft_performance<Arch, 137, fft_type::r2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 251, fft_type::r2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 512, fft_type::r2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 1024, fft_type::r2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 2048, fft_type::r2c, float>(stream, default_verbose);
        block_fft_performance<Arch, 4096, fft_type::r2c, float>(stream, default_verbose);

        block_fft_performance<Arch, 137, fft_type::c2r, float>(stream, default_verbose);
        block_fft_performance<Arch, 251, fft_type::c2r, float>(stream, default_verbose);
        block_fft_performance<Arch, 512, fft_type::c2r, float>(stream, default_verbose);
        block_fft_performance<Arch, 1024, fft_type::c2r, float>(stream, default_verbose);
        block_fft_performance<Arch, 2048, fft_type::c2r, float>(stream, default_verbose);
        block_fft_performance<Arch, 4096, fft_type::c2r, float>(stream, default_verbose);

        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    }
};

int main(int, char**) {
    return example::sm_runner<block_fft_performance_functor>();
}