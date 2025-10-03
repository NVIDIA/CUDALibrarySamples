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
#include "03_block_fft_performance_block_fft_lto_ptx_performance_artifacts/lto_database.hpp.inc" // cuFFT-dumped LTOIR database header file


template<unsigned int      Arch,
         unsigned int      FFTSize,
         cufftdx::fft_type FFTType,
         class PrecisionType,
         cufftdx::experimental::code_type Code,
         cufftdx::fft_direction FFTDirection      = cufftdx::fft_direction::forward,
         bool                   UseSuggested      = true,
         unsigned int           ElementsPerThread = 8,
         unsigned int           FFTsPerBlock      = 1>
void block_fft_code_performance(const cudaStream_t& stream, bool verbose) {
    using namespace cufftdx;

    using FFT_base = decltype(Block() + Type<FFTType>() + Precision<PrecisionType>() + SM<Arch>() + experimental::CodeType<Code>());

    using FFT_with_direction =
        std::conditional_t<FFTType == fft_type::c2c, decltype(FFT_base() + Direction<FFTDirection>()), FFT_base>;

    benchmark_block_fft<FFT_with_direction, FFTSize, ElementsPerThread, FFTsPerBlock, UseSuggested>(stream, verbose);

    if (verbose)
        std::cout << std::endl;
}

template<unsigned int Arch>
struct block_fft_code_performance_functor {
    void operator()() {
        using namespace cufftdx;

        cudaStream_t stream;
        CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))

        bool default_verbose = false;

        std::cout << "cuFFTDx + PTX code type performance:" << "\n";
        block_fft_code_performance<Arch, 16   , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 32   , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 64   , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 128  , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 256  , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 512  , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 1024 , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 2048 , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 4096 , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 8192 , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 16384, fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 544  , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 608  , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 675  , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 686  , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);
        block_fft_code_performance<Arch, 800  , fft_type::c2c, float, cufftdx::experimental::code_type::ptx>(stream, default_verbose);

        std::cout << "cuFFTDx + LTOIR code type performance:" << "\n";
        block_fft_code_performance<Arch, 16   , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 32   , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 64   , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 128  , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 256  , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 512  , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 1024 , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 2048 , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 4096 , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 8192 , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 16384, fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 544  , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 608  , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 675  , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 686  , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);
        block_fft_code_performance<Arch, 800  , fft_type::c2c, float, cufftdx::experimental::code_type::ltoir>(stream, default_verbose);

        CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));
    }
};

int main(int, char**) {
    return example::sm_runner<block_fft_code_performance_functor>();
}