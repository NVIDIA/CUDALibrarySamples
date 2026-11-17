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

#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

// To be able to compile on nvcc <= 13.1.0
#define CUFFTDX_IGNORE_BLOCK_DIM_UNSUPPORTED

#include <cufftdx.hpp>

#include "../common/block_io.hpp"
#include "../common/common.hpp"

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ThreadLinearization allows an arbitrary 3D block layout. Threads are mapped to FFT
    // elements via a linearized thread index: linear_tid = x + Dx*(y + Dy*z).
    // FFT::working_group::batchid() = linear_tid / stride
    // FFT::working_group::fftid()   = linear_tid % stride
    const unsigned int batchid = FFT::working_group::batchid();

    // Load data from global memory to registers
    example::io<FFT>::load(data, thread_data, batchid);

    // Execute FFT
    extern __shared__ __align__(alignof(float4)) cufftdx::byte shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, data, batchid);
}

// In this example a one-dimensional complex-to-complex transform is performed by a CUDA block
// using ThreadLinearization with a 3D block layout.
//
// ThreadLinearization decouples the block thread layout from the FFT's logical thread mapping.
// Instead of the standard dim3(stride, ffts_per_block, 1) layout where x=fft element and
// y=batch, any block shape with the same total thread count can be used.
//
// Here: Size<128>, ElementsPerThread<8> -> stride=16, FFTsPerBlock<2> -> total=16*2=32 threads.
// BlockDim<8,4,2> gives 8*4*2=64 threads spread across all 3 dimensions.
//
// One block is run, computing two 128-point C2C float precision FFTs.
template<unsigned int Arch>
int simple_block_fft_tl() {
    using namespace cufftdx;

    // FFT with ThreadLinearization and a 3D block layout.
    using FFT = decltype(Block() + Size<128>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() +
                         Precision<float>() + ElementsPerThread<8>() + FFTsPerBlock<2>() +
                         BlockDim<8, 4, 2>() + experimental::ThreadLinearization() + SM<Arch>());
    using complex_type = typename FFT::value_type;

    // Allocate managed memory for input/output
    complex_type* data;
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));
    for (size_t i = 0; i < size; i++) {
        data[i] = complex_type {float(i), -float(i)};
    }

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < FFT::input_length; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    // Increase max shared memory if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < FFT::output_length; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(data));
    std::cout << "Success" << std::endl;
    return 0;
}

template<unsigned int Arch>
struct simple_block_fft_tl_functor {
    int operator()() { return simple_block_fft_tl<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_block_fft_tl_functor>();
}
