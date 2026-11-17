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

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>

#include "../common/block_io.hpp"
#include "../common/common.hpp"

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_thread_layout_kernel(typename FFT::value_type* data) {
    CUFFTDX_SKIP_IF_NOT_APPLICABLE_SM_BLOCK(FFT) {

        using complex_type = typename FFT::value_type;

        complex_type thread_data[FFT::storage_size] = {};

        // ThreadLayout<transposed> makes threadIdx.x select the FFT batch and
        // threadIdx.y select the FFT thread. example::io uses the logical IDs, so
        // its input/output layout remains batch-major. It is ThreadLayout agnostic;
        // see block_io.hpp for the logical-ID mapping.
        const unsigned int batch_id = FFT::working_group::batchid();
        example::io<FFT>::load(data, thread_data, batch_id);

        extern __shared__ __align__(alignof(float4)) cufftdx::byte shared_mem[];
        FFT().execute(thread_data, shared_mem);

        example::io<FFT>::store(thread_data, data, batch_id);
    }
}

// This example performs two 128-point C2C FFTs with the register API and a
// transposed thread layout. Thread ownership is transposed, while the input
// and output arrays remain batch-major: data[batch * fft_size + element].
template<unsigned int Arch>
int simple_block_fft_thread_layout() {
    using namespace cufftdx;

    using FFT          = decltype(Block() + Size<128>() + Type<fft_type::c2c>() +
                         Direction<fft_direction::forward>() + Precision<float>() +
                         ElementsPerThread<8>() + FFTsPerBlock<2>() +
                         experimental::ThreadLayout<experimental::thread_layout::transposed>() + SM<Arch>());
    using complex_type = typename FFT::value_type;

    constexpr unsigned int size = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    complex_type*          data = nullptr;
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size * sizeof(complex_type)));

    for (unsigned int i = 0; i < size; ++i) {
        data[i] = complex_type {float(i), -float(i)};
    }

    std::cout << "input [1st FFT]:\n";
    for (unsigned int i = 0; i < FFT::input_length; ++i) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(block_fft_thread_layout_kernel<FFT>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             FFT::shared_memory_size));
    block_fft_thread_layout_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "output [1st FFT]:\n";
    for (unsigned int i = 0; i < FFT::output_length; ++i) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(data));
    std::cout << "Success" << std::endl;
    return 0;
}

template<unsigned int Arch>
struct simple_block_fft_thread_layout_functor {
    int operator()() { return simple_block_fft_thread_layout<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_block_fft_thread_layout_functor>();
}
