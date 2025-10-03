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

#include <cuda_runtime_api.h>
#include <cufftdx.hpp>
#include "01_simple_fft_thread_simple_fft_thread_lto_artifacts/lto_database.hpp.inc" // cuFFT-dumped LTOIR database header file

#include "../common/common.hpp"

template<class FFT>
__global__ void thread_fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // Load data from global memory to registers.
    // thread_data should have all input data in order.
    unsigned int index = threadIdx.x * FFT::elements_per_thread;
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        thread_data[i] = data[index + i];
    }

    // Execute FFT
    FFT().execute(thread_data);

    // Save results
    for (size_t i = 0; i < FFT::elements_per_thread; i++) {
        data[index + i] = thread_data[i];
    }
}

// In this example a one-dimensional complex-to-complex transform is perform by a CUDA thread.
//
// Four (threads_count) threads are run, and each thread calculates 8-point (fft_size) C2C double precision FFT.
// Data is generated on host, copied to device buffer, and then results are copied back to host.
int main(int, char**) {
    using namespace cufftdx;

    // Number of threads to execute
    static constexpr unsigned int threads_count = 4;

    // FFT is defined, its: size, type, direction, precision.
    // - Thread()   operator informs that FFT will be executed on thread level.
    // - CodeType() operator informs that FFT will attempt to use cuFFT-dumped LTOIRs for computes.
    using FFT_without_code_type = decltype(Thread() + Size<8>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>() + Precision<double>());
    using FFT = decltype(FFT_without_code_type() + experimental::CodeType<experimental::code_type::ltoir>());
    using complex_type = typename FFT::value_type;

    // Check if LTO implementation is selected
    static_assert(experimental::code_type_of<FFT>::value == experimental::code_type::ltoir, "Code Type is not set to LTO-IR");
    static_assert(FFT::code == experimental::code_type::ltoir, "Selected implementation code type is not LTO-IR");

    // Host data
    std::vector<complex_type> input(FFT::input_length * threads_count);
    for (size_t i = 0; i < input.size(); i++) {
        input[i] = complex_type {double(i), -double(i)};
    }

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < FFT::input_length; i++) {
        std::cout << input[i].x << " " << input[i].y << std::endl;
    }

    // Device data
    complex_type* device_buffer;
    auto          size_bytes = input.size() * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&device_buffer, size_bytes));
    // Copy host to device
    CUDA_CHECK_AND_EXIT(cudaMemcpy(device_buffer, input.data(), size_bytes, cudaMemcpyHostToDevice));

    // Invokes kernel with 'threads_count' threads in block, each thread calculates one FFT of size
    thread_fft_kernel<FFT><<<1, threads_count>>>(device_buffer);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy device to host
    std::vector<complex_type> output(input.size());
    CUDA_CHECK_AND_EXIT(cudaMemcpy(output.data(), device_buffer, size_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaFree(device_buffer));

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < FFT::output_length; i++) {
        std::cout << output[i].x << " " << output[i].y << std::endl;
    }

    std::cout << "Success" << std::endl;
}