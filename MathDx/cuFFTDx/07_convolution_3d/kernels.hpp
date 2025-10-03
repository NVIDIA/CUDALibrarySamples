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

#pragma once

#include <type_traits>

#include <cufftdx.hpp>

#include "../common/block_io.hpp"
#include "io_strided_conv_smem.hpp"

namespace example {

    template<class FFT, class IO, class LoadOp, class StoreOp, typename InputType, typename OutputType = InputType>
    __launch_bounds__(FFT::max_threads_per_block)
        __global__ void fft_kernel(int                          subbatches,
                                   InputType*                   input,
                                   OutputType*                  output,
                                   typename FFT::workspace_type workspace) {
        using complex_type = typename FFT::value_type;
        IO io;

        if (threadIdx.y + blockIdx.x * FFT::ffts_per_block >= subbatches) {
            return;
        }

        // Local array for thread
        complex_type thread_data[FFT::storage_size];

        // Shared for staging transfers
        extern __shared__ __align__(16) complex_type shared_mem[];

        // Load data from global memory to registers
        io.load_gmem_to_rmem(input, shared_mem, thread_data, LoadOp {});

        __syncthreads();

        // Execute FFT
        FFT().execute(thread_data, shared_mem, workspace);

        __syncthreads();

        // Store data from registers to global memory
        io.store_rmem_to_gmem(thread_data, shared_mem, output, StoreOp {});
    }

    template<class FFT, class IFFT, class FilterFunctor, class IOFront, class IOBack = IOFront>
    __launch_bounds__(FFT::max_threads_per_block)
        __global__ void convolution_kernel(int                          subbatches,
                                           typename FFT::input_type*    input,
                                           typename IFFT::output_type*  output,
                                           typename FFT::workspace_type workspace) {
        using complex_type = typename FFT::value_type;
        FilterFunctor filter;
        IOFront       io_front;
        IOBack        io_back;

        if (threadIdx.y + blockIdx.x * FFT::ffts_per_block >= subbatches) {
            return;
        }

        // Local array for thread
        complex_type thread_data[FFT::storage_size];

        // Shared for staging transfers
        extern __shared__ __align__(16) complex_type shared_mem[];

        // Load data from global memory to registers
        io_front.load_gmem_to_rmem(input, shared_mem, thread_data);

        __syncthreads();

        // Execute FFT
        FFT().execute(thread_data, shared_mem, workspace);

        __syncthreads();

        // This may be different depending on type of convolution.
        // It will be complex for C2C and R2C, but real for C2R.
        using filter_data_t = typename FFT::output_type;
#pragma unroll
        for (int i = 0; i < FFT::output_ept; ++i) {
            reinterpret_cast<filter_data_t*>(thread_data)[i] = filter(reinterpret_cast<filter_data_t*>(thread_data)[i]);
        }

        IFFT().execute(thread_data, shared_mem, workspace);

        __syncthreads();

        // Store data from registers to global memory
        io_back.store_rmem_to_gmem(thread_data, shared_mem, output);
    }
} // namespace example