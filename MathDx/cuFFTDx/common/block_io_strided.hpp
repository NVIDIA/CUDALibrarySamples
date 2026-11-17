/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP
#define CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP

#include "block_io.hpp"
#include "mixed_io.hpp"

namespace example {
    template<class FFT>
    struct io_strided: public io<FFT> {
        using base_type = io<FFT>;

        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static inline __device__ unsigned int global_batchid(unsigned int local_batchid) {
            unsigned int global_fft_id = blockIdx.x * FFT::ffts_per_block + local_batchid;
            return global_fft_id;
        }

        static inline __device__ unsigned int batch_offset_strided(unsigned int local_batchid) {
            return global_batchid(local_batchid);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads / stores for 2d FFTs, based on stride and batches for specific dimension (inner most)
        template<unsigned int Stride, unsigned int Batches = Stride, typename InputOutputType>
        static inline __device__ void load_strided(const InputOutputType* input,
                                                   complex_type*          thread_data,
                                                   unsigned int           local_batchid) {
            // Calculate global offset of FFT batch
            const unsigned int batch_offset = batch_offset_strided(local_batchid);
            const unsigned int bid          = global_batchid(local_batchid);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride       = Stride * FFT::stride;
            const unsigned int fftid        = FFT::working_group::fftid();
            unsigned int       index        = batch_offset + (fftid * Stride);
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + fftid) < cufftdx::size_of<FFT>::value) {
                    if (bid < Batches) {
                        thread_data[i] = convert<complex_type>(input[index]);
                    }
                    index += stride;
                }
            }
        }

        template<unsigned int Stride, unsigned int Batches = Stride, typename InputOutputType>
        static inline __device__ void store_strided(const complex_type* thread_data,
                                                    InputOutputType*    output,
                                                    unsigned int        local_batchid) {
            const unsigned int batch_offset = batch_offset_strided(local_batchid);
            const unsigned int bid          = global_batchid(local_batchid);
            const unsigned int stride       = Stride * FFT::stride;
            const unsigned int fftid        = FFT::working_group::fftid();
            unsigned int       index        = batch_offset + (fftid * Stride);
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + fftid) < cufftdx::size_of<FFT>::value) {
                    if (bid < Batches) {
                        output[index] = convert<InputOutputType>(thread_data[i]);
                    }
                    index += stride;
                }
            }
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads and stores for 2d FFTs with shared memory used, based on stride and batches for specific dimension (inner most)
        template<unsigned int Stride, unsigned int Batches = Stride, typename InputOutputType>
        static inline __device__ void load_strided(const InputOutputType* input,
                                                   complex_type*          thread_data,
                                                   InputOutputType*       shared_memory,
                                                   unsigned int           local_batchid) {

            const unsigned int bpb     = FFT::working_group::implicit_bpb();
            const unsigned int batchid = FFT::working_group::batchid();
            const unsigned int fftid   = FFT::working_group::fftid();

            const unsigned int tid          = fftid + FFT::stride * batchid;
            const unsigned int tidx         = tid / bpb;
            const unsigned int tidy         = tid % bpb;
            // Calculate global offset of FFT batch
            const unsigned int batch_offset = batch_offset_strided(tidy);
            const unsigned int bid          = global_batchid(tidy);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride       = Stride * FFT::stride;
            unsigned int       index        = batch_offset + (tidx * Stride);
            unsigned int       smem_index   = tidx + tidy * FFT::stride;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
                    if (bid < Batches) {
                        shared_memory[smem_index] = input[index];
                    }
                    index += stride;
                    smem_index += (FFT::stride * bpb);
                }
            }
            __syncthreads();
            smem_index = fftid + batchid * FFT::stride;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + fftid) < cufftdx::size_of<FFT>::value) {
                    thread_data[i] = convert<complex_type>(shared_memory[smem_index]);
                    smem_index += (FFT::stride * bpb);
                }
            }
        }

        template<unsigned int Stride, unsigned int Batches = Stride, typename InputOutputType>
        static inline __device__ void store_strided(const complex_type* thread_data,
                                                    InputOutputType*    shared_memory,
                                                    InputOutputType*    output,
                                                    unsigned int        local_batchid) {
            __syncthreads();

            const unsigned int bpb     = FFT::working_group::implicit_bpb();
            const unsigned int batchid = FFT::working_group::batchid();
            const unsigned int fftid   = FFT::working_group::fftid();

            unsigned int smem_index = fftid + batchid * FFT::stride;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + fftid) < cufftdx::size_of<FFT>::value) {
                    shared_memory[smem_index] = convert<InputOutputType>(thread_data[i]);
                    smem_index += (FFT::stride * bpb);
                }
            }
            __syncthreads();
            const unsigned int tid          = fftid + FFT::stride * batchid;
            const unsigned int tidx         = tid / bpb;
            const unsigned int tidy         = tid % bpb;
            const unsigned int batch_offset = batch_offset_strided(tidy);
            const unsigned int bid          = global_batchid(tidy);
            const unsigned int stride       = Stride * FFT::stride;
            unsigned int       index        = batch_offset + (tidx * Stride);
            smem_index                      = tidx + tidy * FFT::stride;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
                    if (bid < Batches) {
                        output[index] = shared_memory[smem_index];
                    }
                    index += stride;
                    smem_index += (FFT::stride * bpb);
                }
            }
        }
    };
} // namespace example

#endif // CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP
