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

#ifndef CUFFTDX_EXAMPLE_BLOCK_IO_HPP_
#define CUFFTDX_EXAMPLE_BLOCK_IO_HPP_

#include <type_traits>

#include "common.hpp"
#include "fp16_common.hpp"

namespace example {
    namespace __io {
        template<bool InRRIILayout = false>
        inline __device__ cufftdx::complex<__half2> convert_to_rrii(const cufftdx::complex<__half2>& value) {
            return to_rrii(value);
        }
        template<>
        inline __device__ cufftdx::complex<__half2> convert_to_rrii<true>(const cufftdx::complex<__half2>& value) {
            return value;
        }
        template<bool InRIRILayout = false>
        inline __device__ cufftdx::complex<__half2> convert_to_riri(const cufftdx::complex<__half2>& value) {
            return to_riri(value);
        }
        template<>
        inline __device__ cufftdx::complex<__half2> convert_to_riri<true>(const cufftdx::complex<__half2>& value) {
            return value;
        }
    } // namespace __io

    template<class FFT>
    struct io {
        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static constexpr bool this_fft_is_folded =
            cufftdx::real_fft_mode_of<FFT>::value == cufftdx::real_mode::folded;

        template<typename RegType, typename MemType>
        static constexpr bool is_type_compatible() {
            return !CUFFTDX_STD::is_void_v<RegType> && (sizeof(RegType) == sizeof(complex_type)) &&
                   (alignof(RegType) == alignof(complex_type));
        }

        static inline __device__ unsigned int input_batch_offset(unsigned int local_fft_id) {
            // Implicit batching is currently mandatory for __half precision, and it forces two
            // batches of data to be put together into a single complex __half2 value. This makes
            // it so a "single" batch of complex __half2 values in reality contains 2 batches of
            // complex __half values. Full reference can be found in documentation:
            // https://docs.nvidia.com/cuda/cufftdx/api/methods.html#half-precision-implicit-batching
            unsigned int global_fft_id =
                blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
            return FFT::input_length * global_fft_id;
        }

        static inline __device__ unsigned int output_batch_offset(unsigned int local_fft_id) {
            // See note regarding implicit batching in input_batch_offset
            unsigned int global_fft_id =
                blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
            // If Fold Optimization is enabled, the real values are packed together
            // into complex values which decreases the effective size twofold
            return FFT::output_length * global_fft_id;
        }

        template<unsigned int EPT, typename DataType>
        static inline __device__ void copy(const DataType* source, DataType* target, unsigned int n) {
            unsigned int stride = blockDim.x * blockDim.y;
            unsigned int index  = threadIdx.y * blockDim.x + threadIdx.x;
            for (int i = 0; i < EPT; i++) {
                if (index < n) {
                    target[index] = source[index];
                }
                index += stride;
            }
        }

        template<class DataType>
        static inline __device__ void load_to_smem(const DataType* global, unsigned char* shared) {
            using input_t = typename FFT::input_type;
            copy<FFT::input_ept>(reinterpret_cast<const input_t*>(global),
                                reinterpret_cast<input_t*>(shared),
                                blockDim.y * FFT::input_length);
            __syncthreads();
        }

        template<class DataType>
        static inline __device__ void store_from_smem(const unsigned char* shared, DataType* global) {
            __syncthreads();
            using output_t = typename FFT::output_type;
            copy<FFT::output_ept>(reinterpret_cast<const output_t*>(shared),
                                 reinterpret_cast<output_t*>(global),
                                 blockDim.y * FFT::output_length);
        }


        // If InputInRRIILayout is false, then function assumes that values in input are in RIRI
        // layout, and before loading them to thread_data they are converted to RRII layout.
        // Otherwise, if InputInRRIILayout is true, then function assumes values in input are in RRII
        // layout, and don't need to be converted before loading to thread_data.
        template<bool InputInRRIILayout = false, typename RegisterType, typename IOType, class LoadOp = example::identity>
        static inline __device__
            CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
            load(const IOType* input,
                 RegisterType* thread_data,
                 unsigned int  local_fft_id,
                 LoadOp op = {}) {
            static constexpr bool needs_half2_format_conversion = cufftdx::type_of<FFT>::value != cufftdx::fft_type::r2c &&
                                                                  std::is_same_v<IOType, cufftdx::detail::complex<__half2>>;
            using input_t                                       = typename FFT::input_type;

            // Calculate global offset of FFT batch
            const unsigned int offset = input_batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = FFT::stride;
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::input_ept; i++) {
                if ((i * stride + threadIdx.x) < FFT::input_length) {
                    if constexpr (needs_half2_format_conversion) {
                        reinterpret_cast<input_t*>(thread_data)[i] =
                            op(__io::convert_to_rrii<InputInRRIILayout>(reinterpret_cast<const input_t*>(input)[index]));
                    } else {
                        reinterpret_cast<input_t*>(thread_data)[i] = op(reinterpret_cast<const input_t*>(input)[index]);
                    }
                    index += stride;
                }
            }
        }

        // Function assumes that values in thread_data are in RRII layout.
        // If OutputInRRIILayout is false, values are saved into output in RIRI layout; otherwise - in RRII.
        template<bool OutputInRRIILayout = false, typename RegisterType, typename IOType, class StoreOp = example::identity>
        static inline __device__
            CUFFTDX_STD::enable_if_t<is_type_compatible<RegisterType, IOType>()>
            store(const RegisterType* thread_data,
                  IOType*             output,
                  unsigned int        local_fft_id,
                  StoreOp op = {}) {
            static constexpr bool needs_half2_format_conversion = cufftdx::type_of<FFT>::value != cufftdx::fft_type::c2r &&
                                                                  std::is_same_v<IOType, cufftdx::detail::complex<__half2>>;
            using output_t                                       = typename FFT::output_type;

            const unsigned int offset = output_batch_offset(local_fft_id);
            const unsigned int stride = FFT::stride;
            unsigned int       index  = offset + threadIdx.x;

            for (int i = 0; i < FFT::output_ept; ++i) {
                if ((i * stride + threadIdx.x) < FFT::output_length) {
                    if constexpr (needs_half2_format_conversion) {
                        reinterpret_cast<output_t*>(output)[index] =
                            op(__io::convert_to_riri<OutputInRRIILayout>(reinterpret_cast<const output_t*>(thread_data)[i]));
                    } else {
                        reinterpret_cast<output_t*>(output)[index] = op(reinterpret_cast<const output_t*>(thread_data)[i]);
                    }
                    index += stride;
                }
            }
        }
    };

    template<class FFT>
    struct io_fp16 {
        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static_assert(std::is_same<scalar_type, __half2>::value, "This IO class is only for half precision FFTs");
        static_assert((FFT::ffts_per_block % 2 == 0), "This IO class works only for even FFT::ffts_per_block");

        static inline __device__ unsigned int batch_offset(unsigned int local_fft_id) {
            unsigned int global_fft_id =
                FFT::ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
            return cufftdx::size_of<FFT>::value * global_fft_id;
        }

        static inline __device__ void load(const __half2* input, complex_type* thread_data, unsigned int local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = FFT::stride;
            unsigned int       index  = offset + threadIdx.x;
            // We bundle __half2 data of X-th and X+(FFT::ffts_per_block/2) batches together.
            const unsigned int batch_stride = (FFT::ffts_per_block / 2) * cufftdx::size_of<FFT>::value;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                thread_data[i] = to_rrii(input[index], input[index + batch_stride]);
                index += stride;
            }
        }

        static inline __device__ void store(const complex_type* thread_data,
                                            __half2*            output,
                                            unsigned int        local_fft_id) {
            const unsigned int offset       = batch_offset(local_fft_id);
            const unsigned int stride       = FFT::stride;
            unsigned int       index        = offset + threadIdx.x;
            const unsigned int batch_stride = (FFT::ffts_per_block / 2) * cufftdx::size_of<FFT>::value;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                output[index]                = to_ri1(thread_data[i]);
                output[index + batch_stride] = to_ri2(thread_data[i]);
                index += stride;
            }
        }
    };
} // namespace example

#endif // CUFFTDX_EXAMPLE_BLOCK_IO_HPP_