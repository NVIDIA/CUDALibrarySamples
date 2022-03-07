// Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
//
// NOTICE TO LICENSEE:
//
// This source code and/or documentation ("Licensed Deliverables") are subject to
// NVIDIA intellectual property rights under U.S. and international Copyright
// laws.
//
// These Licensed Deliverables contained herein is PROPRIETARY and CONFIDENTIAL
// to NVIDIA and is being provided under the terms and conditions of a form of
// NVIDIA software license agreement by and between NVIDIA and Licensee ("License
// Agreement") or electronically accepted by Licensee.  Notwithstanding any terms
// or conditions to the contrary in the License Agreement, reproduction or
// disclosure of the Licensed Deliverables to any third party without the express
// written consent of NVIDIA is prohibited.
//
// NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
// AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THESE
// LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS
// OR IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
// TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
// NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
// AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
// INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM
// LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
// OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
// PERFORMANCE OF THESE LICENSED DELIVERABLES.
//
// U.S. Government End Users.  These Licensed Deliverables are a "commercial
// item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting of
// "commercial computer software" and "commercial computer software
// documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) and is
// provided to the U.S. Government only as a commercial end item.  Consistent
// with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),
// all U.S. Government End Users acquire the Licensed Deliverables with only
// those rights set forth herein.
//
// Any use of the Licensed Deliverables in individual and commercial software
// must include, in the user documentation and internal comments to the code, the
// above Disclaimer and U.S. Government End Users Notice.

#ifndef MATHDX_CUFFTDX_EXAMPLE_BLOCK_IO_HPP_
#define MATHDX_CUFFTDX_EXAMPLE_BLOCK_IO_HPP_

namespace example {
    template<class FFT>
    struct io {
        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static inline __device__ unsigned int stride_size() {
            return FFT::stride;
        }

        static inline __device__ unsigned int batch_offset(unsigned int local_fft_id) {
            unsigned int global_fft_id =
                FFT::ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
            return cufftdx::size_of<FFT>::value * global_fft_id;
        }

        template<typename DataType>
        static inline __device__ void copy(const DataType* source, DataType* target, unsigned int n) {
            unsigned int stride = blockDim.x * blockDim.y;
            unsigned int index = threadIdx.y * blockDim.x + threadIdx.x;
            for (int step = 0; step < FFT::elements_per_thread; step++) {
                if (index < n) {
                    target[index] = source[index];
                }
                index += stride;
            }
        }

        template<class DataType>
        static inline __device__ void load_to_smem(const DataType* global, unsigned char* shared) {
            if (cufftdx::type_of<FFT>::value == cufftdx::fft_type::c2c) {
                unsigned int input_length = blockDim.y * cufftdx::size_of<FFT>::value;
                copy(reinterpret_cast<const complex_type*>(global),
                     reinterpret_cast<complex_type*>(shared),
                     input_length);
            } else if (cufftdx::type_of<FFT>::value == cufftdx::fft_type::c2r) {
                unsigned int input_length = blockDim.y * ((cufftdx::size_of<FFT>::value / 2) + 1);
                copy(reinterpret_cast<const complex_type*>(global),
                     reinterpret_cast<complex_type*>(shared),
                     input_length);
            } else if (cufftdx::type_of<FFT>::value == cufftdx::fft_type::r2c) {
                unsigned int input_length = blockDim.y * cufftdx::size_of<FFT>::value;
                copy(reinterpret_cast<const scalar_type*>(global),
                     reinterpret_cast<scalar_type*>(shared),
                     input_length);
            }
            __syncthreads();
        }

        template<class DataType>
        static inline __device__ void store_from_smem(const unsigned char* shared, DataType* global) {
            __syncthreads();
            if (cufftdx::type_of<FFT>::value == cufftdx::fft_type::c2c) {
                unsigned int output_length = blockDim.y * cufftdx::size_of<FFT>::value;
                copy(reinterpret_cast<const complex_type*>(shared),
                     reinterpret_cast<complex_type*>(global),
                     output_length);
            } else if (cufftdx::type_of<FFT>::value == cufftdx::fft_type::c2r) {
                unsigned int output_length = blockDim.y * cufftdx::size_of<FFT>::value;
                copy(reinterpret_cast<const scalar_type*>(shared),
                     reinterpret_cast<scalar_type*>(global),
                     output_length);
            } else if (cufftdx::type_of<FFT>::value == cufftdx::fft_type::r2c) {
                unsigned int output_length = blockDim.y * ((cufftdx::size_of<FFT>::value / 2) + 1);
                copy(reinterpret_cast<const complex_type*>(shared),
                     reinterpret_cast<complex_type*>(global),
                     output_length);
            }
        }

        template<cufftdx::fft_type FFTType = cufftdx::type_of<FFT>::value, class ComplexType = complex_type>
        static inline __device__ auto load(const void*        input,
                                           ComplexType*      thread_data,
                                           const unsigned int local_fft_id) ->
            typename std::enable_if<FFTType == cufftdx::fft_type::c2c>::type {
            return load_c2c<ComplexType>((ComplexType*)input, thread_data, local_fft_id);
        }

        template<cufftdx::fft_type FFTType = cufftdx::type_of<FFT>::value, class ComplexType = complex_type>
        static inline __device__ auto load(const void*        input,
                                           ComplexType*      thread_data,
                                           const unsigned int local_fft_id) ->
            typename std::enable_if<FFTType == cufftdx::fft_type::c2r>::type {
            return load_c2r<ComplexType>((ComplexType*)input, thread_data, local_fft_id);
        }

        template<cufftdx::fft_type FFTType = cufftdx::type_of<FFT>::value, class ComplexType = complex_type>
        static inline __device__ auto load(const void*        input,
                                           ComplexType*      thread_data,
                                           const unsigned int local_fft_id) ->
            typename std::enable_if<FFTType == cufftdx::fft_type::r2c>::type {
            return load_r2c<ComplexType>((scalar_type*)input, thread_data, local_fft_id);
        }

        template<cufftdx::fft_type FFTType = cufftdx::type_of<FFT>::value, class ComplexType = complex_type>
        static inline __device__ auto store(const ComplexType* thread_data,
                                            void*               output,
                                            const unsigned int  local_fft_id) ->
            typename std::enable_if<FFTType == cufftdx::fft_type::c2c>::type {
            return store_c2c<ComplexType>(thread_data, (ComplexType*)output, local_fft_id);
        }

        template<cufftdx::fft_type FFTType = cufftdx::type_of<FFT>::value, class ComplexType = complex_type>
        static inline __device__ auto store(const ComplexType* thread_data,
                                            void*               output,
                                            const unsigned int  local_fft_id) ->
            typename std::enable_if<FFTType == cufftdx::fft_type::c2r>::type {
            return store_c2r<ComplexType>(thread_data, (scalar_type*)output, local_fft_id);
        }

        template<cufftdx::fft_type FFTType = cufftdx::type_of<FFT>::value, class ComplexType = complex_type>
        static inline __device__ auto store(const ComplexType* thread_data,
                                            void*               output,
                                            const unsigned int  local_fft_id) ->
            typename std::enable_if<FFTType == cufftdx::fft_type::r2c>::type {
            return store_r2c<ComplexType>(thread_data, (ComplexType*)output, local_fft_id);
        }

        // input - global input with all FFTs
        // thread_data - local thread array to load values from input to
        // local_fft_id - ID of FFT batch in CUDA block
        template<class ComplexType = complex_type>
        static inline __device__ void load_c2c(const ComplexType* input,
                                           ComplexType*        thread_data,
                                           unsigned int        local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                thread_data[i] = input[index];
                index += stride;
            }
        }

        template<class ComplexType = complex_type>
        static inline __device__ void store_c2c(const ComplexType* thread_data,
                                            ComplexType*       output,
                                            unsigned int        local_fft_id) {
            const unsigned int offset = batch_offset(local_fft_id);
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                output[index] = thread_data[i];
                index += stride;
            }
        }

        static inline __device__ unsigned int batch_offset_r2c(unsigned int local_fft_id) {
            unsigned int global_fft_id =
                FFT::ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
            return ((cufftdx::size_of<FFT>::value / 2) + 1) * global_fft_id;
        }

        template<class ComplexType = complex_type>
        static inline __device__ void load_r2c(const scalar_type* input,
                                               ComplexType*       thread_data,
                                               unsigned int       local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                reinterpret_cast<scalar_type*>(thread_data)[i] = input[index];
                index += stride;
            }
        }

        template<class ComplexType = complex_type>
        static inline __device__ void store_r2c(const ComplexType* thread_data,
                                                ComplexType*       output,
                                                unsigned int       local_fft_id) {
            const unsigned int offset = batch_offset_r2c(local_fft_id);
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) {
                output[index] = thread_data[i];
                index += stride;
            }
            constexpr unsigned int threads_per_fft        = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
            constexpr unsigned int output_values_to_store = (cufftdx::size_of<FFT>::value / 2) + 1;
            // threads_per_fft == 1 means that EPT == SIZE, so we need to store one more element
            constexpr unsigned int values_left_to_store =
                threads_per_fft == 1 ? 1 : (output_values_to_store % threads_per_fft);
            if (threadIdx.x < values_left_to_store) {
                output[index] = thread_data[FFT::elements_per_thread / 2];
            }
        }

        static inline __device__ unsigned int batch_offset_c2r(unsigned int local_fft_id) {
            unsigned int global_fft_id =
                FFT::ffts_per_block == 1 ? blockIdx.x : (blockIdx.x * FFT::ffts_per_block + local_fft_id);
            return ((cufftdx::size_of<FFT>::value / 2) + 1) * global_fft_id;
        }

        template<class ComplexType = complex_type>
        static inline __device__ void load_c2r(const ComplexType* input,
                                               ComplexType*       thread_data,
                                               unsigned int       local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset_c2r(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread / 2; i++) {
                thread_data[i] = input[index];
                index += stride;
            }
            constexpr unsigned int threads_per_fft       = cufftdx::size_of<FFT>::value / FFT::elements_per_thread;
            constexpr unsigned int output_values_to_load = (cufftdx::size_of<FFT>::value / 2) + 1;
            // threads_per_fft == 1 means that EPT == SIZE, so we need to load one more element
            constexpr unsigned int values_left_to_load =
                threads_per_fft == 1 ? 1 : (output_values_to_load % threads_per_fft);
            if (threadIdx.x < values_left_to_load) {
                thread_data[FFT::elements_per_thread / 2] = input[index];
            }
        }

        template<class ComplexType = complex_type>
        static inline __device__ void store_c2r(const ComplexType* thread_data,
                                                scalar_type*       output,
                                                unsigned int       local_fft_id) {
            const unsigned int offset = batch_offset(local_fft_id);
            const unsigned int stride = stride_size();
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                output[index] = reinterpret_cast<const scalar_type*>(thread_data)[i];
                index += stride;
            }
        }
    };
} // namespace example

#endif // MATHDX_CUFFTDX_HELPERS_EXAMPLE_BLOCK_IO_HPP_
