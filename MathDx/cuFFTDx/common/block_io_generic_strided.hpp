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

#ifndef CUFFTDX_EXAMPLE_BLOCK_IO_GENERIC_STRIDED_HPP
#define CUFFTDX_EXAMPLE_BLOCK_IO_GENERIC_STRIDED_HPP

#include "block_io.hpp"
#include "mixed_io.hpp"

namespace example {

    // Helper lables to mark which dimension is processed
    enum class dimension_description {
        X, Y, Z
    };

    namespace detail {
        // A helper structure that calculates stride / batches / batch stride from FFT sizes and
        // user-provided FFT dimension
        template<dimension_description Dimension, unsigned int SizeX, unsigned int SizeY, unsigned int SizeZ>
        struct data_access_helper {
            private:
                static constexpr bool is_2d_ = (SizeZ == 1);
                // Note: mapping 2d to use the same algo as for 3d
                //  - 3d:
                //    dim(0) := Z (continuous dimension)
                //    dim(1) := Y
                //    dim(2) := X
                //  - 2d:
                //    dim(0) := Y (continuous dimension), Z shouldn't appear here
                //    dim(1) := X
                //    dim(2) := unmapped
                static constexpr unsigned int dim_ = is_2d_
                    ? (Dimension == dimension_description::Z
                        ? 0
                        : (Dimension == dimension_description::Y ? 0 : 1))
                    : (Dimension == dimension_description::Z
                        ? 0
                        : (Dimension == dimension_description::Y ? 1 : 2));
                // Note: mapping 2d to use the same algo as for 3d
                //   - 3d:
                //     X := SizeX,
                //     Y := SizeY,
                //     Z := SizeZ (continuous dimension)
                //   - 2d:
                //     X := 1
                //     Y := SizeX
                //     Z := SizeY (continuous dimension)
                static constexpr unsigned int size_x = (is_2d_) ? 1 : SizeX;
                static constexpr unsigned int size_y = (is_2d_) ? SizeX : SizeY;
                static constexpr unsigned int size_z = (is_2d_) ? SizeY : SizeZ;

                static_assert(!(Dimension == dimension_description::Z && SizeZ == 1),
                    "Unsupported configuration: Z-dimension details are not supported for 2D FFTs");
            public:
                // A helper value with distance in elements by which strided batch offset will change
                // with increase of x-dimension coordinate of 2d mapping of batch ids, an approach
                // used for generalizing batch offset calculations.
                //
                // Example (2D):
                //  - Y (outer most, continuous),
                //   - the next group of FFTs has distance of the size of outermost dim (Y),
                //   - coordinates will have only x-dim increased while y-dim stays 0,
                //  - X (inner most),
                //   - the next group of FFTs has distance of Y * X,
                //   - both coordinates (x,y) will change,
                //   - batch ids with the same x-dim coordinate are offset by Y,
                //   - batch ids with the same y-dim coordinate but x-dim differ by one are offset by Y * X,
                //
                // For 3D FFT the same rules apply. The inner most dimension batch id offset for x-dim
                // 2d mapping is X * Y * Z but in this case x-dim of 2d mapping is always 0 so
                // bid_mapping_x_stride does not change offset for inner most dimension.
                static constexpr unsigned int bid_mapping_x_stride = size_z * (dim_ > 0 ? size_y : 1) * (dim_ > 1 ? size_x : 1);
                // A value that represents the number of FFT batches for user-provided FFT dimension
                static constexpr unsigned int batches = (dim_ == 0 ? 1 : size_z) * (dim_ == 1 ? 1 : size_y) * (dim_ == 2 ? 1 : size_x);
                // A stride value between consecurive elements in a FFT batch for user-provided FFT dimension. For example
                // for the outermost dimension (continuous) element stride is 1.
                // It is also used for mapping batch id into x and y coordinates that enable a simplified way to calculate
                // batch offset for mentioned user-provided FFT dimension
                static constexpr unsigned int element_stride = (dim_ >= 1 ? size_z : 1) * (dim_ >= 2 ? size_y : 1);

                static inline __device__ unsigned int batch_offset_strided(unsigned int bid) {
                    const unsigned int batch_coord_x = bid / element_stride;
                    const unsigned int batch_coord_y = bid % element_stride;
                    return batch_coord_x * bid_mapping_x_stride + batch_coord_y;
                }
        };
    } // namespace detail

    template<class FFT>
    struct io_generic_strided: public io<FFT> {
        using base_type = io<FFT>;

        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static inline __device__ unsigned int batch_id(unsigned int local_fft_id) {
            unsigned int global_fft_id = blockIdx.x * FFT::ffts_per_block + local_fft_id;
            return global_fft_id;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads and stores for 2d and 3d FFTs, based on FFT's sizes and specific dimension
        template<dimension_description Dimension, unsigned int SizeX, unsigned int SizeY, unsigned int SizeZ = 1, typename InputOutputType>
        static inline __device__ void load_strided(const InputOutputType* input,
                                                   complex_type*          thread_data,
                                                   unsigned int           local_fft_id) {
            using dah_type = detail::data_access_helper<Dimension, SizeX, SizeY, SizeZ>;

            const unsigned int bid = batch_id(local_fft_id);
            if (bid < dah_type::batches) {
                // Calculate global offset of FFT batch
                const unsigned int batch_offset  = dah_type::batch_offset_strided(bid);
                // Get stride between elements loaded by specific thread, it takes into an
                // accound ept/number of threads participating in the batch
                const unsigned int ept_stride    = dah_type::element_stride * FFT::stride;
                // Index of the first element loaded by given thread in a block, elements
                // processed by this thread are separated by ept_stride distance
                unsigned int       index         = batch_offset + (threadIdx.x * dah_type::element_stride);
                for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                    if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                        thread_data[i] = convert<complex_type>(input[index]);
                    }
                    index += ept_stride;
                }
            }
        }

        template<dimension_description Dimension, unsigned int SizeX, unsigned int SizeY, unsigned int SizeZ = 1, typename InputOutputType>
        static inline __device__ void store_strided(const complex_type* thread_data,
                                                    InputOutputType*    output,
                                                    unsigned int        local_fft_id) {
            using dah_type = detail::data_access_helper<Dimension, SizeX, SizeY, SizeZ>;

            const unsigned int bid = batch_id(local_fft_id);
            if (bid < dah_type::batches) {
                // Calculate global offset of FFT batch
                const unsigned int batch_offset  = dah_type::batch_offset_strided(bid);
                // Get stride between elements loaded by specific thread, it takes into an
                // accound ept/number of threads participating in the batch
                const unsigned int ept_stride    = dah_type::element_stride * FFT::stride;
                // Index of the first element loaded by given thread in a block, elements
                // processed by this thread are separated by ept_stride distance
                unsigned int       index         = batch_offset + (threadIdx.x * dah_type::element_stride);
                for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                    if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                        output[index] = convert<InputOutputType>(thread_data[i]);
                    }
                    index += ept_stride;
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads and stores for 2d and 3d FFTs with shared memory used, based on FFT's sizes and specific dimension
        template<dimension_description Dimension, unsigned int SizeX, unsigned int SizeY, unsigned int SizeZ = 1, typename InputOutputType>
        static inline __device__ void load_strided(const InputOutputType* input,
                                                   complex_type*          thread_data,
                                                   InputOutputType*       shared_memory,
                                                   unsigned int           local_fft_id) {
            using dah_type = detail::data_access_helper<Dimension, SizeX, SizeY, SizeZ>;

            const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx         = tid / blockDim.y;
            const unsigned int tidy         = tid % blockDim.y;
            unsigned int       smem_index   = tidx + tidy * blockDim.x;

            const unsigned int bid          = batch_id(tidy);
            if (bid < dah_type::batches) {
                // Calculate global offset of FFT batch
                const unsigned int batch_offset = dah_type::batch_offset_strided(bid);
                // Get stride between elements loaded by specific thread, it takes into an
                // accound ept/number of threads participating in the batch
                const unsigned int ept_stride   = dah_type::element_stride * FFT::stride;
                // Index of the first element loaded by given thread in a block, elements
                // processed by this thread are separated by ept_stride distance
                unsigned int       index        = batch_offset + (tidx * dah_type::element_stride);
                for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                    if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
                        shared_memory[smem_index] = input[index];
                        index += ept_stride;
                        smem_index += (blockDim.x * blockDim.y);
                    }
                }
            }
            __syncthreads();
            smem_index = threadIdx.x + threadIdx.y * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    thread_data[i] = convert<complex_type>(shared_memory[smem_index]);
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
        }

        template<dimension_description Dimension, unsigned int SizeX, unsigned int SizeY, unsigned int SizeZ = 1, typename InputOutputType>
        static inline __device__ void store_strided(const complex_type* thread_data,
                                                    InputOutputType*    shared_memory,
                                                    InputOutputType*    output,
                                                    unsigned int        local_fft_id) {
            __syncthreads();
            unsigned int smem_index = threadIdx.x + threadIdx.y * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    shared_memory[smem_index] = convert<InputOutputType>(thread_data[i]);
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
            __syncthreads();
            using dah_type = detail::data_access_helper<Dimension, SizeX, SizeY, SizeZ>;
            const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx         = tid / blockDim.y;
            const unsigned int tidy         = tid % blockDim.y;
            smem_index                      = tidx + tidy * blockDim.x;

            const unsigned int bid          = batch_id(tidy);
            if (bid < dah_type::batches) {
                // Calculate global offset of FFT batch
                const unsigned int batch_offset = dah_type::batch_offset_strided(bid);
                // Get stride between elements loaded by specific thread, it takes into an
                // accound ept/number of threads participating in the batch
                const unsigned int ept_stride   = dah_type::element_stride * FFT::stride;
                // Index of the first element loaded by given thread in a block, elements
                // processed by this thread are separated by ept_stride distance
                unsigned int       index        = batch_offset + (tidx * dah_type::element_stride);
                for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                    if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
                        output[index] = shared_memory[smem_index];
                        index += ept_stride;
                        smem_index += (blockDim.x * blockDim.y);
                    }
                }
            }
        }
    };
} // namespace example

#endif // CUFFTDX_EXAMPLE_BLOCK_IO_GENERIC_STRIDED_HPP