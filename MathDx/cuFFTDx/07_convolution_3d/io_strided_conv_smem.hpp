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

#ifndef CUFFTDX_EXAMPLE_3D_io_strided_conv_smem_HPP
#define CUFFTDX_EXAMPLE_3D_io_strided_conv_smem_HPP

#include "../common/common.hpp"
#include "../common/block_io.hpp"

#include "index_mapper.hpp"

namespace example {

    // This IO struct is meant to facilitate IO for 3D convolution,
    // where the first direction has Front set to true, and the second
    // direction has Front set to false.
    // This is not the same as Forward and Inverse, because when doing
    // C2R conv, the order will be Inverse -> Forward.
    // FFTX, FFTY and FFTZ describe dimensions of 3D FFT and:
    // - FFTX is the outermost (most strided)
    // - FFTY is the second outermost (strided)
    // - FFTZ is contiguous
    template<dimension Dim, bool Front, int Batches, class FFTX_, class IFFTX_, class FFTY_, class IFFTY_, class FFTZ_, class IFFTZ_>
    class io_strided_conv_smem
    {
        // Convolution happens in the x dimension
        static constexpr bool is_c2r_conv =
            cufftdx::type_of<FFTZ_>::value == cufftdx::fft_type::c2r and
            cufftdx::type_of<IFFTZ_>::value == cufftdx::fft_type::r2c;

        static constexpr bool is_r2c_conv =
            cufftdx::type_of<FFTZ_>::value == cufftdx::fft_type::r2c and
            cufftdx::type_of<IFFTZ_>::value == cufftdx::fft_type::c2r;

        static constexpr bool is_c2c_conv = not is_c2r_conv and not is_r2c_conv;

        using FFTX = std::conditional_t<Front, FFTX_, IFFTX_>;
        using FFTY = std::conditional_t<Front, FFTY_, IFFTY_>;
        using FFTZ = std::conditional_t<Front, FFTZ_, IFFTZ_>;

        // This value type is used for X and Y dimensions because they always operate on
        // complex data.
        using value_type = typename FFTX::value_type;
        static_assert(std::is_same_v<value_type, typename FFTY::value_type>);
        static_assert(std::is_same_v<value_type, typename FFTZ::value_type>);

        // X and Y never change and always:
        // FFT::input_length == FFT::output_length == size_of<FFT>::value
        static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;
        static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
        static constexpr unsigned int fft_size_z = cufftdx::size_of<FFTZ>::value;
        // This is a value which determines what length of z other dimensions see,
        // so e.g. in R2C
        static constexpr unsigned int z_dim = is_c2r_conv ? FFTZ_::input_length : FFTZ_::output_length;

        static constexpr unsigned int flat_batch_size = fft_size_x * fft_size_y * z_dim;

        // In this loading scheme the global memory accesses are coalesced and shared
        // memory bank conflicts are minimized by padding. This is achieved by:
        // 1. Contiguous threads loading corresponding elements from subsequent batches
        //      and not from the same batch, as would take place normally. This emulates
        //      switching blockDim.x with blockDim.y, because stride between batches is 1
        //      and stride between elements from the same batch is the total number of
        //      subbatches.
        // 2. Since the stores to shared memory will be performed strided by a high power
        //      of 2 it is necessary to pad them by the number of threads which will be
        //      performing this. Hence the _pad values are created based on fpb and warp
        //      size. This padding works well only for powers of 2.
        static constexpr auto x_fpb = FFTX::ffts_per_block;
        static constexpr auto x_pad = (example::warp_size + (x_fpb - 1)) / x_fpb;

        // This layout defines the offsets in global memory to get a specific element.
        // It can be addressed using only a single integer, which defines the global
        // index of an element, which it maps to an appropriate offset in memory.
        // Single index is decayed to full N-D index by applying modulo and division
        // recursively. The pairs given in index_mapper definition are (Size, Stride) pairs.
        // --------------------------------------------------------------------------
        // Please refer to io_strided_conv_smem.hpp for further explanation of multidimensional addressing.

        using global_layout_x_subbatches = index_mapper<int_pair<z_dim, 1>,           // inner 2d dimension
                                                        int_pair<fft_size_y, z_dim>>; // outer 2d dimension

        using global_layout_x = index_mapper<int_pair<fft_size_x, fft_size_y * z_dim>, // element dimension
                                             global_layout_x_subbatches,               // 2d subbatches
                                             int_pair<Batches, flat_batch_size>>;      // batch dimension

        // Pad shared memory in subbatch dimension to reduce shared memory bank conflicts
        using shared_layout_x = index_mapper<int_pair<fft_size_x, 1>,
                                             int_pair<x_fpb, fft_size_x + x_pad>>;

        static constexpr int x_pad_bytes    = fft_size_x * x_fpb * sizeof(value_type) + x_fpb * x_pad * sizeof(value_type);
        static constexpr int x_shared_bytes = std::max<int>(FFTX::shared_memory_size, x_pad_bytes);

        // Y dimension configuration
        static constexpr auto y_fpb = FFTY::ffts_per_block;
        static constexpr auto y_pad = (example::warp_size + (y_fpb - 1)) / y_fpb;

        // Detailed explanation presented in X dimension
        using global_layout_y_subbatches = index_mapper<int_pair<z_dim, 1>,                        // inner 2d dimension
                                                        int_pair<fft_size_x, z_dim * fft_size_y>>; // outer 2d dimension

        using global_layout_y = index_mapper<int_pair<fft_size_y, z_dim>,         // element dimension
                                             global_layout_y_subbatches,          // 2d subbatches
                                             int_pair<Batches, flat_batch_size>>; // batch dimension

        using shared_layout_y = index_mapper<int_pair<fft_size_y, 1>,
                                             int_pair<y_fpb, fft_size_y + y_pad>>;

        static constexpr int y_pad_bytes    = fft_size_y * y_fpb * sizeof(value_type) + y_fpb * y_pad * sizeof(value_type);
        static constexpr int y_shared_bytes = std::max<int>(FFTY::shared_memory_size, y_pad_bytes);


        // These IO functions perform shared <--> registers transfers based on provided layouts.
        // This enables taking padding under account.
        template<class FFT, int Subbatches, class GlobalLayout, class SharedLayout>
        __device__ __forceinline__ void load_gmem_to_smem(const value_type* gmem, value_type* smem) const {
            GlobalLayout global_layout;
            SharedLayout shared_layout;

            constexpr auto        fpb            = FFT::ffts_per_block;
            const auto            this_block_fpb = (blockIdx.x == Subbatches / fpb) ? Subbatches % fpb : fpb;
            static constexpr auto fft_size       = FFT::input_length;

            // Load data from global by emulating a switch between
            // threadIdx.x and threadIdx.y
            const int tid            = (threadIdx.x + threadIdx.y * blockDim.x);
            const int rev_elem_start = tid / this_block_fpb;
            const int rev_batch_id   = tid % this_block_fpb;

            using input_t   = typename FFT::input_type;
            auto input_smem = reinterpret_cast<input_t*>(smem);
            auto input_gmem = reinterpret_cast<const input_t*>(gmem);

            // Since it's a strided kernel it requires staging the data through shared memory to
            // achieve high global memory coalescing on loads and stores.
#pragma unroll
            for (int i = 0; i < FFT::input_ept; ++i) {
                const auto rev_elem_id         = rev_elem_start + i * FFT::stride;
                const auto global_rev_batch_id = rev_batch_id + blockIdx.x * fpb;
                if (not FFT::requires_workspace or (rev_elem_id < fft_size)) {
                    input_smem[shared_layout(rev_elem_id, rev_batch_id)] =
                        input_gmem[global_layout(rev_elem_id, global_rev_batch_id, blockIdx.y)];
                }
            }
        }

        template<class FFT, int Subbatches, class SharedLayout, class GlobalLayout>
        __device__ __forceinline__ void store_smem_to_gmem(const value_type* smem, value_type* gmem) const {
            GlobalLayout global_layout;
            SharedLayout shared_layout;

            constexpr auto        fpb            = FFT::ffts_per_block;
            const auto            this_block_fpb = (blockIdx.x == Subbatches / fpb) ? Subbatches % fpb : fpb;
            static constexpr auto fft_size       = FFT::output_length;

            // Load data from global by emulating a switch between
            // threadIdx.x and threadIdx.y
            const int tid            = (threadIdx.x + threadIdx.y * blockDim.x);
            const int rev_elem_start = tid / this_block_fpb;
            const int rev_batch_id   = tid % this_block_fpb;

            using output_t   = typename FFT::output_type;
            auto output_gmem = reinterpret_cast<output_t*>(gmem);
            auto output_smem = reinterpret_cast<const output_t*>(smem);

#pragma unroll
            for (int i = 0; i < FFT::output_ept; ++i) {
                const auto rev_elem_id         = rev_elem_start + i * FFT::stride;
                const auto global_rev_batch_id = rev_batch_id + blockIdx.x * fpb;
                if (not FFT::requires_workspace or (rev_elem_id < fft_size)) {
                    output_gmem[global_layout(rev_elem_id, global_rev_batch_id, blockIdx.y)] =
                        output_smem[shared_layout(rev_elem_id, rev_batch_id)];
                }
            }
        }

        // Shared memory must be synchronized before
        // no guarantees on shared memory sync after
        template<class FFT, class SharedLayout, class Op>
        __device__ __forceinline__ void load_smem_to_rmem(const value_type* smem, value_type* rmem) const {
            SharedLayout shared_layout;
            Op           op;

            static constexpr auto fft_size = FFT::input_length;

            using input_t   = typename FFT::input_type;
            auto input_rmem = reinterpret_cast<input_t*>(rmem);
            auto input_smem = reinterpret_cast<const input_t*>(smem);

#pragma unroll
            for (int i = 0; i < FFT::input_ept; ++i) {
                const auto elem_id  = threadIdx.x + i * FFT::stride;
                const auto batch_id = threadIdx.y;
                if (not FFT::requires_workspace or (elem_id < fft_size)) {
                    input_rmem[i] = op(input_smem[shared_layout(elem_id, batch_id)]);
                }
            }
        }

        // Shared memory must be synchronized before
        // no guarantees on shared memory sync after
        template<class FFT, class SharedLayout, class Op>
        __device__ __forceinline__ void store_rmem_to_smem(const value_type* rmem, value_type* smem) const {
            SharedLayout shared_layout;
            Op           op;

            static constexpr auto fft_size = FFT::output_length;
            using output_t                 = typename FFT::output_type;
            auto output_smem               = reinterpret_cast<output_t*>(smem);
            auto output_rmem               = reinterpret_cast<const output_t*>(rmem);

#pragma unroll
            for (int i = 0; i < FFT::output_ept; ++i) {
                const auto elem_id  = threadIdx.x + i * FFT::stride;
                const auto batch_id = threadIdx.y;
                if (not FFT::requires_workspace or (elem_id < fft_size)) {
                    output_smem[shared_layout(elem_id, batch_id)] = op(output_rmem[i]);
                }
            }
        }

    public:
        static constexpr __device__ __host__ __forceinline__
            size_t
            get_shared_bytes() {
            if (Dim == dimension::x) {
                return x_shared_bytes;
            } else if (Dim == dimension::y) {
                return y_shared_bytes;
            } else {
                return FFTZ::shared_memory_size;
            }
        }

        template<typename GmemType, typename SmemType, typename RmemType, class LoadOp = example::identity>
        __device__ __forceinline__ void load_gmem_to_rmem(const GmemType* gmem, [[maybe_unused]] SmemType* smem, RmemType* rmem, [[maybe_unused]] LoadOp op = {}) const {
            if constexpr (Dim == dimension::x) {
                constexpr int x_batches = fft_size_y * z_dim;
                load_gmem_to_smem<FFTX, x_batches, global_layout_x, shared_layout_x>(gmem, smem);
                __syncthreads();
                load_smem_to_rmem<FFTX, shared_layout_x, LoadOp>(smem, rmem);
            } else if constexpr (Dim == dimension::y) {
                constexpr int y_batches = fft_size_x * z_dim;
                load_gmem_to_smem<FFTY, y_batches, global_layout_y, shared_layout_y>(gmem, smem);
                __syncthreads();
                load_smem_to_rmem<FFTY, shared_layout_y, LoadOp>(smem, rmem);
            } else { // Z dimension (contiguous)
                constexpr auto block_offset = (Front and is_r2c_conv) ? (fft_size_x * fft_size_y * FFTZ::input_length) : flat_batch_size;
                using input_t               = typename FFTZ::input_type;
                auto gmem_input             = reinterpret_cast<const input_t*>(gmem);
                example::io<FFTZ>::load(gmem_input + blockIdx.y * block_offset, rmem, threadIdx.y, op);
            }
        }

        template<typename RmemType, typename SmemType, typename GmemType, class StoreOp = example::identity>
        __device__ __forceinline__ void store_rmem_to_gmem(const RmemType* rmem, [[maybe_unused]] SmemType* smem, GmemType* gmem, [[maybe_unused]] StoreOp op = {}) const {
            if constexpr (Dim == dimension::x) {
                constexpr int x_batches = fft_size_y * z_dim;
                store_rmem_to_smem<FFTX, shared_layout_x, StoreOp>(rmem, smem);
                __syncthreads();
                store_smem_to_gmem<FFTX, x_batches, shared_layout_x, global_layout_x>(smem, gmem);
            } else if constexpr (Dim == dimension::y) {
                constexpr int y_batches = fft_size_x * z_dim;
                store_rmem_to_smem<FFTY, shared_layout_y, StoreOp>(rmem, smem);
                __syncthreads();
                store_smem_to_gmem<FFTY, y_batches, shared_layout_y, global_layout_y>(smem, gmem);
            } else { // Z dimension (contiguous)
                constexpr auto block_offset = (not Front and is_r2c_conv) ? (fft_size_x * fft_size_y * FFTZ::output_length) : flat_batch_size;
                using output_t              = typename FFTZ::output_type;
                auto gmem_output            = reinterpret_cast<output_t*>(gmem);
                example::io<FFTZ>::store(rmem, gmem_output + blockIdx.y * block_offset, threadIdx.y, op);
            }
        }
    }; // io_strided_conv_smem
} // namespace example


#endif // CUFFTDX_EXAMPLE_3D_io_strided_conv_smem_HPP