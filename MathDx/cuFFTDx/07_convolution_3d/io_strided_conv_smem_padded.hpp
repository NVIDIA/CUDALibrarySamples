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

#ifndef CUFFTDX_EXAMPLE_3D_IO_STRIDED_CONV_SMEM_PADDED_HPP
#define CUFFTDX_EXAMPLE_3D_IO_STRIDED_CONV_SMEM_PADDED_HPP

#include "../common/common.hpp"
#include "../common/block_io.hpp"
#include "../common/padded_io.hpp"

#include "index_mapper.hpp"

namespace example {
    // This IO struct is meant to facilitate 0 padded IO for 3D
    // convolution with Z as contiguous dimension and X as
    // fused convolution dimension.
    //
    // Padding means that input signal has dimensions (IX, IY, IZ)
    // different than FFT dimensions (X, Y, Z) and
    // (SX, SY, SZ) <= (X, Y, Z)
    // To avoid extra global memory sync requirements, cuFFTDx will
    // emulate full pad data padding since the beginning and pad the
    // convolution dimension only virtually in registers. This way
    // the memory requirement is kept at SX * Y * Z elements.
    //
    // FFTX, FFTY and FFTZ describe dimensions of 3D FFT and:
    // - FFTX is the outermost (most strided)
    // - FFTY is the second outermost (strided)
    // - FFTZ is contiguous
    template<dimension Dim, bool Front, int Batches, class FFTX_, class IFFTX_, class FFTY_, class IFFTY_, class FFTZ_, class IFFTZ_, int SignalLengthX = cufftdx::size_of<FFTX_>::value, int SignalLengthY = cufftdx::size_of<FFTY_>::value, int SignalLengthZ = cufftdx::size_of<FFTZ_>::value>
    class io_strided_conv_smem_padded
    {
        // Convolution happens in the x dimension
        static constexpr bool is_r2c_conv =
            cufftdx::type_of<FFTZ_>::value == cufftdx::fft_type::r2c and
            cufftdx::type_of<IFFTZ_>::value == cufftdx::fft_type::c2r;

        static constexpr bool is_c2c_conv =
            cufftdx::type_of<FFTZ_>::value == cufftdx::fft_type::c2c and
            cufftdx::type_of<IFFTZ_>::value == cufftdx::fft_type::c2c;

        static_assert(is_r2c_conv or is_c2c_conv);

        using FFTX = std::conditional_t<Front, FFTX_, IFFTX_>;
        using FFTY = std::conditional_t<Front, FFTY_, IFFTY_>;
        using FFTZ = std::conditional_t<Front, FFTZ_, IFFTZ_>;

        // This value type is used for X and Y dimensions because they always operate on
        // complex data.
        using value_type = typename FFTX::value_type;
        static_assert(std::is_same_v<value_type, typename FFTY::value_type>);
        static_assert(std::is_same_v<value_type, typename FFTZ::value_type>);

        static constexpr auto num_shared_banks = (32 * sizeof(float)) / sizeof(value_type);

        // X and Y never change and always:
        // FFT::input_length == FFT::output_length == size_of<FFT>::value
        static constexpr unsigned int fft_size_x = cufftdx::size_of<FFTX>::value;
        static constexpr unsigned int fft_size_y = cufftdx::size_of<FFTY>::value;
        static constexpr unsigned int fft_size_z = cufftdx::size_of<FFTZ>::value;

        // Non zero-padded sizes
        static constexpr unsigned int signal_length_x = SignalLengthX;
        static constexpr unsigned int signal_length_y = SignalLengthY;
        static constexpr unsigned int signal_length_z = SignalLengthZ;

        // If padded and non padded are equal, then there is no padding
        static constexpr bool is_x_padded = fft_size_x != signal_length_x;
        static constexpr bool is_y_padded = fft_size_y != signal_length_y;
        static constexpr bool is_z_padded = fft_size_z != signal_length_z;

        // This is a value which determines what length of z other dimensions see,
        // so e.g. in R2C this will be equal to length of complex output
        static constexpr unsigned int z_dim = FFTZ_::output_length;

        static constexpr unsigned int flat_batch_size  = signal_length_x * fft_size_y * z_dim;
        static constexpr unsigned int flat_signal_size = signal_length_x * signal_length_y * signal_length_z;

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
        static constexpr auto x_fpb         = FFTX::ffts_per_block;
        static constexpr auto x_bank_offset = (signal_length_x % num_shared_banks == 0) ? (example::warp_size + (x_fpb - 1)) / x_fpb : 0;

        // This layout defines the offsets in global memory to get a specific element.
        // It can be addressed using only a single integer, which defines the global
        // index of an element, which it maps to an appropriate offset in memory.
        // Single index is decayed to full N-D index by applying modulo and division
        // recursively. The pairs given in index_mapper definition are (Size, Stride) pairs.
        // --------------------------------------------------------------------------
        // Please refer to io_strided_conv_smem.hpp for further explanation of multidimensional addressing.

        using global_layout_x_subbatches = index_mapper<int_pair<z_dim, 1>,           // inner 2d dimension
                                                        int_pair<fft_size_y, z_dim>>; // outer 2d dimension

        using global_layout_x = index_mapper<int_pair<signal_length_x, fft_size_y * z_dim>, // element dimension
                                             global_layout_x_subbatches,                    // 2d subbatches
                                             int_pair<Batches, flat_batch_size>>;           // batch dimension

        // Pad shared memory in subbatch dimension to reduce shared memory bank conflicts
        using shared_layout_x = index_mapper<int_pair<signal_length_x, 1>,
                                             int_pair<x_fpb, signal_length_x + x_bank_offset>>;

        static constexpr int x_bank_offset_bytes = (signal_length_x + x_bank_offset) * x_fpb * sizeof(value_type);
        static constexpr int x_shared_bytes      = std::max<int>(FFTX::shared_memory_size, x_bank_offset_bytes);

        // Y dimension configuration
        static constexpr auto y_fpb         = FFTY::ffts_per_block;
        static constexpr auto y_bank_offset = (fft_size_y % num_shared_banks == 0) ? (example::warp_size + (y_fpb - 1)) / y_fpb : 0;

        // Detailed explanation presented in X dimension
        using global_layout_y_subbatches = index_mapper<int_pair<z_dim, 1>,                             // inner 2d dimension
                                                        int_pair<signal_length_x, z_dim * fft_size_y>>; // outer 2d dimension

        using global_layout_y = index_mapper<int_pair<fft_size_y, z_dim>,         // element dimension
                                             global_layout_y_subbatches,          // 2d subbatches
                                             int_pair<Batches, flat_batch_size>>; // batch dimension

        using shared_layout_y = index_mapper<int_pair<fft_size_y, 1>,
                                             int_pair<y_fpb, fft_size_y + y_bank_offset>>;

        static constexpr int y_bank_offset_bytes = (fft_size_y + y_bank_offset) * y_fpb * sizeof(value_type);
        static constexpr int y_shared_bytes      = std::max<int>(FFTY::shared_memory_size, y_bank_offset_bytes);


        // These IO functions perform shared <--> registers transfers based on provided layouts.
        // This enables taking padding under account.
        template<class FFT, int Subbatches, int SignalLength, class GlobalLayout, class SharedLayout>
        __device__ __forceinline__ void load_gmem_to_smem(const value_type* gmem, value_type* smem) const {
            GlobalLayout global_layout;
            SharedLayout shared_layout;

            constexpr auto fpb            = FFT::ffts_per_block;
            constexpr auto is_padded      = SignalLength != cufftdx::size_of<FFT>::value;

            // While all blocks must be started with the same FPB and number of threads,
            // this example is flexible enough to allow batches % FPB != 0. In that case
            // the last block will process exactly (batches % FPB) subbatches, and
            // this_block_fpb is the actual value of computed FFTs for each block.
            const auto     this_block_fpb = (blockIdx.x == Subbatches / fpb) ? Subbatches % fpb : fpb;

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
                if ((not FFT::requires_workspace and not is_padded) or (rev_elem_id < SignalLength)) {
                    input_smem[shared_layout(rev_elem_id, rev_batch_id)] =
                        input_gmem[global_layout(rev_elem_id, global_rev_batch_id, blockIdx.y)];
                }
            }
        }

        template<class FFT, int Subbatches, int SignalLength, class SharedLayout, class GlobalLayout>
        __device__ __forceinline__ void store_smem_to_gmem(const value_type* smem, value_type* gmem) const {
            GlobalLayout global_layout;
            SharedLayout shared_layout;

            constexpr auto is_padded      = SignalLength != cufftdx::size_of<FFT>::value;
            constexpr auto fpb            = FFT::ffts_per_block;
            const auto     this_block_fpb = (blockIdx.x == Subbatches / fpb) ? Subbatches % fpb : fpb;

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
                if ((not FFT::requires_workspace and not is_padded) or (rev_elem_id < SignalLength)) {
                    output_gmem[global_layout(rev_elem_id, global_rev_batch_id, blockIdx.y)] =
                        output_smem[shared_layout(rev_elem_id, rev_batch_id)];
                }
            }
        }

        // Shared memory must be synchronized before
        // no guarantees on shared memory sync after
        template<class FFT, int SignalLength, class SharedLayout, class Op>
        __device__ __forceinline__ void load_smem_to_rmem(const value_type* smem, value_type* rmem) const {
            SharedLayout shared_layout;
            Op           op;

            static constexpr auto is_padded = SignalLength != cufftdx::size_of<FFT>::value;

            using input_t   = typename FFT::input_type;
            auto input_rmem = reinterpret_cast<input_t*>(rmem);
            auto input_smem = reinterpret_cast<const input_t*>(smem);

#pragma unroll
            for (int i = 0; i < FFT::input_ept; ++i) {
                const auto elem_id  = threadIdx.x + i * FFT::stride;
                const auto batch_id = threadIdx.y;
                if ((not FFT::requires_workspace and not is_padded) or (elem_id < SignalLength)) {
                    input_rmem[i] = op(input_smem[shared_layout(elem_id, batch_id)]);
                } else if (is_padded and elem_id < FFT::input_length) {
                    input_rmem[i] = get_zero<input_t>();
                }
            }
        }

        // Shared memory must be synchronized before
        // no guarantees on shared memory sync after
        template<class FFT, int SignalLength, class SharedLayout, class Op>
        __device__ __forceinline__ void store_rmem_to_smem(const value_type* rmem, value_type* smem) const {
            SharedLayout shared_layout;
            Op           op;

            static constexpr auto is_padded = SignalLength != cufftdx::size_of<FFT>::value;

            using output_t   = typename FFT::output_type;
            auto output_smem = reinterpret_cast<output_t*>(smem);
            auto output_rmem = reinterpret_cast<const output_t*>(rmem);

#pragma unroll
            for (int i = 0; i < FFT::output_ept; ++i) {
                const auto elem_id  = threadIdx.x + i * FFT::stride;
                const auto batch_id = threadIdx.y;
                if ((not FFT::requires_workspace and not is_padded) or (elem_id < SignalLength)) {
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
                constexpr int x_batches = z_dim * fft_size_y;
                load_gmem_to_smem<FFTX, x_batches, signal_length_x, global_layout_x, shared_layout_x>(gmem, smem);
                __syncthreads();
                load_smem_to_rmem<FFTX, signal_length_x, shared_layout_x, LoadOp>(smem, rmem);
            } else if constexpr (Dim == dimension::y) {
                constexpr int signal_length = Front ? signal_length_y : fft_size_y;
                constexpr int y_batches     = z_dim * signal_length_x;
                load_gmem_to_smem<FFTY, y_batches, signal_length, global_layout_y, shared_layout_y>(gmem, smem);
                __syncthreads();
                load_smem_to_rmem<FFTY, signal_length, shared_layout_y, LoadOp>(smem, rmem);
            } else { // Z dimension (contiguous)
                constexpr bool is_load_padded   = is_z_padded and Front;
                using io_t                      = std::conditional_t<is_load_padded, example::io_padded<FFTZ, signal_length_z>, example::io<FFTZ>>;
                using input_t                   = std::conditional_t<is_load_padded, GmemType, typename FFTZ::input_type>;
                constexpr int y_pad             = fft_size_y - signal_length_y;
                const auto    additional_offset = (not Front and is_y_padded) ? ((blockIdx.x * FFTZ::ffts_per_block + threadIdx.y) / signal_length_y) * y_pad : 0;

                constexpr auto non_padded_block_offset = (Front and is_r2c_conv) ? (signal_length_x * fft_size_y * FFTZ::input_length) : flat_batch_size;
                constexpr auto block_offset            = Front ? flat_signal_size : non_padded_block_offset;

                auto gmem_input = reinterpret_cast<const input_t*>(gmem);
                io_t::load(reinterpret_cast<const input_t*>(gmem + blockIdx.y * block_offset), rmem, threadIdx.y + additional_offset, op);
            }
        }

        template<typename RmemType, typename SmemType, typename GmemType, class StoreOp = example::identity>
        __device__ __forceinline__ void store_rmem_to_gmem(const RmemType* rmem, [[maybe_unused]] SmemType* smem, GmemType* gmem, [[maybe_unused]] StoreOp op = {}) const {
            if constexpr (Dim == dimension::x) {
                constexpr int x_batches = z_dim * fft_size_y;
                store_rmem_to_smem<FFTX, signal_length_x, shared_layout_x, StoreOp>(rmem, smem);
                __syncthreads();
                store_smem_to_gmem<FFTX, x_batches, signal_length_x, shared_layout_x, global_layout_x>(smem, gmem);
            } else if constexpr (Dim == dimension::y) {
                constexpr int signal_length = Front ? fft_size_y : signal_length_y;
                constexpr int y_batches     = z_dim * signal_length_x;
                store_rmem_to_smem<FFTY, signal_length, shared_layout_y, StoreOp>(rmem, smem);
                __syncthreads();
                store_smem_to_gmem<FFTY, y_batches, signal_length, shared_layout_y, global_layout_y>(smem, gmem);
            } else { // Z dimension (contiguous)
                constexpr bool is_store_padded  = is_z_padded and not Front;
                using io_t                      = std::conditional_t<is_store_padded, example::io_padded<FFTZ, signal_length_z>, example::io<FFTZ>>;
                using output_t                  = std::conditional_t<is_store_padded, GmemType, typename FFTZ::output_type>;
                constexpr int y_pad             = fft_size_y - signal_length_y;
                const auto    additional_offset = (Front and is_y_padded) ? ((blockIdx.x * FFTZ::ffts_per_block + threadIdx.y) / signal_length_y) * y_pad : 0;

                constexpr auto non_padded_block_offset = (not Front and is_r2c_conv) ? (signal_length_x * fft_size_y * FFTZ::output_length) : flat_batch_size;
                constexpr auto block_offset            = (not Front) ? flat_signal_size : non_padded_block_offset;

                auto gmem_output = reinterpret_cast<output_t*>(gmem);
                io_t::store(rmem, reinterpret_cast<output_t*>(gmem + blockIdx.y * block_offset), threadIdx.y + additional_offset, op);
            }
        }
    }; // io_strided_conv_smem
} // namespace example


#endif // CUFFTDX_EXAMPLE_3D_IO_STRIDED_CONV_SMEM_PADDED_HPP