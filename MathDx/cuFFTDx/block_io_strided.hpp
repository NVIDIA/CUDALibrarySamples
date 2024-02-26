#ifndef CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP
#define CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP

#include "block_io.hpp"

namespace example {
    template<class FFT>
    struct io_strided: public io<FFT> {
        using base_type = io<FFT>;

        using complex_type = typename FFT::value_type;
        using scalar_type  = typename complex_type::value_type;

        static inline __device__ unsigned int batch_id(unsigned int local_fft_id) {
            unsigned int global_fft_id = blockIdx.x * FFT::ffts_per_block + local_fft_id;
            return global_fft_id;
        }

        static inline __device__ unsigned int batch_offset_strided(unsigned int local_fft_id) {
            return batch_id(local_fft_id);
        }

        template<unsigned int Stride, unsigned int Batches = Stride>
        static inline __device__ void load_strided(const complex_type* input,
                                                   complex_type*       thread_data,
                                                   unsigned int        local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset_strided(local_fft_id);
            const unsigned int bid = batch_id(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = Stride * base_type::stride_size();
            unsigned int       index  = offset + (threadIdx.x * Stride);
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * base_type::stride_size() + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    if(bid < Batches) {
                        thread_data[i] = input[index];
                    }
                    index += stride;
                }
            }
        }

        template<unsigned int Stride, unsigned int Batches = Stride>
        static inline __device__ void store_strided(const complex_type* thread_data,
                                                    complex_type*       output,
                                                    unsigned int        local_fft_id) {
            const unsigned int offset = batch_offset_strided(local_fft_id);
            const unsigned int bid = batch_id(local_fft_id);
            const unsigned int stride = Stride * base_type::stride_size();
            unsigned int       index  = offset + (threadIdx.x * Stride);
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * base_type::stride_size() + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    if(bid < Batches) {
                        output[index] = thread_data[i];
                    }
                    index += stride;
                }
            }
        }

        template<unsigned int Stride, unsigned int Batches = Stride>
        static inline __device__ void load_strided(const complex_type* input,
                                                   complex_type*       thread_data,
                                                   complex_type*       shared_memory,
                                                   unsigned int        local_fft_id) {
            const unsigned int tid  = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx = tid / blockDim.y;
            const unsigned int tidy = tid % blockDim.y;
            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset_strided(tidy);
            const unsigned int bid = batch_id(tidy);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride     = Stride * base_type::stride_size();
            unsigned int       index      = offset + (tidx * Stride);
            unsigned int       smem_index = tidx + tidy * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * base_type::stride_size() + tidx) < cufftdx::size_of<FFT>::value) {
                    if(bid < Batches) {
                        shared_memory[smem_index] = input[index];
                    }
                    index += stride;
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
            __syncthreads();
            smem_index = threadIdx.x + threadIdx.y * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * base_type::stride_size() + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    thread_data[i] = shared_memory[smem_index];
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
        }

        template<unsigned int Stride, unsigned int Batches = Stride>
        static inline __device__ void store_strided(const complex_type* thread_data,
                                                    complex_type*       shared_memory,
                                                    complex_type*       output,
                                                    unsigned int        local_fft_id) {
            __syncthreads();
            unsigned int smem_index = threadIdx.x + threadIdx.y * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * base_type::stride_size() + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    shared_memory[smem_index] = thread_data[i];
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
            __syncthreads();
            const unsigned int tid    = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx   = tid / blockDim.y;
            const unsigned int tidy   = tid % blockDim.y;
            const unsigned int offset = batch_offset_strided(tidy);
            const unsigned int bid = batch_id(tidy);
            const unsigned int stride = Stride * base_type::stride_size();
            unsigned int       index  = offset + (tidx * Stride);
            smem_index                = tidx + tidy * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * base_type::stride_size() + tidx) < cufftdx::size_of<FFT>::value) {
                    if(bid < Batches) {
                        output[index] = shared_memory[smem_index];
                    }
                    index += stride;
                    smem_index += (blockDim.x * blockDim.y);
                }
            }
        }
    };
} // namespace example

#endif // CUFFTDX_EXAMPLE_BLOCK_IO_STRIDED_HPP
