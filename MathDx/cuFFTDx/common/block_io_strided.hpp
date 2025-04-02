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

        static inline __device__ unsigned int batch_id(unsigned int local_fft_id) {
            unsigned int global_fft_id = blockIdx.x * FFT::ffts_per_block + local_fft_id;
            return global_fft_id;
        }

        static inline __device__ unsigned int batch_offset_strided(unsigned int local_fft_id) {
            return batch_id(local_fft_id);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////
        // Note: loads / stores for 2d FFTs, based on stride and batches for specific dimension (inner most)
        template<unsigned int Stride, unsigned int Batches = Stride, typename InputOutputType>
        static inline __device__ void load_strided(const InputOutputType* input,
                                                   complex_type*          thread_data,
                                                   unsigned int           local_fft_id) {
            // Calculate global offset of FFT batch
            const unsigned int batch_offset = batch_offset_strided(local_fft_id);
            const unsigned int bid          = batch_id(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride       = Stride * FFT::stride;
            unsigned int       index        = batch_offset + (threadIdx.x * Stride);
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
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
                                                    unsigned int        local_fft_id) {
            const unsigned int batch_offset = batch_offset_strided(local_fft_id);
            const unsigned int bid          = batch_id(local_fft_id);
            const unsigned int stride       = Stride * FFT::stride;
            unsigned int       index        = batch_offset + (threadIdx.x * Stride);
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
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
                                                   unsigned int           local_fft_id) {
            const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx         = tid / blockDim.y;
            const unsigned int tidy         = tid % blockDim.y;
            // Calculate global offset of FFT batch
            const unsigned int batch_offset = batch_offset_strided(tidy);
            const unsigned int bid          = batch_id(tidy);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride       = Stride * FFT::stride;
            unsigned int       index        = batch_offset + (tidx * Stride);
            unsigned int       smem_index   = tidx + tidy * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
                    if (bid < Batches) {
                        shared_memory[smem_index] = input[index];
                    }
                    index += stride;
                    smem_index += (blockDim.x * blockDim.y);
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

        template<unsigned int Stride, unsigned int Batches = Stride, typename InputOutputType>
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
            const unsigned int tid          = threadIdx.x + blockDim.x * threadIdx.y;
            const unsigned int tidx         = tid / blockDim.y;
            const unsigned int tidy         = tid % blockDim.y;
            const unsigned int batch_offset = batch_offset_strided(tidy);
            const unsigned int bid          = batch_id(tidy);
            const unsigned int stride       = Stride * FFT::stride;
            unsigned int       index        = batch_offset + (tidx * Stride);
            smem_index                      = tidx + tidy * blockDim.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * FFT::stride + tidx) < cufftdx::size_of<FFT>::value) {
                    if (bid < Batches) {
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
