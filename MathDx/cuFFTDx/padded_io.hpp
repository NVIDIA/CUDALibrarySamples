#ifndef CUFFTDX_EXAMPLE_PADDED_IO_HPP_
#define CUFFTDX_EXAMPLE_PADDED_IO_HPP_

#include <cufft.h>

#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include "block_io.hpp"
#include "common.hpp"

namespace example {


    template<typename FFT, unsigned int SignalLength>
    struct io_padded {
        static inline __device__ unsigned int batch_offset(unsigned int local_fft_id) {
            // Implicit batching is currently mandatory for __half precision, and it forces two
            // batches of data to be put together into a single complex __half2 value. This makes
            // it so a "single" batch of complex __half2 values in reality contains 2 batches of
            // complex __half values. Full reference can be found in documentation:
            // https://docs.nvidia.com/cuda/cufftdx/api/methods.html#half-precision-implicit-batching
            unsigned int global_fft_id = blockIdx.x * (FFT::ffts_per_block / FFT::implicit_type_batching) + local_fft_id;
            return SignalLength * global_fft_id;
        }

        // If InputInRRIILayout is false, then function assumes that values in input are in RIRI
        // layout, and before loading them to thread_data they are converted to RRII layout.
        // Otherwise, if InputInRRIILayout is true, then function assumes values in input are in RRII
        // layout, and don't need to be converted before loading to thread_data.
        template<bool InputInRRIILayout = false, typename RegisterType, typename IOType, typename LoadOp = example::identity>
        static inline __device__ void
            load(const IOType* input,
                 RegisterType* thread_data,
                 unsigned int  local_fft_id,
                 LoadOp op = {}) {
            using input_t                                       = typename FFT::input_type;
            using complex_type                                  = typename FFT::value_type;

            // Calculate global offset of FFT batch
            const unsigned int offset = batch_offset(local_fft_id);
            constexpr auto inner_loop_limit = sizeof(input_t) / sizeof(IOType);

            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = FFT::stride;
            unsigned int       index  = offset + threadIdx.x * inner_loop_limit;

            for (unsigned int i = 0; i < FFT::input_ept; i++) {
                for(unsigned int j = 0; j < inner_loop_limit; ++j) {
                    if ((i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit) < SignalLength) {
                            reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] = op(reinterpret_cast<const IOType*>(input)[index + j]);
                    } else {
                        reinterpret_cast<IOType*>(thread_data)[i * inner_loop_limit + j] = get_zero<typename complex_type::value_type>();
                    }
                }
                index += inner_loop_limit * stride;
            }
        }

        // Function assumes that values in thread_data are in RRII layout.
        // If OutputInRRIILayout is false, values are saved into output in RIRI layout; otherwise - in RRII.
        template<bool OutputInRRIILayout = false, typename RegisterType, typename IOType, typename StoreOp = example::identity>
        static inline __device__ void
            store(const RegisterType* thread_data,
                  IOType*             output,
                  unsigned int        local_fft_id,
                  StoreOp op = {}) {
            using output_t                                       = typename FFT::output_type;
            constexpr auto inner_loop_limit = sizeof(output_t) / sizeof(IOType);

            const unsigned int offset = batch_offset(local_fft_id);
            const unsigned int stride = FFT::stride;
            unsigned int       index  = offset + threadIdx.x * inner_loop_limit;

            for (int i = 0; i < FFT::output_ept; ++i) {
                for (int j = 0; j < inner_loop_limit; ++j) {
                    if (i * stride * inner_loop_limit + j + threadIdx.x * inner_loop_limit < SignalLength) {
                            reinterpret_cast<IOType*>(output)[index + j] = op(reinterpret_cast<const IOType*>(thread_data)[i * inner_loop_limit + j]);
                    }
                }
                index += inner_loop_limit * stride;
            }
        }
    };

}

#endif
