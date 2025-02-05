#ifndef CUFFTDX_EXAMPLE_MIXED_IO_HPP_
#define CUFFTDX_EXAMPLE_MIXED_IO_HPP_

#include <type_traits>
#include <iostream>

#include <cufft.h>

#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include "block_io.hpp"

namespace example {

    namespace detail {

        template<typename T1, typename T1R, typename T2, typename T2R>
        struct are_same {
            constexpr static bool value = std::is_same_v<T1, T1R> && std::is_same_v<T2, T2R>;
        };

        template<typename T1, typename T1R, typename T2, typename T2R>
        inline constexpr bool are_same_v = are_same<T1, T1R, T2, T2R>::value;

        template<typename T, typename = void>
        struct has_value_type: std::false_type {
        };

        template<typename T>
        struct has_value_type<T, decltype((void)typename T::value_type(), void())>: std::true_type {
        };

        template<typename T>
        inline constexpr bool has_value_type_v = has_value_type<T>::value;

        template<typename T, typename = void>
        struct get_precision {
            using type = T;
        };

        template<typename T>
        struct get_precision<T, std::enable_if_t<has_value_type_v<T>, void>> {
            using type = typename T::value_type;
        };

    } // namespace detail

    // Type Helpers
    template<typename ComplexType>
    struct make_cufft_compatible {
        using type =
            std::conditional_t<std::is_same_v<typename ComplexType::value_type, float>,
                               cufftComplex,
                               cufftDoubleComplex>;

        static_assert(sizeof(ComplexType) == sizeof(type), "cuFFT incompatible complex type was provided, size requirement not fulfilled");
        static_assert(std::alignment_of_v<ComplexType> == std::alignment_of_v<type>, "cuFFT incompatible complex type was provided, alignment requirement not fulfilled");
    };

    // Conversion Utilities
    template<typename TargetPrecision, typename SourcePrecision>
    __host__ __device__ constexpr TargetPrecision
    convert_scalar(const SourcePrecision& v) {
        using TP = TargetPrecision;
        using SP = SourcePrecision;

        TargetPrecision ret {};

        if constexpr (std::is_same_v<TP, SP>) {
            ret = v;
        } else if constexpr (detail::are_same_v<TP, float, SP, __half>) {
            ret = __half2float(v);
        } else if constexpr (detail::are_same_v<TP, __half, SP, float>) {
            ret = __float2half(v);
        } else if constexpr (detail::are_same_v<TP, float, SP, __nv_bfloat16>) {
            ret = __bfloat162float(v);
        } else if constexpr (detail::are_same_v<TP, __nv_bfloat16, SP, float>) {
            ret = __float2bfloat16(v);
        } else {
            ret = static_cast<TP>(v);
        }

        return ret;
    }

    // Precision given as target argument
    // for complex Source Type
    template<typename TargetTypeOrPrecision, typename SourceType>
    __host__ __device__ constexpr auto
    convert(const SourceType& v) {
        constexpr bool is_source_complex = detail::has_value_type_v<SourceType>;
        using target_precision           = typename detail::get_precision<TargetTypeOrPrecision>::type;
        using converted_type             = std::conditional_t<is_source_complex,
                                                  cufftdx::complex<target_precision>,
                                                  TargetTypeOrPrecision>;

        converted_type ret {};

        if constexpr (is_source_complex) {
            ret = converted_type {convert_scalar<target_precision>(v.real()),
                                  convert_scalar<target_precision>(v.imag())};
        } else {
            ret = converted_type {convert_scalar<target_precision>(v)};
        }

        return ret;
    }

    template<typename ComputeType, typename IOType>
    struct mixed_precision;

    template<>
    struct mixed_precision<double, float> {
        using compute_type    = double;
        using io_type         = float;
        using complex_io_type = cufftdx::complex<float>;
    };

    template<>
    struct mixed_precision<float, __half> {
        using compute_type    = float;
        using io_type         = __half;
        using complex_io_type = cufftdx::complex<__half>;
    };

    template<>
    struct mixed_precision<float, __nv_bfloat16> {
        using compute_type    = float;
        using io_type         = __nv_bfloat16;
        using complex_io_type = cufftdx::complex<__nv_bfloat16>;
    };

    template<typename FFT>
    struct io_mixed {
        using complex_compute_type = typename FFT::value_type;
        using compute_precision    = typename cufftdx::precision_of<FFT>::type;

        // input - global input with all FFTs
        // thread_data - local thread array to load values from input to
        // local_fft_id - ID of FFT batch in CUDA block
        template<typename InputOutputType>
        static inline __device__ void load(const InputOutputType* input,
                                           complex_compute_type*  thread_data,
                                           unsigned int           local_fft_id) {
            using complex_type = typename FFT::value_type;
            // Calculate global offset of FFT batch
            const unsigned int offset = example::io<FFT>::input_batch_offset(local_fft_id);
            // Get stride, this shows how elements from batch should be split between threads
            const unsigned int stride = FFT::stride;
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if (i * stride + threadIdx.x < cufftdx::size_of<FFT>::value) {
                    thread_data[i] = convert<compute_precision>(input[index]);
                    index += stride;
                }
            }
        }

        template<typename InputOutputType>
        static inline __device__ void store(const complex_compute_type* thread_data,
                                            InputOutputType*            output,
                                            unsigned int                local_fft_id) {
            using io_precision = typename InputOutputType::value_type;

            const unsigned int offset = example::io<FFT>::output_batch_offset(local_fft_id);
            const unsigned int stride = FFT::stride;
            unsigned int       index  = offset + threadIdx.x;
            for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
                if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) {
                    output[index] = convert<io_precision>(thread_data[i]);
                }
                index += stride;
            }
        }

        template<typename InputOutputType>
        static inline __device__ void load(const InputOutputType* source,
                                           complex_compute_type*  target) {
            unsigned int n      = blockDim.y * cufftdx::size_of<FFT>::value;
            unsigned int stride = blockDim.x * blockDim.y;
            unsigned int index  = threadIdx.y * blockDim.x + threadIdx.x;
            for (int step = 0; step < FFT::elements_per_thread; step++) {
                if (index < n) {
                    target[index] = convert<compute_precision>(source[index]);
                }
                index += stride;
            }
            __syncthreads();
        }

        template<typename InputOutputType>
        static inline __device__ void store(const complex_compute_type* source,
                                            InputOutputType*            target) {
            using io_precision = typename InputOutputType::value_type;

            unsigned int n      = blockDim.y * cufftdx::size_of<FFT>::value;
            unsigned int stride = blockDim.x * blockDim.y;
            unsigned int index  = threadIdx.y * blockDim.x + threadIdx.x;

            __syncthreads();
            for (int step = 0; step < FFT::elements_per_thread; step++) {
                if (index < n) {
                    target[index] = convert<io_precision>(source[index]);
                }
                index += stride;
            }
        }
    };
} // namespace example

#endif
