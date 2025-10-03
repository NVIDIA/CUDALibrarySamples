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

#ifndef CUBLASDX_EXAMPLE_COMMON_HPP_
#define CUBLASDX_EXAMPLE_COMMON_HPP_

#include <type_traits>
#include <vector>
#include <sstream>
#include <random>
#include <complex>
#include <algorithm>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cublas_v2.h>

#if !defined(CUBLASDX_EXAMPLE_NVRTC) && !defined(CUBLASDX_EXAMPLE_NO_THRUST)
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#endif

#ifndef CUBLASDX_EXAMPLE_NVRTC
#include <cuda/std/complex>
#endif

#ifndef CUBLASDX_EXAMPLE_NVRTC
#include <cublasdx.hpp>
#include <cuda_fp16.h>
#include "arch_runner.hpp"
#endif

#ifdef __NVCC__
#    if (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2)
#        define CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND 1
#    endif
#endif

#ifndef CUBLASDX_EXAMPLE_SUPPORTS_FP8
#   define CUBLASDX_EXAMPLE_SUPPORTS_FP8 ((__CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 8) || __CUDACC_VER_MAJOR__ >= 12)
#endif // CUBLASDX_EXAMPLE_SUPPORTS_FP8

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif

#ifndef CUFFT_CHECK_AND_EXIT
#    define CUFFT_CHECK_AND_EXIT(error)                                                 \
        {                                                                               \
            auto status = static_cast<cufftResult>(error);                              \
            if (status != CUFFT_SUCCESS) {                                              \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUFFT_CHECK_AND_EXIT

#ifndef CUBLAS_CHECK_AND_EXIT
#    define CUBLAS_CHECK_AND_EXIT(error)                                                    \
        {                                                                                   \
            auto status = static_cast<cublasStatus_t>(error);                               \
            if (status != CUBLAS_STATUS_SUCCESS) {                                          \
                if (status == CUBLAS_STATUS_NOT_SUPPORTED) {                                \
                    std::cout << "Config not supported by cuBLASLt, "                       \
                              << "please consult https://docs.nvidia.com/cuda/cublas/#id81 "\
                              << "for more detail on supported reference configurations"    \
                              << std::endl;                                                 \
                }                                                                           \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl;     \
                std::exit(status);                                                          \
            }                                                                               \
        }
#endif // CUBLAS_CHECK_AND_EXIT

namespace example {

    template<typename T>
    struct is_uniform_value_type {
        static constexpr bool value = std::is_same<typename T::a_value_type, typename T::b_value_type>::value &&
                                      std::is_same<typename T::a_value_type, typename T::c_value_type>::value;
    };

    template<typename T>
    struct uniform_value_type {
        static_assert(is_uniform_value_type<T>::value);
        using type = typename T::c_value_type;
    };

    template<typename T>
    using uniform_value_type_t = typename uniform_value_type<T>::type;

    template<typename T>
    using value_type_t = typename T::value_type;

    template <typename T>
    using a_value_type_t = typename T::a_value_type;

    template <typename T>
    using b_value_type_t = typename T::b_value_type;

    template <typename T>
    using c_value_type_t = typename T::c_value_type;

    inline unsigned int get_cuda_device_arch() {
        int device;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));

        int major = 0;
        int minor = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

        return static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
    }

    inline unsigned int get_multiprocessor_count(int device) {
        int multiprocessor_count = 0;
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
        return multiprocessor_count;
    }

    inline unsigned int get_multiprocessor_count() {
        int device = 0;
        CUDA_CHECK_AND_EXIT(cudaGetDevice(&device));
        return get_multiprocessor_count(device);
    }


    #ifndef CUBLASDX_EXAMPLE_NVRTC

    // Don't use thrust::device_vector to avoid unnecessary
    // device destructors (parallel_for CUDA errors in some Volta/Driver setups)
    template<typename T, typename Alloc = void>
    struct device_vector {
            T* _ptr;
            size_t _size = 0;

            device_vector() = default;

            device_vector(const std::vector<T>& other) {
                *this = other;
            }

            device_vector(const device_vector<T>& other) {
                *this = other;
            }

            template<typename U>
            device_vector(const U& other) {
                if constexpr(cute::is_integral_v<U>) {
                    _size = other;
                    CUDA_CHECK_AND_EXIT(cudaMalloc(&_ptr, _size * sizeof(T)));
                } else {
                    *this = other;
                }
            }

            device_vector(device_vector<T>&& other) {
                *this = std::move(other);
            }

            operator std::vector<T>() const {
                std::vector<T> ret(_size);
                CUDA_CHECK_AND_EXIT(cudaMemcpy(ret.data(), _ptr, _size * sizeof(T), cudaMemcpyDeviceToHost));
                return ret;
            }

            device_vector& operator=(const std::vector<T>& other) {
                reset();
                _size = other.size();
                CUDA_CHECK_AND_EXIT(cudaMalloc(&_ptr, _size * sizeof(T)));
                CUDA_CHECK_AND_EXIT(cudaMemcpy(_ptr, other.data(), _size * sizeof(T), cudaMemcpyHostToDevice));
                return *this;
            }

            device_vector& operator=(const device_vector<T>& other) {
                reset();
                _size = other.size();
                CUDA_CHECK_AND_EXIT(cudaMalloc(&_ptr, _size * sizeof(T)));
                CUDA_CHECK_AND_EXIT(cudaMemcpy(_ptr, other.data(), _size * sizeof(T), cudaMemcpyDeviceToDevice));
                return *this;
            }

            device_vector& operator=(device_vector<T>&& other) {
                reset();
                std::swap(_size, other._size);
                std::swap(_ptr, other._ptr);
                return *this;
            }

            template<typename U>
            device_vector& operator=(const U& other) {
                static_assert(std::is_same_v<T, typename U::element_type>, "Source element type must match target element type");
                reset();
                _size = cute::cosize(other.layout());
                CUDA_CHECK_AND_EXIT(cudaMalloc(&_ptr, _size * sizeof(T)));
                CUDA_CHECK_AND_EXIT(cudaMemcpy(_ptr, other.data(), _size * sizeof(T), cudaMemcpyHostToDevice));
                return *this;
            }

            T      * begin()              { return _ptr; }
            T      * end()                { return _ptr + _size; }
            T      * data()         const { return _ptr; }
            T const* cbegin()       const { return _ptr; }
            T const* cend()         const { return _ptr + _size; }
            size_t   size()         const { return _size; }

            void reset() {
                if(_size != 0) {
                    _size = 0;
                    CUDA_CHECK_AND_EXIT(cudaFree(_ptr));
                }
            }

            ~device_vector() {
                reset();
            }
    };

    namespace detail {

        template<class T>
        struct is_complex_helper {
            static constexpr bool value = false;
        };

        template<class T>
        struct is_complex_helper<cublasdx::complex<T>> {
            static constexpr bool value = true;
        };

        template<class T>
        struct is_complex_helper<std::complex<T>> {
            static constexpr bool value = true;
        };

        template<class T>
        struct is_complex_helper<cuda::std::complex<T>> {
            static constexpr bool value = true;
        };

    } // namespace detail


    template<typename T>
    CUBLASDX_HOST_DEVICE
    constexpr bool is_complex() {
        return detail::is_complex_helper<T>::value;
    }

    template<typename T, typename Enable = void>
    struct get_precision;

    template<typename T>
    struct get_precision<T, std::enable_if_t<is_complex<T>()>> {
        using type = typename T::value_type;
    };

    template<typename T>
    struct get_precision<T, std::enable_if_t<!is_complex<T>()>> {
        using type = T;
    };

    namespace detail {

        template<class T, class = void>
        struct promote;

        template<class T>
        struct promote<T, std::enable_if_t<commondx::is_signed_integral_v<T> and
                                           not is_complex<T>()>> {
            using value_type = int64_t;
        };

        template<class T>
        struct promote<T, std::enable_if_t<commondx::is_unsigned_integral_v<T> and
                                           not is_complex<T>()>> {
            using value_type = uint64_t;
        };

        template<class T>
        struct promote<T, std::enable_if_t<commondx::is_floating_point_v<T> and
                                           not is_complex<T>()>> {
            using value_type = double;
        };

        template<class T, template<class> class Complex>
        struct promote<Complex<T>, std::enable_if_t<is_complex<Complex<T>>()>> {
            using promoted_internal = typename promote<T>::value_type;

            using value_type = cublasdx::complex<promoted_internal>;
        };

        template<class ValueType>
        using get_reference_value_type_t = typename promote<ValueType>::value_type;
    }

    template<typename T1, typename T2>
    CUBLASDX_HOST_DEVICE
    constexpr T1 convert(T2 v) {
        constexpr bool is_output_complex = cublasdx::detail::has_complex_interface_v<T1>;
        constexpr bool is_input_complex = cublasdx::detail::has_complex_interface_v<T2>;
        if constexpr (is_input_complex and is_output_complex) {
            using t1_vt = typename T1::value_type;
            return T1(convert<t1_vt>(v.real()), convert<t1_vt>(v.imag()));
        } else if constexpr (is_output_complex) {
            using t1_vt = typename T1::value_type;
            return T1(convert<t1_vt>(v), convert<t1_vt>(v));
        } else if constexpr (is_input_complex) {
            return convert<T1>(v.real());
        } else if constexpr (COMMONDX_STL_NAMESPACE::is_convertible_v<T2, T1>){
            return static_cast<T1>(v);
        } else if constexpr (COMMONDX_STL_NAMESPACE::is_constructible_v<T1, T2>){
            return T1(v);
        } else {
            static_assert(COMMONDX_STL_NAMESPACE::is_convertible_v<T2, T1>, "Please provide your own conversion function");
        }
    }

    template<typename T>
    struct converter {
        template<class V>
        CUBLASDX_HOST_DEVICE constexpr
        T operator()(V const& v) const { return convert<T>(v); }
    };

    template<typename Layout>
    constexpr CUBLASDX_HOST_DEVICE auto
    swap_layout_modes(const Layout& l) {
        if constexpr(cute::rank(Layout{}) == 2) {
            return cute::select<1, 0>(l);
        } else if constexpr(cute::rank(Layout{}) == 3) {
            return cute::select<0, 2, 1>(l);
        } else {
            static_assert(cute::rank(Layout{}) > 3, "Unsupported layout rank");
        }
    }

    template<typename Swizzle, typename Offset, typename Layout>
    constexpr CUBLASDX_HOST_DEVICE auto
    swap_layout_modes(const cute::ComposedLayout<Swizzle, Offset, Layout>& l) {
        return cute::composition(l.layout_a(),
            l.offset(),
            swap_layout_modes(l.layout_b()));
    }

    template<typename T, typename Layout>
    constexpr CUBLASDX_HOST_DEVICE auto
    swap_tensor_modes(const cute::Tensor<T, Layout>& t) {
        return cute::make_tensor(t.data(), swap_layout_modes(t.layout()));
    }

    // device_gemm_performance utilities
    template<class GEMMArr>
    CUBLASDX_HOST_DEVICE
    auto get_block_coord(GEMMArr) {
        static_assert(cute::is_static<GEMMArr>::value, "Arrangement must be static");

        constexpr cublasdx::arrangement arr_a = cute::get<0>(GEMMArr{});
        constexpr cublasdx::arrangement arr_b = cute::get<1>(GEMMArr{});

        constexpr bool is_a_global_col = arr_a == cublasdx::col_major;
        constexpr bool is_b_global_col = arr_b == cublasdx::col_major;

        // Handle row-major symmetrically to col-major
        constexpr bool reverse_block_coord = not is_a_global_col and not is_b_global_col;
        return cute::conditional_return<reverse_block_coord>
            (cute::make_coord(blockIdx.y, blockIdx.x, cute::_), cute::make_coord(blockIdx.x, blockIdx.y, cute::_));
    }

    template<class Precision, cublasdx::type Type>
    struct get_value_type {
        using type = std::conditional_t<Type == cublasdx::type::real,
                                        Precision,
                                        cublasdx::complex<Precision>>;
    };

    template<class Precision, cublasdx::type Type>
    using get_value_type_t = typename get_value_type<Precision, Type>::type;

    template<bool Condition, class FirstElem, class SecondElem>
    CUBLASDX_HOST_DEVICE
    auto swap_if(FirstElem const& first_elem, SecondElem const& second_elem) {
        if constexpr(Condition) {
            return cute::make_tuple(second_elem, first_elem);
        } else {
            return cute::make_tuple(first_elem, second_elem);
        }

        CUTE_GCC_UNREACHABLE;
    }

    template<class GEMMShape, class GEMMArr, class GEMMLD, class AIter, class BIter, class CIter>
    CUBLASDX_HOST_DEVICE
    auto make_global_tensors(GEMMShape const& gemm_shape, GEMMArr, GEMMLD const& gemm_ld,
                             AIter a_iter, BIter b_iter, CIter c_iter) {

        static_assert(cute::is_static<GEMMArr>::value, "Arrangement must be static");

        constexpr cublasdx::arrangement arr_a = cute::get<0>(GEMMArr{});
        const auto stride_a = swap_if<arr_a == cublasdx::row_major>(cute::_1{}, cute::get<0>(gemm_ld));

        constexpr cublasdx::arrangement arr_b = cute::get<1>(GEMMArr{});
        const auto stride_b = swap_if<arr_b == cublasdx::row_major>(cute::_1{}, cute::get<1>(gemm_ld));

        constexpr cublasdx::arrangement arr_c = cute::get<2>(GEMMArr{});
        const auto stride_c = swap_if<arr_c == cublasdx::row_major>(cute::_1{}, cute::get<2>(gemm_ld));

        return cute::make_tuple(
            cublasdx::make_tensor(a_iter, cute::make_layout(cute::select<0, 2>(gemm_shape), stride_a)),
            cublasdx::make_tensor(b_iter, cute::make_layout(cute::select<2, 1>(gemm_shape), stride_b)),
            cublasdx::make_tensor(c_iter, cute::make_layout(cute::select<0, 1>(gemm_shape), stride_c)));
    }

    template<class GEMMShape, class GEMMArr, class GEMMLD, class TA, class TB, class TC>
    CUBLASDX_HOST_DEVICE
    auto make_device_global_tensors(GEMMShape const& gemm_shape, GEMMArr gemm_arr, GEMMLD const& gemm_ld,
                                    TA* a_ptr, TB* b_ptr, TC* c_ptr) {
        return make_global_tensors(gemm_shape, gemm_arr, gemm_ld, cute::make_gmem_ptr(a_ptr), cute::make_gmem_ptr(b_ptr), cute::make_gmem_ptr(c_ptr));
    }

    template<class BLAS, class ATensor, class BlockCoord>
    CUBLASDX_HOST_DEVICE
    auto get_block_tile_slice_a(ATensor    const& a_tensor,
                                BlockCoord const& block_coord) {
        constexpr auto tile_m = cublasdx::size_of<BLAS>::m;
        constexpr auto tile_k = cublasdx::size_of<BLAS>::k;
        return cute::local_tile(a_tensor, cute::make_shape(cute::Int<tile_m>{}, cute::Int<tile_k>{}), cute::select<0, 2>(block_coord));
    }

    template<class BLAS, class BTensor, class BlockCoord>
    CUBLASDX_HOST_DEVICE
    auto get_block_tile_slice_b(BTensor    const& b_tensor,
                                BlockCoord const& block_coord) {
        constexpr auto tile_k = cublasdx::size_of<BLAS>::k;
        constexpr auto tile_n = cublasdx::size_of<BLAS>::n;
        return cute::local_tile(b_tensor, cute::make_shape(cute::Int<tile_k>{}, cute::Int<tile_n>{}), cute::select<2, 1>(block_coord));
    }

    template<class BLAS, class CTensor, class BlockCoord>
    CUBLASDX_HOST_DEVICE
    auto get_block_tile_c(CTensor         & c_tensor,
                          BlockCoord const& block_coord) {
        constexpr auto tile_m = cublasdx::size_of<BLAS>::m;
        constexpr auto tile_n = cublasdx::size_of<BLAS>::n;
        return cute::local_tile(c_tensor, cute::make_shape(cute::Int<tile_m>{}, cute::Int<tile_n>{}), cute::select<0, 1>(block_coord));
    }

    template<class Stage, class Tensor>
    CUBLASDX_HOST_DEVICE
    auto get_tile_from_slice(Tensor & tensor, Stage const& stage) {
        // We assume a local_tile partition, where the first 2 dimensions
        // are tile itself, and the third is iteration dimension
        return tensor(cute::_, cute::_, stage);
    }

    // end of device_gemm_performance utilities

    // This assumed no customized leading dimension
    template<typename BLAS>
    struct global_memory_size_of {
        static constexpr unsigned int m = cublasdx::size_of<BLAS>::m;
        static constexpr unsigned int n = cublasdx::size_of<BLAS>::n;
        static constexpr unsigned int k = cublasdx::size_of<BLAS>::k;

        static constexpr unsigned int a_size = m * k;
        static constexpr unsigned int b_size = k * n;
        static constexpr unsigned int c_size = m * n;
    };

    // Create a complex or real number with the specified precision from a pair of floats.
    template <typename T>
    T make_value(float real, float imag=0.f) {
        if constexpr (example::is_complex<T>()) {
            return {real, imag};
        }
        else {
            return T(real);
        }
    }

    template <typename TA, typename TB = TA, typename TC = TA>
    double gemm_flops(unsigned int m, unsigned int n, unsigned int k) {
        static_assert( (  example::is_complex<TA>() &&  example::is_complex<TB>() &&  example::is_complex<TC>() ) ||
                       ( !example::is_complex<TA>() && !example::is_complex<TB>() && !example::is_complex<TC>() ) );
        if constexpr (example::is_complex<TA>()) {
            return 8. * m * n * k;
        }
        else {
            return 2. * m * n * k;
        }
    }

    template <typename T>
    std::string type_string() {
        if constexpr (example::is_complex<T>()) {
            return "complex";
        }
        else {
            return "real";
        }
    }

    template <typename T>
    std::string precision_string() {
        using value_type = typename get_precision<T>::type;
        if constexpr (std::is_same_v<value_type, __half>) {
            return "half";
        }
        else if constexpr (std::is_same_v<value_type, __nv_bfloat16>) {
            return "bfloat16";
        }
        #if CUBLASDX_EXAMPLE_SUPPORTS_FP8
        else if constexpr (std::is_same_v<value_type, __nv_fp8_e4m3>) {
            return "fp8_e4m3";
        }
        else if constexpr (std::is_same_v<value_type, __nv_fp8_e5m2>) {
            return "fp8_e4m3";
        }
        #endif
        else if constexpr (std::is_same_v<value_type, cublasdx::tfloat32_t>) {
            return "tfloat32";
        }
        else if constexpr (std::is_same_v<value_type, float>) {
            return "float";
        }
        else if constexpr (std::is_same_v<value_type, double>) {
            return "double";
        }
        else if constexpr (std::is_same_v<value_type, int8_t>) {
            return "int8";
        }
        else if constexpr (std::is_same_v<value_type, uint8_t>) {
            return "uint8";
        }
        else if constexpr (std::is_same_v<value_type, int32_t>) {
            return "int32";
        }
        else {
            return "unsupported";
        }
    }

    struct measure {
        // Returns execution time in ms.
        template<typename Kernel>
        static float execution(Kernel&& kernel, const unsigned int warm_up_runs, const unsigned int runs, cudaStream_t stream) {
            cudaEvent_t startEvent, stopEvent;
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            for (unsigned int i = 0; i < warm_up_runs; i++) {
                kernel(stream);
            }

            CUDA_CHECK_AND_EXIT(cudaGetLastError());
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
            for (unsigned int i = 0; i < runs; i++) {
                kernel(stream);
            }
            CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent, stream));
            CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

            float time;
            CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
            CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
            return time;
        }
    };

    namespace detail {
        template<typename T>
        double cbabs(T v) {
            if constexpr (is_complex<T>()) {
                auto imag = std::abs(static_cast<double>(v.imag()));
                auto real = std::abs(static_cast<double>(v.real()));
                return (real + imag) / 2.0;
            } else {
                return std::abs(static_cast<double>(v));
            }
        }
    } // namespace detail

    template<typename T>
    std::vector<T> get_random_data(const float  min,
                                   const float  max,
                                   const size_t size,
                                   const int    seed = -1) {
        std::random_device                    rd;
        std::mt19937                          gen((seed != -1) ? seed : rd());
        using gen_t = std::conditional_t<commondx::is_floating_point_v<T>, float, int>;
        using dist_t = std::conditional_t<commondx::is_floating_point_v<T>,
                                          std::uniform_real_distribution<float>,
                                          std::uniform_int_distribution<int>>;

        dist_t dist(static_cast<gen_t>(min), static_cast<gen_t>(max));

        std::vector<T> ret(size);

        for (size_t v = 0; v < ret.size(); ++v) {
            if constexpr(is_complex<T>()) {
                using scalar_type = typename T::value_type;
                scalar_type r     = static_cast<scalar_type>(dist(gen));
                scalar_type i     = static_cast<scalar_type>(dist(gen));
                ret[v]            = T(r, i);
            } else {
                ret[v] = convert<T>(dist(gen));
            }
        }
        return ret;
    }

    template<typename Tin, typename Tout>
    std::vector<Tout> convert(const std::vector<Tin>& input) {
        std::vector<Tout> output;
        for(auto v: input) {
            output.push_back(Tout(v));
        }
        return output;
    }

    template <class ValueType, class Functor> CUBLASDX_DEVICE
    void transform(ValueType *data, int size, Functor transformer) {
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            data[i] = transformer(i, data[i]);
        }
    }

    template <class ValueType> CUBLASDX_DEVICE
    void set(ValueType *data, int size, ValueType value) {
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            data[i] = value;
        }
    }

    template <class ValueType> CUBLASDX_DEVICE
    auto exp(ValueType value) {
        return cuda::std::exp(value);
    }

    CUBLASDX_DEVICE
    auto exp(__half value) {
        return hexp(value);
    }

    template<class T1, class T2>
    CUBLASDX_HOST_DEVICE
    void swap(T1& v1, T2& v2) {
        auto tmp = v1;
        v1 = v2;
        v2 = tmp;
    }

    struct cublasdx_enable_example_sm {
        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_70)
        static constexpr bool sm_70 = true;
        #else
        static constexpr bool sm_70 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_72)
        static constexpr bool sm_72 = true;
        #else
        static constexpr bool sm_72 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_75)
        static constexpr bool sm_75 = true;
        #else
        static constexpr bool sm_75 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_80)
        static constexpr bool sm_80 = true;
        #else
        static constexpr bool sm_80 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_86)
        static constexpr bool sm_86 = true;
        #else
        static constexpr bool sm_86 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_87)
        static constexpr bool sm_87 = true;
        #else
        static constexpr bool sm_87 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_89)
        static constexpr bool sm_89 = true;
        #else
        static constexpr bool sm_89 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_90)
        static constexpr bool sm_90 = true;
        #else
        static constexpr bool sm_90 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_100)
        static constexpr bool sm_100 = true;
        #else
        static constexpr bool sm_100 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_101) && (CUDA_VERSION < 13000)
        static constexpr bool sm_101 = true;
        #else
        static constexpr bool sm_101 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_103)
        static constexpr bool sm_103 = true;
        #else
        static constexpr bool sm_103 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_110) && (CUDA_VERSION >= 13000)
        static constexpr bool sm_110 = true;
        #else
        static constexpr bool sm_110 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_120)
        static constexpr bool sm_120 = true;
        #else
        static constexpr bool sm_120 = false;
        #endif

        #if defined(CUBLASDX_EXAMPLE_ENABLE_SM_121)
        static constexpr bool sm_121 = true;
        #else
        static constexpr bool sm_121 = false;
        #endif
        
    };

    template<class Functor, class ... Args>
    auto sm_runner(Functor functor, Args&& ... args) {
        auto cuda_device_arch = get_cuda_device_arch();
        return arch_runner<cublasdx_enable_example_sm, int>(cuda_device_arch, functor, static_cast<Args&&>(args)...);
    }

    #endif // CUBLASDX_EXAMPLE_NVRTC

} // namespace example

#endif