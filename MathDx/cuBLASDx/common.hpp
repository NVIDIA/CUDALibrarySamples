#ifndef CUBLASDX_EXAMPLE_COMMON_HPP_
#define CUBLASDX_EXAMPLE_COMMON_HPP_

#include <type_traits>
#include <vector>
#include <random>

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <cublas_v2.h>

#ifndef CUBLASDX_EXAMPLE_NVRTC
#include <cuda/std/complex>
#endif

#ifndef CUBLASDX_EXAMPLE_NVRTC
#include <cublasdx.hpp>
#include <cuda_fp16.h>
#endif

#ifdef __NVCC__
#    if (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2)
#        define CUBLASDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND 1
#    endif
#endif

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
#    define CUBLAS_CHECK_AND_EXIT(error)                                                \
        {                                                                               \
            auto status = static_cast<cublasStatus_t>(error);                           \
            if (status != CUBLAS_STATUS_SUCCESS) {                                      \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CUBLAS_CHECK_AND_EXIT

namespace example {
    template <typename T>
    using value_type_t = typename T::value_type;

    #ifndef CUBLASDX_EXAMPLE_NVRTC
    template<typename T>
    constexpr bool is_complex() {
        return std::is_same_v<T, cublasdx::complex<__half>> || std::is_same_v<T, cublasdx::complex<float>> ||
               std::is_same_v<T, cublasdx::complex<double>> || std::is_same_v<T, cuda::std::complex<__half>> ||
               std::is_same_v<T, cuda::std::complex<float>> || std::is_same_v<T, cuda::std::complex<double>>;
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

    template<typename T1, typename T2>
    constexpr T1 convert(T2 v) {
        if constexpr (is_complex<T1>() && is_complex<T2>()) {
            return T1(v.real(), v.imag());
        } else if constexpr (is_complex<T1>()) {
            return T1(v, v);
        } else if constexpr (is_complex<T2>()) {
            return v.real();
        } else {
            return T1(v);
        }
    }

    // Create a complex or real number with the specified precision from a pair of floats.
    template <typename T>
    T make_value(float real, float imag=0.f) {
        if constexpr (example::is_complex<T>()) {
            return {real, imag};
        }
        else {
            return {real};
        }
    }

    template <typename T>
    double gemm_flops(unsigned int m, unsigned int n, unsigned int k) {
        if constexpr (example::is_complex<T>()) {
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
        else if constexpr (std::is_same_v<value_type, float>) {
            return "float";
        }
        else if constexpr (std::is_same_v<value_type, double>) {
            return "double";
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
    std::enable_if_t<!is_complex<T>(), std::vector<T>> get_random_data(const float  min,
                                                                       const float  max,
                                                                       const size_t size) {
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        std::vector<T> ret(size);
        for (auto& v : ret) {
            v = convert<T>(dist(gen));
        }
        return ret;
    }

    template<typename T>
    std::enable_if_t<is_complex<T>(), std::vector<T>> get_random_data(const float  min,
                                                                      const float  max,
                                                                      const size_t size) {
        std::random_device                    rd;
        std::mt19937                          gen(rd());
        std::uniform_real_distribution<float> dist(min, max);

        std::vector<T> ret(size);
        for (auto& v : ret) {
            using scalar_type = typename T::value_type;
            scalar_type r     = static_cast<scalar_type>(dist(gen));
            scalar_type i     = static_cast<scalar_type>(dist(gen));
            v                 = T(r, i);
        }
        return ret;
    }

    template <class ValueType, class Functor> __device__ __forceinline__
    void transform(ValueType *data, int size, Functor transformer) {
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            data[i] = transformer(i, data[i]);
        }
    }

    template <class ValueType> __device__ __forceinline__
    void set(ValueType *data, int size, ValueType value) {
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            data[i] = value;
        }
    }

    template <class ValueType> __device__  __forceinline__
    auto exp(ValueType value) {
        return cuda::std::exp(value);
    }

    __device__  __forceinline__
    auto exp(__half value) {
        return hexp(value);
    }

    #endif // CUBLASDX_EXAMPLE_NVRTC

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

    template<template<unsigned int> class Functor>
    inline int sm_runner() {
        // Get CUDA device compute capability
        const auto cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
// All SM supported by cuBLASDx
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_70
            case 700: return Functor<700>()();
#endif
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_72
            case 720: return Functor<720>()();
#endif
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_75
            case 750: return Functor<750>()();
#endif
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_80
            case 800: return Functor<800>()();
#endif
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_86
            case 860: return Functor<860>()();
#endif
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_87
            case 870: return Functor<870>()();
#endif
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_89
            case 890: return Functor<890>()();
#endif
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_90
            case 900: return Functor<900>()();
#endif
#ifdef CUBLASDX_EXAMPLE_ENABLE_SM_90
            default: {
                if (cuda_device_arch > 900) {
                    return Functor<900>()();
                }
            }
#endif
        }
        return 1;
    }
} // namespace example

#endif
