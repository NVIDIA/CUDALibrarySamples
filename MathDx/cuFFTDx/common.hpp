#ifndef CUFFTDX_EXAMPLE_COMMON_HPP_
#define CUFFTDX_EXAMPLE_COMMON_HPP_

#include <vector>
#include <tuple>
#include <cmath>
#include <chrono>

#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK_AND_EXIT
#    define CUDA_CHECK_AND_EXIT(error)                                                                      \
        {                                                                                                   \
            auto status = static_cast<cudaError_t>(error);                                                  \
            if (status != cudaSuccess) {                                                                    \
                std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                                          \
            }                                                                                               \
        }
#endif // CUDA_CHECK_AND_EXIT

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

#ifdef __NVCC__
#    if (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 2)
#        define CUFFTDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND 1
#    endif
#endif

namespace example {
    template <typename T>
    using value_type_t = typename T::value_type;

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

    struct fft_signal_error {
        double l2_relative_error;
        double peak_error;
        double peak_error_relative;
        size_t peak_error_index;

        template<class T, class K>
        static inline fft_signal_error calculate_for_complex_values(const std::vector<T>& results, const std::vector<K>& reference) {
            fft_signal_error error {0.0, 0.0, 0.0, 0};
            double           nerror = 0.0;
            double           derror = 0.0;
            for (size_t i = 0; i < results.size(); i++) {
                calculate_for_real_value(results[i].x, reference[i].x, error, i, nerror, derror);
                calculate_for_real_value(results[i].y, reference[i].y, error, i, nerror, derror);
            }
            error.l2_relative_error = std::sqrt(nerror) / std::sqrt(derror);
            return error;
        }

        template<class T, class K>
        static inline fft_signal_error calculate_for_real_values(const std::vector<T>& results, const std::vector<K>& reference) {
            fft_signal_error error {0.0, 0.0, 0.0, 0};
            double           nerror = 0.0;
            double           derror = 0.0;
            for (size_t i = 0; i < results.size(); i++) {
                calculate_for_real_value(results[i], reference[i], error, i, nerror, derror);
            }
            error.l2_relative_error = std::sqrt(nerror) / std::sqrt(derror);
            return error;
        }

    private:
        template<class T, class K>
        static inline void calculate_for_real_value(const T&          results_value,
                                                    const K&          reference_value,
                                                    fft_signal_error& error,
                                                    const size_t      i,
                                                    double&           nerror,
                                                    double&           derror) {
            double serr = std::fabs(results_value - reference_value);
            if (serr > error.peak_error) {
                error.peak_error          = serr;
                error.peak_error_relative = std::fabs(serr / reference_value);
                error.peak_error_index    = i;
            }
            nerror += std::pow(serr, 2);
            derror += std::pow(results_value, 2);
        }
    };

    // Returns execution time in ms
    template<typename Kernel>
    float measure_execution_ms(Kernel&& kernel, const unsigned int warm_up_runs, const unsigned int runs, cudaStream_t stream) {
        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
        CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        for (size_t i = 0; i < warm_up_runs; i++) {
            kernel(stream);
        }
        CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

        CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent, stream));
        for (size_t i = 0; i < runs; i++) {
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

    template<typename Function>
    float measure_host_ms(Function&& kernel) {
        auto t1 = std::chrono::high_resolution_clock::now();
        kernel();
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> ms_float = t2 - t1;
        return ms_float.count();
    }

    template<class T>
    struct fft_results {
        std::vector<T> output;
        float avg_time_in_ms;
    };

    template<template<unsigned int> class Functor>
    inline int sm_runner() {
        // Get CUDA device compute capability
        const auto cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
// If examples are compiled via Makefile all cases are enabled, if via CMake only the SMs
// that are part of CUFFTDX_TARGET_ARCHS/CUFFTDX_CUDA_ARCHITECTURES are enabled.
#if !defined(CUFFTDX_EXAMPLE_CMAKE) || defined(CUFFTDX_EXAMPLE_ENABLE_SM_70)
            case 700: Functor<700>()(); return 0;
#endif
#if !defined(CUFFTDX_EXAMPLE_CMAKE) || defined(CUFFTDX_EXAMPLE_ENABLE_SM_72)
            case 720: Functor<720>()(); return 0;
#endif
#if !defined(CUFFTDX_EXAMPLE_CMAKE) || defined(CUFFTDX_EXAMPLE_ENABLE_SM_75)
            case 750: Functor<750>()(); return 0;
#endif
#if !defined(CUFFTDX_EXAMPLE_CMAKE) || defined(CUFFTDX_EXAMPLE_ENABLE_SM_80)
            case 800: Functor<800>()(); return 0;
#endif
#if !defined(CUFFTDX_EXAMPLE_CMAKE) || defined(CUFFTDX_EXAMPLE_ENABLE_SM_86)
            case 860: Functor<860>()(); return 0;
#endif
#if !defined(CUFFTDX_EXAMPLE_CMAKE) || defined(CUFFTDX_EXAMPLE_ENABLE_SM_87)
            case 870: Functor<870>()(); return 0;
#endif
#if !defined(CUFFTDX_EXAMPLE_CMAKE) || defined(CUFFTDX_EXAMPLE_ENABLE_SM_89)
            case 890: Functor<890>()(); return 0;
#endif
#if !defined(CUFFTDX_EXAMPLE_CMAKE) || defined(CUFFTDX_EXAMPLE_ENABLE_SM_90)
            case 900: Functor<900>()(); return 0;
#endif
        }
        return 1;
    }
} // namespace example

#endif // CUFFTDX_EXAMPLE_COMMON_HPP_
