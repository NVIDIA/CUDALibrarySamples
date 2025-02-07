#ifndef CURANDDX_EXAMPLE_COMMON_HPP_
#define CURANDDX_EXAMPLE_COMMON_HPP_

#include <type_traits>
#include <vector>
#include <random>

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

#ifndef CURAND_CHECK_AND_EXIT
#    define CURAND_CHECK_AND_EXIT(error)                                                \
        {                                                                               \
            auto status = static_cast<curandStatus_t>(error);                           \
            if (status != CURAND_STATUS_SUCCESS) {                                      \
                std::cout << status << " " << __FILE__ << ":" << __LINE__ << std::endl; \
                std::exit(status);                                                      \
            }                                                                           \
        }
#endif // CURAND_CHECK

namespace example {

    struct measure {
        // Returns execution time in ms.
        template<typename Kernel>
        static float execution(Kernel&&           kernel,
                               const unsigned int warm_up_runs,
                               const unsigned int runs,
                               cudaStream_t       stream) {
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
// All SM supported by cuRANDDx
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_70
            case 700: return Functor<700>()();
#endif
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_72
            case 720: return Functor<720>()();
#endif
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_75
            case 750: return Functor<750>()();
#endif
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_80
            case 800: return Functor<800>()();
#endif
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_86
            case 860: return Functor<860>()();
#endif
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_87
            case 870: return Functor<870>()();
#endif
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_89
            case 890: return Functor<890>()();
#endif
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_90
            case 900: return Functor<900>()();
#endif
#ifdef CURANDDX_EXAMPLE_ENABLE_SM_90
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
