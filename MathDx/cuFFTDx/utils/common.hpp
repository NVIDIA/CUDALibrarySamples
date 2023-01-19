// Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
//
// NOTICE TO LICENSEE:
//
// This source code and/or documentation ("Licensed Deliverables") are subject to
// NVIDIA intellectual property rights under U.S. and international Copyright
// laws.
//
// These Licensed Deliverables contained herein is PROPRIETARY and CONFIDENTIAL
// to NVIDIA and is being provided under the terms and conditions of a form of
// NVIDIA software license agreement by and between NVIDIA and Licensee ("License
// Agreement") or electronically accepted by Licensee.  Notwithstanding any terms
// or conditions to the contrary in the License Agreement, reproduction or
// disclosure of the Licensed Deliverables to any third party without the express
// written consent of NVIDIA is prohibited.
//
// NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
// AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THESE
// LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS
// OR IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD
// TO THESE LICENSED DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
// NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
// AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT,
// INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM
// LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
// OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
// PERFORMANCE OF THESE LICENSED DELIVERABLES.
//
// U.S. Government End Users.  These Licensed Deliverables are a "commercial
// item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting of
// "commercial computer software" and "commercial computer software
// documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) and is
// provided to the U.S. Government only as a commercial end item.  Consistent
// with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),
// all U.S. Government End Users acquire the Licensed Deliverables with only
// those rights set forth herein.
//
// Any use of the Licensed Deliverables in individual and commercial software
// must include, in the user documentation and internal comments to the code, the
// above Disclaimer and U.S. Government End Users Notice.

#ifndef MATHDX_CUFFTDX_EXAMPLE_COMMON_HPP
#define MATHDX_CUFFTDX_EXAMPLE_COMMON_HPP

#include <algorithm>
#include <random>
#include <vector>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cufftdx.hpp>

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

namespace example {
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

#endif // MATHDX_CUFFTDX_EXAMPLE_COMMON_HPP
