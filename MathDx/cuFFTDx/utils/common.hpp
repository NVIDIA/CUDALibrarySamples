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

#ifndef MATHDX_CUFFTDX_EXAMPLE_COMMON_HPP_
#define MATHDX_CUFFTDX_EXAMPLE_COMMON_HPP_

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

namespace example {
    template<class T>
    inline auto get_random_complex_data(size_t size, T min, T max) ->
        typename std::enable_if<std::is_floating_point<T>::value,
                                std::vector<cufftdx::make_complex_type_t<T>>>::type {
        using complex_type = cufftdx::make_complex_type_t<T>;
        std::random_device                rd;
        std::default_random_engine        gen(rd());
        std::uniform_real_distribution<T> distribution(min, max);
        std::vector<complex_type>         output(size);
        std::generate(output.begin(), output.end(), [&]() {
            return complex_type {distribution(gen), distribution(gen)};
        });
        return output;
    }

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
#ifdef SM_70
            case 700: Functor<700>()(); return 0;
#endif
#ifdef SM_72
            case 720: Functor<720>()(); return 0;
#endif
#ifdef SM_75
            case 750: Functor<750>()(); return 0;
#endif
#ifdef SM_80
            case 800: Functor<800>()(); return 0;
#endif
#ifdef SM_86
            case 860: Functor<860>()(); return 0;
#endif
            default: {
                if (cuda_device_arch > 860) {
#ifdef SM_86
                    Functor<860>()();
                    return 0;
#else
                    return 1;
#endif
                }
            }
        }
        return 1;
    }
} // namespace example

#endif // CUFFTDX_EXAMPLE_COMMON_HPP_
