#ifndef CUSOLVERDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP
#define CUSOLVERDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP

#include "macros.hpp"
#include <cusolverdx.hpp>

namespace common {
    // This function enables creating architecture agnostic examples
    // and functions while avoid compilation overhead, by compiling
    // only the enabled branches and then based on runtime CUDA compute
    // capability dispatching with appropriate argument.
    //
    // Functor is example function which takes static integer type as
    // its argument. Then the example can read this value and use it
    // for its SM<Val>() operator.
    template<template<int> class Functor>
    inline int run_example_with_sm() {
        // Get CUDA device compute capability
        const auto cuda_device_arch = get_cuda_device_arch();

        switch (cuda_device_arch) {
// All SM supported by cuSOLVERDx
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_70
            case 700: return Functor<700>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_72
            case 720: return Functor<720>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_75
            case 750: return Functor<750>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_80
            case 800: return Functor<800>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_86
            case 860: return Functor<860>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_87
            case 870: return Functor<870>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_89
            case 890: return Functor<890>()();
#endif
#ifdef CUSOLVERDX_EXAMPLE_ENABLE_SM_90
            case 900: return Functor<900>()();
#endif
            default: {
                // Fail
                return 1;
            }
        }
    }
} // namespace common

#endif // CUSOLVERDX_EXAMPLE_COMMON_EXAMPLE_SM_RUNNER_HPP
