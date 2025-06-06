#ifndef CUSOLVERDX_EXAMPLE_COMMON_HPP_
#define CUSOLVERDX_EXAMPLE_COMMON_HPP_

#include <type_traits>
#include <vector>
#include <random>
#include <iostream>

#ifndef CUSOLVERDX_EXAMPLE_NVRTC
#    include <cuda/std/complex>
#    include <cusolverdx.hpp>
#endif

// the nvcc bug in CUDA 12.2-12.4, fixed in 12.5
#ifdef __NVCC__
#    if (__CUDACC_VER_MAJOR__ == 12 && (__CUDACC_VER_MINOR__ >= 2 && __CUDACC_VER_MINOR__ <= 5))
#        define CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND 1
#    endif
#endif

namespace example {
    // Used when CUSOLVERDX_EXAMPLE_DETAIL_NVCC_12_2_BUG_WORKAROUND is defined
    template<typename T>
    using a_data_type_t = typename T::a_data_type;

    template<typename T>
    using a_cuda_data_type_t = typename T::a_cuda_data_type;
} // namespace example

#endif
