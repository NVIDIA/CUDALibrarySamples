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

#include <iostream>
#include <vector>

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

#ifndef CUDA_ARCH
#   define CUDA_ARCH 80
#endif

using namespace cufftdx;
using FFT = decltype(Size<1024>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>()
                     + Precision<float>() + Block() + SM<CUDA_ARCH * 10>());

template<class FFT>
__global__ void block_fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;

    // Local array for each thread
    complex_type thread_data[FFT::storage_size];

    // Id of FFT in CUDA block is identified by its thread index in 2nd dimension,
    // in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Calculate global offset for FFT in data
    const unsigned int offset = cufftdx::size_of<FFT>::value * (blockIdx.x * FFT::ffts_per_block + local_fft_id);

    // Load data from global memory (data) to registers (thread_data) as described in
    // https://docs.nvidia.com/cuda/cufftdx/api/methods.html#input-output-data-format
    {
        unsigned int index = offset + threadIdx.x;
        for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
            thread_data[i] = data[index];
            // FFT::stride shows how elements from a single FFT should be split between threads
            index += FFT::stride;
        }
    }

    // Shared memory required by FFT, size is set dynamically from host on kernel launch
    extern __shared__ complex_type shared_mem[];
    // Execute FFT
    FFT().execute(thread_data, shared_mem);

    // Save results to global memory as described in
    // https://docs.nvidia.com/cuda/cufftdx/api/methods.html#input-output-data-format
    {
        unsigned int index = offset + threadIdx.x;
        for (unsigned i = 0; i < FFT::elements_per_thread; i++) {
            data[index] = thread_data[i];
            index += FFT::stride;
        }
    }
}

template<class FFT>
void run_fft(typename FFT::value_type* data /* pointer to device memory with FFT values */) {
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>();
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
}

int main(int, char**) {
    using complex_type = typename FFT::value_type;

    // Allocate device memory for FFT values
    complex_type* data;
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));

    // Set data
    for (size_t i = 0; i < size; i++) {
        data[i] = complex_type {float(i), -float(i)};
    }
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // std::cout << "input [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Increase max shared memory per CUDA block if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // Launch kernel using FFT type as template parameter
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // std::cout << "output [1st FFT]:\n";
    // for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
    //     std::cout << data[i].x << " " << data[i].y << std::endl;
    // }

    // Free allocated device memory
    CUDA_CHECK_AND_EXIT(cudaFree(data));
    std::cout << "Success" << std::endl;
}
