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

#include "block_io.hpp"
#include "common.hpp"

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void block_fft_kernel(typename FFT::value_type* data) {
    using complex_type = typename FFT::value_type;

    // Local array for thread
    complex_type thread_data[FFT::storage_size];

    // ID of FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Load data from global memory to registers
    example::io<FFT>::load(data, thread_data, local_fft_id);

    // Execute FFT
    extern __shared__ complex_type shared_mem[];
    FFT().execute(thread_data, shared_mem);

    // Save results
    example::io<FFT>::store(thread_data, data, local_fft_id);
}

// An example of performing 64-point FFT in a CUDA block using cuFFTDx
template<unsigned int Arch>
void block_fft() {
    using namespace cufftdx;

    // FFT description in cuFFTDx
    // size - 64
    // type - complex-to-complex
    // direction - forward
    // precisiion - float
    using FFT          = decltype(Size<64>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>()
                                  + Precision<float>() + Block() + SM<Arch>());
    using complex_type = typename FFT::value_type;

    // Allocate managed memory for input/output
    complex_type* data;
    auto          size       = FFT::ffts_per_block * cufftdx::size_of<FFT>::value;
    auto          size_bytes = size * sizeof(complex_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&data, size_bytes));
    for (size_t i = 0; i < size; i++) {
        data[i] = complex_type {float(i), -float(i)};
    }

    std::cout << "input [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    // Increase max shared memory if required by FFT
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        block_fft_kernel<FFT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        FFT::shared_memory_size));

    // Invokes kernel with FFT::block_dim threads in CUDA block
    block_fft_kernel<FFT><<<1, FFT::block_dim, FFT::shared_memory_size>>>(data);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::cout << "output [1st FFT]:\n";
    for (size_t i = 0; i < cufftdx::size_of<FFT>::value; i++) {
        std::cout << data[i].x << " " << data[i].y << std::endl;
    }

    CUDA_CHECK_AND_EXIT(cudaFree(data));
    std::cout << "Success" << std::endl;
}

template<unsigned int Arch>
struct block_fft_functor {
    void operator()() { return block_fft<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<block_fft_functor>();
}
