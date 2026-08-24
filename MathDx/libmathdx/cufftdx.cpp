/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <libcufftdx.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <string>
#include <vector>

#include "common_examples.hpp"
#include "macros.hpp"

using namespace examples;

int main() {

    int size = 32;
    int ept = 4;
    int bpb = 2;

    arch_t dx_sm = get_dx_sm();
    arch_t target_sm = get_target_sm();

    /**
     * Create a descriptor
     * This is equivalent to `using FFT = ...` in cuFFTDx C++
     */

    cufftdxDescriptor h { 0 };
    LIBMATHDX_CHECK(cufftdxCreateDescriptor(&h));
    // CUFFTDX_API_LMEM means the function will be of signature:
    //     void(value_type*, value_type*)
    //   with the first argument being local memory ("registers"), with each thread holding "EPT" elements
    //   and the second being a pointer to a shared memory scratch buffer
    // CUFFTDX_API_SMEM would mean that the function will be of signature:
    //     void(value_type*)
    //   and takes a shared memory pointer with all the elements laid out in natural order
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_API, CUFFTDX_API_LMEM));
    // COMMONDX_EXECUTION_BLOCK means multiple threads in a block participate in the FFT
    // COMMONDX_EXECUTION_THREAD would mean that each thread computes a single FFT
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_EXECUTION, COMMONDX_EXECUTION_BLOCK));
    // This is the FFT size
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_SIZE, size));
    // CUFFTDX_TYPE_C2C means complex-to-complex FFT type
    // CUFFTDX_TYPE_R2C means real-to-complex FFT type
    // CUFFTDX_TYPE_C2R means complex-to-real FFT type
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_TYPE, CUFFTDX_TYPE_C2C));
    // CUFFTDX_DIRECTION_FORWARD means a forward FFT
    // CUFFTDX_DIRECTION_INVERSE means an inverse FFT
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_DIRECTION, CUFFTDX_DIRECTION_FORWARD));
    // COMMONDX_PRECISION_F16 for half precision
    // COMMONDX_PRECISION_F32 for single precision
    // COMMONDX_PRECISION_F64 for double precision
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_PRECISION, COMMONDX_PRECISION_F32));
    // Compute capability to target
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_SM, dx_sm.operator_sm()));
    // Number of elements per thread (only for BLOCK execution)
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD, ept));
    // Number of ffts per block (only for BLOCK execution)
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_FFTS_PER_BLOCK, bpb));

    // COMMONDX_OPTION_SYMBOL_NAME indicates the required name for the device function.
    LIBMATHDX_CHECK(cufftdxSetOptionStr(h, COMMONDX_OPTION_SYMBOL_NAME, "my_fft"));

    /**
     * Compile the device function
     */

    commondxCode code;
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    // Specify arch to compile to
    LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, target_sm.operator_sm()));
    LIBMATHDX_CHECK(cufftdxFinalizeCode(code, h));
    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    std::vector<char> lto(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, lto.size(), lto.data()));
    LIBMATHDX_CHECK(commondxDestroyCode(code));

    printf("Successfully generated LTOIR, %zu bytes for FFT of size %d (%d elements per thread, %d FFTs per block)\n",
           lto_size,
           size,
           ept,
           bpb);

    /**
     * Query traits
     */

    // How much shared memory to allocate?
    long long int shared_memory_size = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_SHARED_MEMORY_SIZE, &shared_memory_size));

    // What size of blocks to launch?
    std::array<long long int, 3> block_dim = { 0, 0, 0 };
    LIBMATHDX_CHECK(cufftdxGetTraitInt64s(h, CUFFTDX_TRAIT_BLOCK_DIM, block_dim.size(), block_dim.data()));

    // How much local memory to allocate?
    long long int storage_size = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_STORAGE_SIZE, &storage_size));

    // Stride between elements?
    long long int stride = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_STRIDE, &stride));

    // What is the name of the function?
    size_t str_size = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitStrSize(h, CUFFTDX_TRAIT_SYMBOL_NAME, &str_size));
    std::vector<char> symbol_name(str_size);
    LIBMATHDX_CHECK(cufftdxGetTraitStr(h, CUFFTDX_TRAIT_SYMBOL_NAME, str_size, symbol_name.data()));

    printf(
        "Function %s requires %lld B of shared memory %lld # of local memory elements and a block_dim of %lld %lld %lld\n",
        symbol_name.data(),
        shared_memory_size,
        storage_size,
        block_dim[0],
        block_dim[1],
        block_dim[2]);

    LIBMATHDX_CHECK(cufftdxDestroyDescriptor(h));

    int total_size = size * bpb;

    const char kernel_template[] = R"(
    // traits from cufftdx descriptor
    #define fft_function %s
    constexpr unsigned size = %d;
    constexpr unsigned storage_size = %d;
    constexpr unsigned ept = %d;
    constexpr unsigned stride = %d;

    // function from cufftdx descriptor
    extern "C" __device__ void fft_function(float2* rmem, float2* smem);

    extern "C" __global__ void simple_fft_kernel(float2* in, float2* out) {
        extern __shared__ float2 smem[];
        float2 rmem[storage_size]= {};

        // indexing for per thread FFT data copy
        int fft_id = threadIdx.y;
        int thread_id = threadIdx.x;
        int base_index = fft_id * size + thread_id;

        // copy input to rmem
        for (int i = 0; i < ept; i++) {
            rmem[i] = in[base_index + i * stride];
        }
    
        // call JIT compiled FFT device function
        fft_function(rmem, smem);

        // copy output to out
        for (int i = 0; i < ept; i++) {
            out[base_index + i * stride] = rmem[i];
        }
    }
    )";

    std::string final_kernel =
        strprintf(kernel_template, symbol_name.data(), size, (int)storage_size, (int)ept, (int)stride);
    std::vector<char> cubin = compile_and_link(final_kernel, lto, target_sm);

    CUmodule module {};
    CUfunction kernel {};
    CUDA_CHECK(cudaSetDevice(0));
    CU_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, 0, 0));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "simple_fft_kernel"));

    std::complex<float>*in = nullptr, *out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&in, total_size * sizeof(std::complex<float>)));
    CUDA_CHECK(cudaMallocManaged(&out, total_size * sizeof(std::complex<float>)));
    std::fill(in, in + total_size, std::complex<float>(1.0f, 0.0f));
    std::fill(out, out + total_size, std::complex<float>(1.0f, 0.0f));

    std::vector<void*> kernel_args = { reinterpret_cast<void*>(&in), reinterpret_cast<void*>(&out) };
    CU_CHECK(cuLaunchKernel(kernel,
                            1,
                            1,
                            1,
                            static_cast<unsigned int>(block_dim[0]),
                            static_cast<unsigned int>(block_dim[1]),
                            static_cast<unsigned int>(block_dim[2]),
                            static_cast<unsigned int>(shared_memory_size),
                            0,
                            kernel_args.data(),
                            nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    // with an input of 1.0f, the output of the FFT should be the sum of inputs and equaling the FFT size
    // we are performing a batch of size 2 so we should have 4 values (real, imag) equalling 32 with remaining values
    // being 0
    for (int i = 0; i < total_size; i++) {
        std::complex<float> expected =
            ((i % size == 0) ? std::complex<float>(static_cast<float>(size), 0.0f) : std::complex<float>(0.0f, 0.0f));

        if (std::abs(out[i] - expected) > 1e-7) {
            printf("Error: out[%d] = %f, %f, expected %f, %f\n",
                   i,
                   out[i].real(),
                   out[i].imag(),
                   expected.real(),
                   expected.imag());
            abort();
        }
    }
    printf("Successfully ran the kernel\n");

    CU_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));

    return 0;
}
