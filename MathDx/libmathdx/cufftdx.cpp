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
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h,
                                            CUFFTDX_OPERATOR_DEVICE_FUNCTION,
                                            CUFFTDX_DEVICE_FUNCTION_EXECUTE | CUFFTDX_DEVICE_FUNCTION_FFT_ID |
                                                CUFFTDX_DEVICE_FUNCTION_BATCH_ID |
                                                CUFFTDX_DEVICE_FUNCTION_IS_THREAD_ACTIVE));

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

    // Let cuFFTDx choose a valid configuration for the requested size.
    long long int ept = 0, bpb = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_ELEMENTS_PER_THREAD, &ept));
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_FFTS_PER_BLOCK, &bpb));

    printf(
        "Successfully generated LTOIR, %zu bytes for FFT of size %d (%lld elements per thread, %lld FFTs per block)\n",
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

    // Query the execution and indexing helper symbols.
    const auto get_symbol = [h](cufftdxTraitType trait) {
        size_t str_size = 0;
        LIBMATHDX_CHECK(cufftdxGetTraitStrSize(h, trait, &str_size));
        std::string name(str_size, '\0');
        LIBMATHDX_CHECK(cufftdxGetTraitStr(h, trait, str_size, name.data()));
        return name;
    };
    const auto symbol_name = get_symbol(CUFFTDX_TRAIT_SYMBOL_NAME);
    const auto fft_id_name = get_symbol(CUFFTDX_TRAIT_SYMBOL_FFT_ID_NAME);
    const auto batch_id_name = get_symbol(CUFFTDX_TRAIT_SYMBOL_BATCH_ID_NAME);
    const auto is_thread_active_name = get_symbol(CUFFTDX_TRAIT_SYMBOL_IS_THREAD_ACTIVE_NAME);

    printf(
        "Function %s requires %lld B of shared memory %lld # of local memory elements and a block_dim of %lld %lld %lld\n",
        symbol_name.data(),
        shared_memory_size,
        storage_size,
        block_dim[0],
        block_dim[1],
        block_dim[2]);

    LIBMATHDX_CHECK(cufftdxDestroyDescriptor(h));

    const auto total_size = size * bpb;

    const char kernel_template[] = R"(
    // traits from cufftdx descriptor
    #define fft_function %s
    #define fft_thread_id %s
    #define fft_batch_id %s
    #define fft_is_thread_active %s
    constexpr unsigned size = %d;
    constexpr unsigned storage_size = %d;
    constexpr unsigned stride = %d;

    // function from cufftdx descriptor
    extern "C" __device__ void fft_function(float2* rmem, float2* smem);
    extern "C" __device__ void fft_thread_id(unsigned int*);
    extern "C" __device__ void fft_batch_id(unsigned int*);
    extern "C" __device__ void fft_is_thread_active(int*);

    extern "C" __global__ void simple_fft_kernel(float2* in, float2* out) {
        extern __shared__ float2 smem[];
        float2 rmem[storage_size]= {};

        // indexing for per thread FFT data copy
        unsigned int fft_id, thread_id;
        int active;
        fft_batch_id(&fft_id);
        fft_thread_id(&thread_id);
        fft_is_thread_active(&active);
        unsigned int batch_offset = fft_id * size;

        // copy input to rmem
        if (active) {
            for (unsigned int index = thread_id, i = 0; index < size; index += stride, ++i) {
                rmem[i] = in[batch_offset + index];
            }
        }
    
        // call JIT compiled FFT device function
        fft_function(rmem, smem);

        // copy output to out
        if (active) {
            for (unsigned int index = thread_id, i = 0; index < size; index += stride, ++i) {
                out[batch_offset + index] = rmem[i];
            }
        }
    }
    )";

    std::string final_kernel = strprintf(kernel_template,
                                         symbol_name.data(),
                                         fft_id_name.data(),
                                         batch_id_name.data(),
                                         is_thread_active_name.data(),
                                         size,
                                         (int)storage_size,
                                         (int)stride);
    std::vector<char> cubin = compile_and_link(final_kernel, lto, target_sm);

    CUmodule module {};
    CUfunction kernel {};
    CUDA_CHECK(cudaSetDevice(0));
    CU_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, 0, 0));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "simple_fft_kernel"));
    CU_CHECK(cuFuncSetAttribute(
        kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(shared_memory_size)));

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

    // Each all-ones FFT has a DC value of size and zero elsewhere, within FP32 roundoff.
    for (int i = 0; i < total_size; i++) {
        std::complex<float> expected =
            ((i % size == 0) ? std::complex<float>(static_cast<float>(size), 0.0f) : std::complex<float>(0.0f, 0.0f));

        const float error = std::abs(out[i] - expected);
        if (!std::isfinite(error) || error > 1e-5f * size) {
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
