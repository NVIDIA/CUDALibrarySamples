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
#include <libcuranddx.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "common_examples.hpp"
#include "macros.hpp"

using namespace examples;

int main() {

    constexpr int count = 32;

    arch_t dx_sm = get_dx_sm();
    arch_t target_sm = get_target_sm();

    // Create descriptor. See: https://docs.nvidia.com/cuda/curanddx/
    curanddxDescriptor h { 0 };
    LIBMATHDX_CHECK(curanddxCreateDescriptor(&h));
    LIBMATHDX_CHECK(curanddxSetOperatorInt64(h, CURANDDX_OPERATOR_GENERATOR, CURANDDX_GENERATOR_PCG));
    LIBMATHDX_CHECK(curanddxSetOperatorInt64(h, CURANDDX_OPERATOR_EXECUTION, COMMONDX_EXECUTION_THREAD));
    LIBMATHDX_CHECK(curanddxSetOperatorInt64(h, CURANDDX_OPERATOR_DISTRIBUTION, CURANDDX_DISTRIBUTION_UNIFORM));
    LIBMATHDX_CHECK(curanddxSetOperatorInt64(h, CURANDDX_OPERATOR_OUTPUT_TYPE, COMMONDX_R_32F));
    LIBMATHDX_CHECK(curanddxSetOperatorInt64(h, CURANDDX_OPERATOR_GENERATE_METHOD, CURANDDX_GENERATE_METHOD_SINGLE));
    LIBMATHDX_CHECK(curanddxSetOperatorInt64(h, CURANDDX_OPERATOR_SM, dx_sm.operator_sm()));
    LIBMATHDX_CHECK(curanddxSetOperatorInt64(h,
                                             CURANDDX_OPERATOR_DEVICE_FUNCTIONS,
                                             CURANDDX_DEVICE_FUNCTION_GENERATE | CURANDDX_DEVICE_FUNCTION_INIT_STATE |
                                                 CURANDDX_DEVICE_FUNCTION_DESTROY_STATE));

    // Generate device function (LTOIR)
    commondxCode code;
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, target_sm.operator_sm()));
    LIBMATHDX_CHECK(curanddxFinalizeCode(code, h));

    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    std::vector<char> lto(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, lto.size(), lto.data()));
    LIBMATHDX_CHECK(commondxDestroyCode(code));

    // Query symbol names for declaration and call
    size_t size = 0;
    LIBMATHDX_CHECK(curanddxGetTraitStrSize(h, CURANDDX_TRAIT_SYMBOL_GENERATE_NAME, &size));
    std::vector<char> generate_name(size);
    LIBMATHDX_CHECK(curanddxGetTraitStr(h, CURANDDX_TRAIT_SYMBOL_GENERATE_NAME, size, generate_name.data()));
    LIBMATHDX_CHECK(curanddxGetTraitStrSize(h, CURANDDX_TRAIT_SYMBOL_INIT_STATE_NAME, &size));
    std::vector<char> init_name(size);
    LIBMATHDX_CHECK(curanddxGetTraitStr(h, CURANDDX_TRAIT_SYMBOL_INIT_STATE_NAME, size, init_name.data()));
    LIBMATHDX_CHECK(curanddxGetTraitStrSize(h, CURANDDX_TRAIT_SYMBOL_DESTROY_STATE_NAME, &size));
    std::vector<char> destroy_name(size);
    LIBMATHDX_CHECK(curanddxGetTraitStr(h, CURANDDX_TRAIT_SYMBOL_DESTROY_STATE_NAME, size, destroy_name.data()));

    // Query opaque state size & alignment
    long long int state_size = 0;
    long long int state_align = 0;
    LIBMATHDX_CHECK(curanddxGetTraitInt64(h, CURANDDX_TRAIT_STATE_SIZE, &state_size));
    LIBMATHDX_CHECK(curanddxGetTraitInt64(h, CURANDDX_TRAIT_STATE_ALIGNMENT, &state_align));

    printf("Successfully generated LTOIR, %zu bytes for cuRANDDx device functions %s, %s, and %s\n",
           lto_size,
           generate_name.data(),
           init_name.data(),
           destroy_name.data());

    LIBMATHDX_CHECK(curanddxDestroyDescriptor(h));

    const float MIN = -2.0;
    const float MAX = 3.0;
    // Create CUDA C++ kernel that calls the device function
    const char kernel_template[] = R"(
    #define RNG_INIT %s
    #define MIN %f
    #define MAX %f
    #define RNG_GENERATE %s
    #define RNG_FREE %s
    constexpr unsigned state_size = %d;
    constexpr unsigned state_alignment = %d;

    extern "C" __device__ void RNG_INIT(unsigned long long* seed,
                                        unsigned long long* subseq,
                                        unsigned long long* offset,
                                        void* state);

    extern "C" __device__ void RNG_FREE(void* state);

    extern "C" __device__ void RNG_GENERATE(void* state,
                                            float* out, 
                                            float* params);

    extern "C" __global__ void curand_example_kernel(float* out) {
        const unsigned tid = threadIdx.x;
        unsigned long long seed = 1234ull;
        unsigned long long subseq = 0ull;
        unsigned long long offset = (unsigned long long)(1 + tid);
        alignas(state_alignment) char state[state_size];
        RNG_INIT(&seed, &subseq, &offset, state);
        float params[2] = {MIN, MAX};
        RNG_GENERATE(state, out + tid, params);
        RNG_FREE(state);
    }
    )";

    // Compile and link
    std::string final_kernel = strprintf(kernel_template,
                                         init_name.data(),
                                         MIN,
                                         MAX,
                                         generate_name.data(),
                                         destroy_name.data(),
                                         (int)state_size,
                                         (int)state_align);
    std::vector<char> cubin = compile_and_link(final_kernel, lto, target_sm);

    // Load and run
    CUmodule module {};
    CUfunction kernel {};
    CUDA_CHECK(cudaSetDevice(0));
    CU_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, 0, 0));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "curand_example_kernel"));

    float* out = nullptr;
    CUDA_CHECK(cudaMallocManaged(&out, count * sizeof(float)));
    std::fill(out, out + count, 0.0f);

    std::array args = { static_cast<void*>(&out) };
    CU_CHECK(cuLaunchKernel(kernel, 1, 1, 1, count, 1, 1, 0, 0, args.data(), nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Print and basic range check
    std::cout << "Generated " << count << " uniform random values in [" << MIN << "," << MAX << "):\n";
    for (unsigned i = 0; i < count; i++) {
        float v = out[i];
        std::cout << v << (i + 1 == count ? '\n' : ' ');
        if (!(v >= MIN && v < MAX)) {
            std::cerr << "Value out of range at index " << i << ": " << v << "\n";
            std::abort();
        }
    }

    CU_CHECK(cuModuleUnload(module));
    CUDA_CHECK(cudaFree(out));

    return 0;
}
