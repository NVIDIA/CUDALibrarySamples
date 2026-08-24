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

#include <libcufftdx.h>

#include <array>
#include <vector>

#include "common_examples.hpp"
#include "macros.hpp"

using namespace examples;

int main() {

    long long int size = 2048;
    long long int arch = 890;

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
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_API, cufftdxApi::CUFFTDX_API_LMEM));
    // COMMONDX_EXECUTION_BLOCK means multiple threads in a block participate in the FFT
    // COMMONDX_EXECUTION_THREAD would mean that each thread computes a single FFT
    LIBMATHDX_CHECK(
        cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
    // This is the FFT size
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_SIZE, size));
    // CUFFTDX_TYPE_C2C means complex-to-complex FFT type
    // CUFFTDX_TYPE_R2C means real-to-complex FFT type
    // CUFFTDX_TYPE_C2R means complex-to-real FFT type
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_TYPE, cufftdxType::CUFFTDX_TYPE_C2C));
    // CUFFTDX_DIRECTION_FORWARD means a forward FFT
    // CUFFTDX_DIRECTION_INVERSE means an inverse FFT
    LIBMATHDX_CHECK(
        cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_DIRECTION, cufftdxDirection::CUFFTDX_DIRECTION_FORWARD));
    // COMMONDX_PRECISION_F16 for half precision
    // COMMONDX_PRECISION_F32 for single precision
    // COMMONDX_PRECISION_F64 for double precision
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F32));
    // Compute capability to target
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_SM, arch));

    // COMMONDX_OPTION_SYMBOL_NAME indicates the required name for the device function.
    LIBMATHDX_CHECK(cufftdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, "my_fft"));

    /**
     * Find valid EPTs
     */

    cufftdxKnobType_t knobs = CUFFTDX_KNOB_ELEMENTS_PER_THREAD;
    size_t num_epts = 0;
    LIBMATHDX_CHECK(cufftdxGetKnobInt64Size(h, 1, &knobs, &num_epts));
    std::vector<long long int> epts(num_epts, 0);
    LIBMATHDX_CHECK(cufftdxGetKnobInt64s(h, 1, &knobs, epts.size(), epts.data()));

    for (auto ept : epts) {
        printf("Valid ept: %lld\n", ept);
    }

    /**
     * Pick of one of them directly
     */

    if (epts.empty()) {
        printf("No valid EPTs!");
        return 1;
    }
    LIBMATHDX_CHECK(cufftdxSetOperatorInt64(h, CUFFTDX_OPERATOR_ELEMENTS_PER_THREAD, epts.front()));

    /**
     * Confirms that this descriptor is valid and will compile
     */

    int valid = -1;
    LIBMATHDX_CHECK(cufftdxIsSupported(h, &valid));
    printf("Descriptor is %s\n", valid ? "valid" : "invalid");
    if (!valid) {
        return 1;
    }

    /**
     * Compile the device function
     */

    commondxCode code;
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    // Specify arch to compile to
    LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, 800ll));
    LIBMATHDX_CHECK(cufftdxFinalizeCode(code, h));
    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    std::vector<char> lto(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, lto.size(), lto.data()));
    LIBMATHDX_CHECK(commondxDestroyCode(code));

    // `lto` is an LTOIR container with NVVM/LTO inside. It contains a device function called `my_fft` with signature
    // void my_fft(void* rmem, void* smem)
    // which can be called to compute the FFT described above
    // smem must point to `CUFFTDX_TRAIT_SHARED_MEMORY_SIZE` bytes of shared memory
    // rmem is the usual register input
    //
    // In order to create a valid kernel, `lto` must be linked to the user kernel (compiled, e.g., with NVRTC) using
    // nvJitLink.

    printf("Successfully generated LTOIR, %zu bytes for FFT of size %lld\n", lto_size, size);

    /**
     * Query shared memory size, suggested elements per thread and suggested ffts per block
     */

    // How much shared memory to allocate?
    long long int shared_memory_size = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_SHARED_MEMORY_SIZE, &shared_memory_size));

    // What EPT is suggested?
    long long int ept = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_ELEMENTS_PER_THREAD, &ept));

    // How many FFTs per Block?
    long long int fpb = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_FFTS_PER_BLOCK, &fpb));

    // How many FFTs per Block is *suggested*?
    long long int sfpb = 0;
    LIBMATHDX_CHECK(cufftdxGetTraitInt64(h, CUFFTDX_TRAIT_SUGGESTED_FFTS_PER_BLOCK, &sfpb));

    // What's the corresponding block dim (for "non suggested" FFTs per block)?
    std::array<long long int, 3> block_dim = { 0 };
    LIBMATHDX_CHECK(cufftdxGetTraitInt64s(h, CUFFTDX_TRAIT_BLOCK_DIM, block_dim.size(), block_dim.data()));

    printf("Function requires %lld B of shared memory with EPT %lld and FPB %lld / %lld, block_dim is %lld %lld %lld\n",
           shared_memory_size,
           ept,
           fpb,
           sfpb,
           block_dim.at(0),
           block_dim.at(1),
           block_dim.at(2));

    LIBMATHDX_CHECK(cufftdxDestroyDescriptor(h));
}
