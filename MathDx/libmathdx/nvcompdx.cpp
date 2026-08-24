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

#include <libnvcompdx.h>
#include <nvrtc.h>

#include <vector>

#include "common_examples.hpp"
#include "macros.hpp"

using namespace examples;

int main() {

    /**
     * Create a descriptor
     * This is equivalent to `using COMP = decltype(...)` in nvCOMPDx C++
     */

    nvcompdxDescriptor h { 0 };
    LIBMATHDX_CHECK(nvcompdxCreateDescriptor(&h));
    // NVCOMPDX_ALGORITHM_LZ4 is a general-purpose byte-level compressor
    // NVCOMPDX_ALGORITHM_ANS is a proprietary entropy encoder
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_ALGORITHM, NVCOMPDX_ALGORITHM_LZ4));
    // NVCOMPDX_DIRECTION_COMPRESS for compression, NVCOMPDX_DIRECTION_DECOMPRESS for decompression
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_DIRECTION, NVCOMPDX_DIRECTION_COMPRESS));
    // For LZ4: COMMONDX_R_8UI / COMMONDX_R_16UI / COMMONDX_R_32UI
    // For ANS: COMMONDX_R_8UI / COMMONDX_R_16F
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_DATATYPE, COMMONDX_R_8UI));
    // COMMONDX_EXECUTION_WARP: one warp (de)compresses one chunk.
    // COMMONDX_EXECUTION_BLOCK: one block (de)compresses one chunk.
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_EXECUTION, COMMONDX_EXECUTION_WARP));
    // Maximum uncompressed chunk size, in bytes (required for compress; not needed for decompress)
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_MAX_UNCOMP_CHUNK_SIZE, 65536));
    // Compute capability to target
    LIBMATHDX_CHECK(nvcompdxSetOperatorInt64(h, NVCOMPDX_OPERATOR_SM, get_dx_sm().operator_sm()));

    // COMMONDX_OPTION_SYMBOL_NAME indicates the required name for the device function.
    LIBMATHDX_CHECK(nvcompdxSetOptionStr(h, COMMONDX_OPTION_SYMBOL_NAME, "my_lz4_compress"));

    /**
     * Compile the device function
     */

    commondxCode code {};
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    // Specify arch to compile to
    LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, get_target_sm().operator_sm()));
    LIBMATHDX_CHECK(nvcompdxFinalizeCode(code, h));
    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    std::vector<char> lto(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, lto.size(), lto.data()));
    LIBMATHDX_CHECK(commondxDestroyCode(code));

    /**
     * Query the universal nvCOMPDx fatbin
     *
     * Unlike cuFFTDx/cuBLASDx, nvCOMPDx ships pre-compiled algorithm implementations
     * (LZ4, ANS, ...) inside libmathdx. The descriptor's LTOIR is just a thin wrapper that
     * calls into those implementations, so to launch a kernel you must link both the
     * per-descriptor LTOIR and this universal fatbin. See nvcompdx_batch.cpp for
     * an end-to-end run that does the linking.
     */

    size_t fatbin_size = 0;
    LIBMATHDX_CHECK(nvcompdxGetUniversalFATBINSize(h, &fatbin_size));
    std::vector<char> fatbin(fatbin_size);
    LIBMATHDX_CHECK(nvcompdxGetUniversalFATBIN(h, fatbin.size(), fatbin.data()));

    printf("Successfully generated LTOIR (%zu bytes) and universal nvCOMPDx fatbin (%zu bytes) for LZ4 compress\n",
           lto_size,
           fatbin_size);

    /**
     * Query traits
     */

    long long int shmem_size = 0;
    long long int tmp_size = 0;
    long long int max_comp_chunk_size = 0;
    LIBMATHDX_CHECK(nvcompdxGetTraitInt64(h, NVCOMPDX_TRAIT_SHMEM_SIZE_GROUP, &shmem_size));
    LIBMATHDX_CHECK(nvcompdxGetTraitInt64(h, NVCOMPDX_TRAIT_TMP_SIZE_GROUP, &tmp_size));
    LIBMATHDX_CHECK(nvcompdxGetTraitInt64(h, NVCOMPDX_TRAIT_MAX_COMP_CHUNK_SIZE, &max_comp_chunk_size));

    size_t name_size = 0;
    LIBMATHDX_CHECK(nvcompdxGetTraitStrSize(h, NVCOMPDX_TRAIT_SYMBOL_NAME, &name_size));
    std::vector<char> symbol_name(name_size);
    LIBMATHDX_CHECK(nvcompdxGetTraitStr(h, NVCOMPDX_TRAIT_SYMBOL_NAME, name_size, symbol_name.data()));

    printf("Function %s requires %lld B of shared memory and %lld B of global temp memory per warp;"
           " worst-case compressed chunk is %lld B\n",
           symbol_name.data(),
           shmem_size,
           tmp_size,
           max_comp_chunk_size);

    LIBMATHDX_CHECK(nvcompdxDestroyDescriptor(h));
    return 0;
}
