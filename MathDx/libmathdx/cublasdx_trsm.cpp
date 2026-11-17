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

#include <libcublasdx.h>

#include <array>
#include <vector>

#include "arch.hpp"
#include "macros.hpp"

using namespace examples;

/**
 * Basic cuBLASDx TRSM example: configures a block-level triangular solve descriptor,
 * compiles to LTOIR, and reports the generated code size.
 *
 * TRSM solves op(A) X = B (side left) or X op(A) = B (side right) where A is triangular.
 * B is overwritten in-place with the solution X. There are no alpha / beta scalars.
 */
int main() {

    int m = 32;
    int n = 16;
    int num_threads = 128;
    arch_t dx_sm = get_dx_sm();
    arch_t target_sm = get_target_sm();

    /**
     * Create a descriptor
     * This is equivalent to `using TRSM = ...` in cuBLASDx C++
     */

    cublasdxDescriptor h { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDescriptor(&h));

    // This means we generate a triangular solve ("TRSM")
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_FUNCTION, cublasdxFunction::CUBLASDX_FUNCTION_TRSM));
    // COMMONDX_EXECUTION_BLOCK means we are generating a function with a "Block" API semantic. All threads in the CUDA
    //   block must participate and will cooperate to solve the triangular system.
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
    // CUBLASDX_API_SMEM means the function takes inputs in shared memory and solves in-place in shared memory.
    //   The leading dimensions are fixed at compile time. The function signature is:
    //     void trsm(a_value_type*, b_value_type*)
    // CUBLASDX_API_SMEM_DYNAMIC_LD would mean that the function takes runtime leading dimension values.
    //   Such function has the following signature:
    //     void trsm(a_value_type*, unsigned*, b_value_type*, unsigned*)
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_API, cublasdxApi::CUBLASDX_API_SMEM));
    // COMMONDX_PRECISION_F32 means the matrices are filled with single-precision floating point numbers
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F32));
    // CUBLASDX_OPERATOR_SM indicates the target architecture (700 for SM70, etc)
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SM, dx_sm.operator_sm()));
    // CUBLASDX_TYPE_REAL means the matrices contain real type data
    // CUBLASDX_TYPE_COMPLEX would be for matrices with complex type data
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_TYPE, cublasdxType::CUBLASDX_TYPE_REAL));
    // CUBLASDX_OPERATOR_BLOCK_DIM indicates the block dimension
    std::array<long long int, 3> block_dim = { num_threads, 1, 1 };
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_BLOCK_DIM, block_dim.size(), block_dim.data()));
    // CUBLASDX_OPERATOR_SIZE indicates the (M, N, K) of the problem. For TRSM, B and X are M x N and
    //   A is M x M (side left) or N x N (side right). K is ignored.
    std::array<long long int, 3> size = { m, n, 1 };
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SIZE, size.size(), size.data()));
    // CUBLASDX_SIDE_LEFT means we solve op(A) X = B
    // CUBLASDX_SIDE_RIGHT would solve X op(A) = B
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SIDE, cublasdxSide::CUBLASDX_SIDE_LEFT));
    // CUBLASDX_FILL_MODE_LOWER means A is lower-triangular
    // CUBLASDX_FILL_MODE_UPPER would mean A is upper-triangular
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_FILL_MODE, cublasdxFillMode::CUBLASDX_FILL_MODE_LOWER));
    // CUBLASDX_DIAG_NON_UNIT means the diagonal of A is used as-is
    // CUBLASDX_DIAG_UNIT would mean the diagonal of A is assumed to be all ones
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_DIAG, cublasdxDiag::CUBLASDX_DIAG_NON_UNIT));

    // COMMONDX_OPTION_SYMBOL_NAME indicates the required name for the device function.
    LIBMATHDX_CHECK(cublasdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, "my_trsm"));

    /**
     * Compile the device function
     */

    commondxCode code;
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    // Specify arch to compile to
    LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, target_sm.operator_sm()));
    LIBMATHDX_CHECK(cublasdxFinalizeCode(code, h));
    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    std::vector<char> lto(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, lto.size(), lto.data()));
    long long int isa = 0;
    LIBMATHDX_CHECK(commondxGetCodeOptionInt64(code, COMMONDX_OPTION_CODE_ISA, &isa));
    LIBMATHDX_CHECK(commondxDestroyCode(code));

    /**
     * Retrieve the universal fatbin
     *
     * cuBLASDx's universal fatbin contains precompiled TRSM device solvers.
     * Link the fatbin with the LTOIR. Otherwise, TRSM symbols stay unresolved at device link time.
     * For example, use NVJITLINK_INPUT_FATBIN alongside NVJITLINK_INPUT_LTOIR with nvJitLink.
     */

    size_t fatbin_size = 0;
    LIBMATHDX_CHECK(cublasdxGetUniversalFATBINSize(h, &fatbin_size));
    std::vector<char> fatbin(fatbin_size);
    LIBMATHDX_CHECK(cublasdxGetUniversalFATBIN(h, fatbin.size(), fatbin.data()));

    printf("Successfully generated LTOIR (version %lld), %zu bytes, and retrieved the universal "
           "fatbin, %zu bytes, for TRSM %d x %d\n",
           isa,
           lto_size,
           fatbin_size,
           m,
           n);

    LIBMATHDX_CHECK(cublasdxDestroyDescriptor(h));
}
