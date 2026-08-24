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
 * Basic cuBLASDx GEMM example: configures a block-level matrix multiply descriptor,
 * compiles to LTOIR, and reports the generated code size.
 */
int main() {

    int m = 32;
    int n = 8;
    int k = 16;
    int num_threads = 32;
    arch_t dx_sm = get_dx_sm();
    arch_t target_sm = get_target_sm();

    /**
     * Create a descriptor
     * This is equivalent to `using BLAS = ...` in cuBLASDx C++
     */

    cublasdxDescriptor h { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDescriptor(&h));

    // This means we generate a Matmul ("MM")
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_FUNCTION, cublasdxFunction::CUBLASDX_FUNCTION_MM));
    // COMMONDX_EXECUTION_BLOCK means we are generating a function with a "Block" API semantic. All threads in the CUDA
    //   block must participate and will cooperate to compute the matmul.
    // COMMONDX_EXECUTION_THREAD would mean that each thread compute its own matmul independently from other threads.
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
    // CUBLASDX_API_SMEM means the function take inputs in shared memory and produce output in shared memory.
    //   The leading dimensions are fixed at compile time. The function signature is:
    //     void gemm(value_type, value_type*, value_type*, value_type, value_type)
    // CUBLASDX_API_SMEM_DYNAMIC_LD would mean that the function takes runtime leading dimension values.
    //   Such function has the following signature:
    //     void gemm(value_type, value_type*, unsigned, value_type*, unsigned, value_type, value_type*, unsigned)
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_API, cublasdxApi::CUBLASDX_API_SMEM));
    // COMMONDX_PRECISION_F16 means the matrices are filled with half-precision floating point numbers
    // COMMONDX_PRECISION_F32 would be for single-precision
    // COMMONDX_PRECISION_F64 would be for double precision
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F16));
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
    // CUBLASDX_OPERATOR_SIZE indicates the (M, N, K) of the problem.
    std::array<long long int, 3> size = { m, n, k };
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SIZE, size.size(), size.data()));
    // CUBLASDX_OPERATOR_TRANSPOSE_MODE is a tuple (ta, tb), where 'ta' and 'tb' can take one of the following
    //   values:
    // - CUBLASDX_TRANSPOSE_MODE_NON_TRANSPOSED - the input matrix (A or B) is not transposed
    // - CUBLASDX_TRANSPOSE_MODE_TRANSPOSED - the input matrix (A or B) is transposed
    // - CUBLASDX_TRANSPOSE_MODE_CONJ_TRANSPOSED - the input matrix (A or B) is conjugated and transposed
    std::array<long long int, 2> transpose_mode = { cublasdxTransposeMode_t::CUBLASDX_TRANSPOSE_MODE_NON_TRANSPOSED,
                                                    cublasdxTransposeMode_t::CUBLASDX_TRANSPOSE_MODE_NON_TRANSPOSED };
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_TRANSPOSE_MODE, transpose_mode.size(), transpose_mode.data()));

    // COMMONDX_OPTION_SYMBOL_NAME indicates the required name for the device function.
    LIBMATHDX_CHECK(cublasdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, "my_gemm"));

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

    printf("Successfully generated LTOIR (version %lld), %zu bytes for GEMM %d x %d x %d\n", isa, lto_size, m, n, k);

    LIBMATHDX_CHECK(cublasdxDestroyDescriptor(h));
}
