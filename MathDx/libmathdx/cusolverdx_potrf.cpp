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

#include <libcusolverdx.h>
#include <nvrtc.h>

#include <vector>

#include "common_examples.hpp"
#include "macros.hpp"

using namespace examples;

int main() {

    long long int size[1] = { 64 };
    long long int block_dim[3] = { 256, 1, 1 };

    /**
     * Create a descriptor
     * This is equivalent to `using SOLVER = ...` in cuSOLVERDx C++
     */

    cusolverdxDescriptor h { 0 };
    LIBMATHDX_CHECK(cusolverdxCreateDescriptor(&h));
    // Sets problem size
    LIBMATHDX_CHECK(cusolverdxSetOperatorInt64s(h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_SIZE, 1, size));
    // CUSOLVERDX_OPERATOR_BLOCK_DIM indicates the block dimension
    LIBMATHDX_CHECK(
        cusolverdxSetOperatorInt64s(h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_BLOCK_DIM, 3, block_dim));
    // CUSOLVERDX_TYPE_REAL means the inputs contain real type data
    // CUSOLVERDX_TYPE_COMPLEX would be for inputs with complex type data
    LIBMATHDX_CHECK(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_TYPE, cusolverdxType::CUSOLVERDX_TYPE_REAL));
    // CUSOLVERDX_API_SMEM means that inputs are in smem
    LIBMATHDX_CHECK(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_API, cusolverdxApi::CUSOLVERDX_API_SMEM));
    // This means we generate a solver based on picked algorithm ("POTRF")
    LIBMATHDX_CHECK(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_FUNCTION, cusolverdxFunction::CUSOLVERDX_FUNCTION_POTRF));
    // COMMONDX_EXECUTION_BLOCK means multiple threads in a block participate in the solver
    LIBMATHDX_CHECK(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
    // COMMONDX_PRECISION_F32 for single precision
    // COMMONDX_PRECISION_F64 for double precision
    LIBMATHDX_CHECK(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_PRECISION, commondxPrecision::COMMONDX_PRECISION_F64));
    // Sets fill mode for symmetric matrices
    LIBMATHDX_CHECK(cusolverdxSetOperatorInt64(
        h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_FILL_MODE, cusolverdxFillMode::CUSOLVERDX_FILL_MODE_LOWER));
    // Compute capability to target
    LIBMATHDX_CHECK(cusolverdxSetOperatorInt64(h, cusolverdxOperatorType::CUSOLVERDX_OPERATOR_SM, 800));

    // COMMONDX_OPTION_SYMBOL_NAME indicates the required name for the device function.
    LIBMATHDX_CHECK(cusolverdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, "my_solver"));

    /**
     * Compile the device function
     */

    commondxCode code;
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    // Specify arch to compile to
    LIBMATHDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, 800ll));
    LIBMATHDX_CHECK(cusolverdxFinalizeCode(code, h));
    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    std::vector<char> lto(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, lto.size(), lto.data()));
    LIBMATHDX_CHECK(commondxDestroyCode(code));

    /**
     * Query the universal cuSOLVERDx fatbin
     */

    size_t fatbin_size = 0;
    LIBMATHDX_CHECK(cusolverdxGetUniversalFATBINSize(h, &fatbin_size));
    std::vector<char> fatbin(fatbin_size);
    LIBMATHDX_CHECK(cusolverdxGetUniversalFATBIN(h, fatbin.size(), fatbin.data()));

    printf("Successfully generated LTOIR, %zu bytes for POTRF of size %d; universal fatbin %zu bytes\n",
           lto_size,
           (int)size[0],
           fatbin_size);

    LIBMATHDX_CHECK(cusolverdxDestroyDescriptor(h));
}
