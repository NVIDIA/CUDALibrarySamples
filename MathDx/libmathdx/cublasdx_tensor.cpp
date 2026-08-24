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

#include <libcommondx.h>
#include <libcublasdx.h>

#include <array>
#include <vector>

#include "arch.hpp"
#include "macros.hpp"

using namespace examples;

int main() {

    long long int m = 256;
    long long int n = 128;
    long long int k = 16;
    long long int num_threads = 128;

    arch_t dx_sm = get_dx_sm();
    arch_t target_sm = get_target_sm();

    auto dx_sm_array = dx_sm.to_array();
    auto target_sm_array = target_sm.to_array();

    /**
     * Create the cuBLASDx descriptor
     */
    cublasdxDescriptor h { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDescriptor(&h));

    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_FUNCTION, cublasdxFunction::CUBLASDX_FUNCTION_MM));
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_EXECUTION, commondxExecution::COMMONDX_EXECUTION_BLOCK));
    // Using the Opaque Tensor API
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_API, cublasdxApi::CUBLASDX_API_TENSORS));
    std::array<long long int, 3> prec = { COMMONDX_PRECISION_F32, COMMONDX_PRECISION_F32, COMMONDX_PRECISION_F32 };
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_PRECISION, prec.size(), prec.data()));
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_SM, dx_sm_array.size(), dx_sm_array.data()));
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64(h, cublasdxOperatorType::CUBLASDX_OPERATOR_TYPE, cublasdxType::CUBLASDX_TYPE_REAL));
    std::array<long long int, 3> block_dim = { num_threads, 1, 1 };
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_BLOCK_DIM, block_dim.size(), block_dim.data()));
    std::array<long long int, 3> size = { m, n, k };
    LIBMATHDX_CHECK(
        cublasdxSetOperatorInt64s(h, cublasdxOperatorType::CUBLASDX_OPERATOR_SIZE, size.size(), size.data()));

    std::array<long long int, 3> arrangement = { CUBLASDX_ARRANGEMENT_COL_MAJOR,
                                                 CUBLASDX_ARRANGEMENT_COL_MAJOR,
                                                 CUBLASDX_ARRANGEMENT_COL_MAJOR };
    LIBMATHDX_CHECK(cublasdxSetOperatorInt64s(
        h, cublasdxOperatorType::CUBLASDX_OPERATOR_ARRANGEMENT, arrangement.size(), arrangement.data()));

    LIBMATHDX_CHECK(cublasdxSetOptionStr(h, commondxOption::COMMONDX_OPTION_SYMBOL_NAME, "matmul"));

    /**
     * Define the input and output tensors
     */
    cublasdxTensor smem_a { 0 };
    cublasdxTensor smem_b { 0 };
    cublasdxTensor acc_c { 0 };
    cublasdxTensor smem_c { 0 };
    cublasdxTensor rmem_c { 0 };
    LIBMATHDX_CHECK(cublasdxCreateTensor(h, CUBLASDX_TENSOR_SUGGESTED_SMEM_A, &smem_a));
    LIBMATHDX_CHECK(cublasdxCreateTensor(h, CUBLASDX_TENSOR_SUGGESTED_SMEM_B, &smem_b));
    LIBMATHDX_CHECK(cublasdxCreateTensor(h, CUBLASDX_TENSOR_SUGGESTED_ACCUMULATOR_C, &acc_c));
    LIBMATHDX_CHECK(cublasdxCreateTensor(h, CUBLASDX_TENSOR_SUGGESTED_SMEM_C, &smem_c));
    LIBMATHDX_CHECK(cublasdxCreateTensor(h, CUBLASDX_TENSOR_SUGGESTED_RMEM_C, &rmem_c));

    cublasdxTensor big_gmem {};
    std::vector<long long int> shape = { m, k };
    std::vector<long long int> strides = { LIBMATHDX_RUNTIME, 1 };
    LIBMATHDX_CHECK(cublasdxCreateTensorStrided(
        CUBLASDX_MEMORY_SPACE_GMEM, COMMONDX_R_32F, nullptr, shape.size(), shape.data(), strides.data(), &big_gmem));

    std::array tensors = { smem_a, smem_b, acc_c, smem_c, rmem_c, big_gmem };
    LIBMATHDX_CHECK(cublasdxFinalizeTensors(tensors.size(), tensors.data()));

    for (auto t : tensors) {
        long long int alignment = 0;
        long long int size = 0;
        size_t name_size = 0;
        LIBMATHDX_CHECK(cublasdxGetTensorTraitInt64(t, CUBLASDX_TENSOR_TRAIT_ALIGNMENT_BYTES, &alignment));
        LIBMATHDX_CHECK(cublasdxGetTensorTraitInt64(t, CUBLASDX_TENSOR_TRAIT_STORAGE_BYTES, &size));
        LIBMATHDX_CHECK(cublasdxGetTensorTraitStrSize(t, CUBLASDX_TENSOR_TRAIT_OPAQUE_NAME, &name_size));
        std::vector<char> name(name_size);
        LIBMATHDX_CHECK(cublasdxGetTensorTraitStr(t, CUBLASDX_TENSOR_TRAIT_OPAQUE_NAME, name.size(), name.data()));
        printf("Tensor %lld: name %s, storage size %lld B, alignment %lld B\n",
               static_cast<long long int>(t),
               name.data(),
               size,
               alignment);
    }

    /**
     * Define a function operating on those input and output tensors.
     *
     * The device function output is an opaque and stateful accumulator.
     */
    std::array gemm_tensors = { smem_a, smem_b, acc_c };
    cublasdxDeviceFunction gemm_sa_sb_rc { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(
        h, CUBLASDX_DEVICE_FUNCTION_EXECUTE, gemm_tensors.size(), gemm_tensors.data(), &gemm_sa_sb_rc));

    cublasdxDeviceFunction init { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(h, CUBLASDX_DEVICE_FUNCTION_CREATE, 1, &acc_c, &init));

    cublasdxDeviceFunction destroy { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(h, CUBLASDX_DEVICE_FUNCTION_DESTROY, 1, &acc_c, &destroy));

    cublasdxDeviceFunction clear { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(h, CUBLASDX_DEVICE_FUNCTION_CLEAR, 1, &acc_c, &clear));

    std::array copy_tensors_smem = { acc_c, smem_c };
    cublasdxDeviceFunction copy_smem { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(
        LIBMATHDX_NONE, CUBLASDX_DEVICE_FUNCTION_COPY, copy_tensors_smem.size(), copy_tensors_smem.data(), &copy_smem));
    LIBMATHDX_CHECK(
        cublasdxSetDeviceFunctionOptionInt64(copy_smem, CUBLASDX_DEVICE_FUNCTION_OPTION_NUM_THREADS, num_threads));

    std::array copy_tensors_rmem = { acc_c, rmem_c };
    cublasdxDeviceFunction copy_rmem { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(
        h, CUBLASDX_DEVICE_FUNCTION_COPY, copy_tensors_rmem.size(), copy_tensors_rmem.data(), &copy_rmem));

    std::array copy_tensors_big = { big_gmem, smem_a };
    cublasdxDeviceFunction copy_big { 0 };
    LIBMATHDX_CHECK(cublasdxCreateDeviceFunction(
        h, CUBLASDX_DEVICE_FUNCTION_COPY, copy_tensors_big.size(), copy_tensors_big.data(), &copy_big));

    {
        size_t symbol_size { 0 };
        LIBMATHDX_CHECK(
            cublasdxGetDeviceFunctionTraitStrSize(gemm_sa_sb_rc, CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL, &symbol_size));
        std::vector<char> symbol(symbol_size);
        LIBMATHDX_CHECK(cublasdxGetDeviceFunctionTraitStr(
            gemm_sa_sb_rc, CUBLASDX_DEVICE_FUNCTION_TRAIT_SYMBOL, symbol.size(), symbol.data()));
        printf("Device function %lld: symbol: %s\n", static_cast<long long int>(gemm_sa_sb_rc), symbol.data());
    }

    std::vector<cublasdxDeviceFunction> functions = { gemm_sa_sb_rc, init,  destroy, copy_smem,
                                                      copy_rmem,     clear, copy_big };

    /**
     * Compile the device function to lto
     */
    commondxCode code { 0 };
    LIBMATHDX_CHECK(commondxCreateCode(&code));
    LIBMATHDX_CHECK(
        commondxSetCodeOptionInt64s(code, COMMONDX_OPTION_TARGET_SM, target_sm_array.size(), target_sm_array.data()));
    LIBMATHDX_CHECK(cublasdxFinalizeDeviceFunctions(code, functions.size(), functions.data()));

    /**
     * Extract the LTOIR
     */
    std::vector<char> lto;
    size_t lto_size = 0;
    LIBMATHDX_CHECK(commondxGetCodeLTOIRSize(code, &lto_size));
    lto.resize(lto_size);
    LIBMATHDX_CHECK(commondxGetCodeLTOIR(code, lto_size, lto.data()));

    printf("Generated LTOIR for GEMM device functions, %zu bytes\n", lto.size());

    /**
     * Destroy handles
     */
    LIBMATHDX_CHECK(cublasdxDestroyTensor(smem_a));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(smem_b));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(acc_c));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(smem_c));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(rmem_c));
    LIBMATHDX_CHECK(cublasdxDestroyTensor(big_gmem));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(gemm_sa_sb_rc));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(init));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(destroy));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(copy_smem));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(copy_rmem));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(clear));
    LIBMATHDX_CHECK(cublasdxDestroyDeviceFunction(copy_big));
    LIBMATHDX_CHECK(commondxDestroyCode(code));
    LIBMATHDX_CHECK(cublasdxDestroyDescriptor(h));
}
