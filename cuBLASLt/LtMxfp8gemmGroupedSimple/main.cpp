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

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "helpers.h"
#include "sample_cublasLt_LtMxfp8gemmGroupedSimple.h"

int main() {
    // MXFP8 grouped GEMM with VEC32_UE8M0 block scaling
    // Uses __nv_fp8_e4m3 for A, B and __nv_bfloat16 for C, D
    // ScaleType = __nv_fp8_e8m0 (E8M0 scales for A, B), host alpha/beta shared across groups
    TestBench<__nv_fp8_e4m3, __nv_bfloat16, float, __nv_fp8_e8m0, __nv_fp8_e8m0, __nv_bfloat16> props(
        CUBLAS_OP_T, CUBLAS_OP_N, 64, 128, 256, 2.0f, 0.0f, 32ULL * 1024 * 1024, 2,
        __nv_fp8_e8m0{2.0f}, __nv_fp8_e8m0{0.5f}, __nv_fp8_e8m0{1.0f}, __nv_fp8_e8m0{1.0f},
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, false, true, true);

    props.run([&props] {
        LtMxfp8gemmGroupedSimple(
            props.ltHandle, props.transa, props.transb, props.mArrayDev, props.avgM, props.nArrayDev, props.avgN,
            props.kArrayDev, props.avgK, &props.alpha,
            reinterpret_cast<const __nv_fp8_e8m0 *const *>(props.AscalePtrArrayDev), props.APtrArrayDev,
            props.ldaArrayDev, reinterpret_cast<const __nv_fp8_e8m0 *const *>(props.BscalePtrArrayDev),
            props.BPtrArrayDev, props.ldbArrayDev, &props.beta, props.CPtrArrayDev, props.ldcArrayDev,
            props.DPtrArrayDev, props.lddArrayDev, props.N, props.workspace, props.workspaceSize, props.AScaleMode,
            props.BScaleMode);
    });

    return 0;
}
