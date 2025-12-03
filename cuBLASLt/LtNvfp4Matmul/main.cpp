/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <cuda_fp4.h>
#include <cuda_bf16.h>

#include "sample_cublasLt_LtNvfp4Matmul.h"
#include "helpers.h"

int main() {
    TestBench<__nv_fp4_e2m1, __nv_fp4_e2m1, float, __nv_fp8_e4m3, float, __nv_bfloat16> props(
        CUBLAS_OP_T, CUBLAS_OP_N, 64, 128, 256, 2.0f, 1.0f, 32ULL * 1024 * 1024, 1,
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
        CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3);

    props.run([&props] {
        LtNvfp4Matmul(props.ltHandle, props.transa, props.transb, props.m, props.n, props.k, &props.alpha,
                      props.AscaleDev, props.Adev, props.lda, props.BscaleDev, props.Bdev, props.ldb, &props.beta,
                      props.CscaleDev, props.Cdev, props.ldc, props.DscaleDev, props.Ddev, props.ldd,
                      props.DOutscaleDev, props.workspace, props.workspaceSize, props.AScaleMode, props.BScaleMode,
                      props.CScaleMode, props.DScaleMode, props.DOutScaleMode);
    });

    return 0;
}