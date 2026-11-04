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

#include <cstdio>

#include "sample_cublasLt_LtDequantGemmSimple.h"
#include "helpers.h"

int main() {
    {
        // A is the quantized operand: the K-inner requirement makes this a TN gemm
        printf("Dequantization gemm with a quantized A matrix: int8 A x fp16 B -> fp16 D\n");
        MixedTestBench<int8_t, __half, __half, float> props(CUBLAS_OP_T, CUBLAS_OP_N, 128, 128, 128, 2.0f, 0.0f,
                                                            32 * 1024 * 1024);

        props.run([&props] {
            LtDequantGemmSimple(props.ltHandle, props.transa, props.transb, props.m, props.n, props.k, &props.alpha,
                                CUDA_R_8I, props.Adev, props.lda, CUDA_R_16F, props.Bdev, props.ldb, &props.beta,
                                CUDA_R_16F, props.Cdev, props.ldc, props.workspace, props.workspaceSize);
        });
    }

    {
        // B is the quantized operand: the K-inner requirement makes this an NN gemm
        printf("Dequantization gemm with a quantized B matrix: fp16 A x int8 B -> fp16 D\n");
        MixedTestBench<__half, int8_t, __half, float> props(CUBLAS_OP_N, CUBLAS_OP_N, 128, 128, 128, 2.0f, 0.0f,
                                                            32 * 1024 * 1024);

        props.run([&props] {
            LtDequantGemmSimple(props.ltHandle, props.transa, props.transb, props.m, props.n, props.k, &props.alpha,
                                CUDA_R_16F, props.Adev, props.lda, CUDA_R_8I, props.Bdev, props.ldb, &props.beta,
                                CUDA_R_16F, props.Cdev, props.ldc, props.workspace, props.workspaceSize);
        });
    }

    return 0;
}
