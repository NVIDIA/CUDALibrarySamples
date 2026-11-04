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

#include <cublasLt.h>

/// Sample wrapper executing a dequantization (mixed-precision input) gemm with cublasLtMatmul
///
/// Either A or B holds quantized CUDA_R_8I values while the other input matrix, C, and D use the
/// full-precision 16-bit type (CUDA_R_16F or CUDA_R_16BF); aType, bType, and cType select which
/// operand is the quantized one
void LtDequantGemmSimple(cublasLtHandle_t ltHandle,
                         cublasOperation_t transa,
                         cublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float *alpha,
                         cudaDataType_t aType,
                         const void *A,
                         int lda,
                         cudaDataType_t bType,
                         const void *B,
                         int ldb,
                         const float *beta,
                         cudaDataType_t cType,
                         void *C,
                         int ldc,
                         void *workspace,
                         size_t workspaceSize);
