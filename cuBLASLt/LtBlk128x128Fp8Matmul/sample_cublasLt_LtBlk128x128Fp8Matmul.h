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


#include <cuda_fp8.h>
#include <cublasLt.h>

/// Sample wrapper executing matmul with cublasLtMatmul, with addition of hopper block scaling, and
/// the workspace to support split-K algorithms.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
void LtBlk128x128Fp8Matmul(cublasLtHandle_t ltHandle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *a_scale, /* device pointer */
                 const __nv_fp8_e4m3 *A,
                 int lda,
                 const float *b_scale, /* device pointer */
                 const __nv_fp8_e4m3 *B,
                 int ldb,
                 const float *beta, /* host pointer */
                 __nv_bfloat16 *C,
                 int ldc,
                 __nv_bfloat16 *D,
                 int ldd,
                 void *workspace,
                 size_t workspaceSize,
                 cublasLtMatmulMatrixScale_t AScaleMode,
                 cublasLtMatmulMatrixScale_t BScaleMode);