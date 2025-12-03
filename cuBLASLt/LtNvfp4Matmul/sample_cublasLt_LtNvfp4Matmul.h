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
#include <cublasLt.h>

#include "helpers.h"

/// Sample wrapper executing nvfp4 matmul with cublasLtMatmul, with addition of per-tensor block scaling, and
/// the workspace to support split-K algorithms.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
void LtNvfp4Matmul(cublasLtHandle_t ltHandle,
                   cublasOperation_t transa,
                   cublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float *alpha,           /* host pointer */
                   const __nv_fp8_e4m3 *a_scale, /* device pointer */
                   const typename StorageType<__nv_fp4_e2m1>::type *A,
                   int lda,
                   const __nv_fp8_e4m3 *b_scale, /* device pointer */
                   const typename StorageType<__nv_fp4_e2m1>::type *B,
                   int ldb,
                   const float *beta,
                   const __nv_fp8_e4m3 *c_scale, /* device pointer */
                   __nv_bfloat16 *C,
                   int ldc,
                   const float *d_scale, /* device pointer */
                   typename StorageType<__nv_fp4_e2m1>::type *D,
                   int ldd,
                   __nv_fp8_e4m3 *d_out_scale, /* device pointer */
                   void *workspace,
                   size_t workspaceSize,
                   cublasLtMatmulMatrixScale_t AScaleMode,
                   cublasLtMatmulMatrixScale_t BScaleMode,
                   cublasLtMatmulMatrixScale_t CScaleMode,
                   cublasLtMatmulMatrixScale_t DScaleMode,
                   cublasLtMatmulMatrixScale_t DOutScaleMode);
