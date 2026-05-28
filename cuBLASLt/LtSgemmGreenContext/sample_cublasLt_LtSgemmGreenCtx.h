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

/// Sample wrapper executing single precision gemm with cublasLtMatmul
/// with the addition of the usage of a green context with a subset of SMs.
/// The green context is used through the stream passed to the available API,
/// which decides which algorithm to use.
void LtSgemmGCtx(cublasLtHandle_t ltHandle,
                 cublasOperation_t transa,
                 cublasOperation_t transb,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *A,
                 int lda,
                 const float *B,
                 int ldb,
                 const float *beta, /* host pointer */
                 float *C,
                 int ldc,
                 void *workspace,
                 size_t workspaceSize,
                 unsigned int minGreenContextSmCount,
                 cudaStream_t primaryContextStream);
