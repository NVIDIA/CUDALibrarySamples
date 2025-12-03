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


#pragma once

#include <cublasLt.h>

/// Sample wrapper executing double precision GEMM with a predefined algorithm using cublasLtMatmul, nearly a drop-in
/// replacement for cublasDgemm, with addition of the workspace to support fp64 emulation.
///
/// Pointer mode is always host. To change it, configure the appropriate matmul descriptor attribute.
/// Matmul is not using cuBLAS handle's configuration of math mode (where tensor ops are implicitly allowed).
/// To change this, configure the appropriate attribute in the preference handle.
///
/// NOTE: This sample may not work on all architectures or all problem sizes.
void LtDgemmEmulated(cublasLtHandle_t ltHandle,
                     cublasOperation_t transa,
                     cublasOperation_t transb,
                     int m,
                     int n,
                     int k,
                     const double *alpha, /* host pointer */
                     const double *A,
                     int lda,
                     const double *B,
                     int ldb,
                     const double *beta, /* host pointer */
                     double *C,
                     int ldc,
                     void *workspace,
                     size_t workspaceSize,
                     cublasEmulationStrategy_t emulationStrategy,
                     cudaEmulationSpecialValuesSupport specialValuesSupport,
                     cudaEmulationMantissaControl mantissaControl,
                     int maxMantissaBitCount,
                     int mantissaBitOffset,
                     cudaStream_t stream);