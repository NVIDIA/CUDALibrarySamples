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


#include <vector>
#include <cstdlib>

#include <cuda_runtime_api.h>
#include <cublasLt.h>

#include "sample_cublasLt_LtDgemmEmulated.h"
#include "helpers.h"

/* 
 * This example computes the emulated DGEMM in fixed mode with 55 retained mantissa bits.
 *
 * Emulated DGEMM environment variables for cublasLt:
 *  - CUBLAS_EMULATE_DOUBLE_PRECISION: A value of 1 will allow cublasLt to utilize FP64 emulation algorithms in double precision
 *                                     and double complex precision routines.
 *
 *  - CUBLAS_EMULATION_STRATEGY: This supports two values: (1) performant -- the default value which enables a layer
 *                               of heuristics to pick between emulation and native algorithms to choose the most
 *                               performant option (2) eager -- a value which will leverage emulation whenever possible.
 *
 *  - CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT: Number of mantissa bits to be used for fixed emulation.  When set,
 *                                                    if an emulation algorithm is used, it will use fixed emulation instead
 *                                                    of emulation with the number of mantissa bits specified by the user.
 *                                                    If an invalid value is set, the user-provided value will be ignored.
 */
int main() {
    /*
     * If you see the following error and trace bundled together:
     * [cublasLt][52944][Error][cublasLtMatmulAlgoGetHeuristic] Failed to query heuristics.
     *
     * It is very likely that you need a larger workspace size.
     * The emulated DGEMM kernel requires a large amount of workspace.
     * 
     * See getFixedPointWorkspaceSizeInBytes() in https://docs.nvidia.com/cuda/cublas/index.html#fixed-point-workspace-requirements 
     * for more details on how much workspace is required.
     */
    TestBench<double> props(CUBLAS_OP_N, CUBLAS_OP_N, 16, 16, 64, 2.0f, 0.0f, (size_t)1024 * 1024 * 1024);

    // Allow emulation to run whenever possible
    cublasEmulationStrategy_t emulationStrategy = CUBLAS_EMULATION_STRATEGY_EAGER;

    // Ensure that emulation supports the default special values
    cudaEmulationSpecialValuesSupport specialValuesSupport = CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT;

    // Choose the fixed emulation mode 
    cudaEmulationMantissaControl mantissaControl = CUDA_EMULATION_MANTISSA_CONTROL_FIXED;

    // For a fixed mantissa control, the max mantissa bit count is the number of retained mantissa bits
    int maxMantissaBitCount = 55;

    // For a fixed mantissa control, this parameter has no effect
    int mantissaBitOffset = 0;

    props.run([&props, &emulationStrategy, &specialValuesSupport, &mantissaControl, &maxMantissaBitCount, &mantissaBitOffset] {
        LtDgemmEmulated(props.ltHandle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                props.m,
                props.n,
                props.k,
                &props.alpha,
                props.Adev,
                props.m,
                props.Bdev,
                props.k,
                &props.beta,
                props.Cdev,
                props.m,
                props.workspace,
                props.workspaceSize,
                emulationStrategy,
                specialValuesSupport,
                mantissaControl,
                maxMantissaBitCount,
                mantissaBitOffset,
                props.stream);
    });

    return 0;
}