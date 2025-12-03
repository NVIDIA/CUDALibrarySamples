/*
 * SPDX-FileCopyrightText: Copyright (c) 2020 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cublasLt.h>

#include "sample_cublasLt_LtSgemmSimpleAutoTuning.h"
#include "helpers.h"

void printAlgo(const cublasLtMatmulAlgo_t &algo) {
    int algoId, tile, swizzle, customOption, numSplitsK, reductionScheme;

    checkCublasStatus(
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), NULL));
    checkCublasStatus(
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &numSplitsK,
                                                           sizeof(numSplitsK), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                           &reductionScheme, sizeof(reductionScheme), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle,
                                                           sizeof(swizzle), NULL));
    checkCublasStatus(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &customOption,
                                                           sizeof(customOption), NULL));

    printf("algo={ Id=%d, tileIdx=%d splitK=%d reduc=%d swizzle=%d custom=%d }\n", algoId, tile, numSplitsK,
           reductionScheme, swizzle, customOption);
}

int main() {
    TestBench<float> props(CUBLAS_OP_N, CUBLAS_OP_N, 1024, 1024, 1024, 2.0f, 0.0f, 1024 * 1024 * 4);

    cublasLtMatmulAlgo_t algo;

    props.run([&props, &algo] {
        LtSgemmSimpleAutoTuning(props.ltHandle, props.transa, props.transb, props.m, props.n, props.k, &props.alpha,
                                props.Adev, props.lda, props.Bdev, props.ldb, &props.beta, props.Cdev, props.ldc,
                                props.workspace, props.workspaceSize, algo);
    });

    printAlgo(algo);

    return 0;
}