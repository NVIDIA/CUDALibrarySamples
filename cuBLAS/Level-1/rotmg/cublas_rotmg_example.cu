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
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    /*
     *   A = 1.0
     *   B = 5.0
     *   X = 2.1
     *   Y = 1.2
     */

    data_type A = 1.0;
    data_type B = 5.0;
    data_type X = 2.1;
    data_type Y = 1.2;
    std::vector<data_type> param = {1.0, 5.0, 6.0, 7.0, 8.0}; // flag = param[0]

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;

    printf("A\n");
    std::printf("%0.2f\n", A);
    printf("=====\n");

    printf("B\n");
    std::printf("%0.2f\n", B);
    printf("=====\n");

    printf("X\n");
    std::printf("%0.2f\n", X);
    printf("=====\n");

    printf("Y\n");
    std::printf("%0.2f\n", Y);
    printf("=====\n");

    printf("param\n");
    print_vector(param.size(), param.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasDrotmg(cublasH, &A, &B, &X, &Y, param.data()));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   A = 3.10
     *   B = 0.62
     *   X = 1.94
     */

    printf("A\n");
    std::printf("%0.2f\n", A);
    printf("=====\n");

    printf("B\n");
    std::printf("%0.2f\n", B);
    printf("=====\n");

    printf("X\n");
    std::printf("%0.2f\n", X);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}