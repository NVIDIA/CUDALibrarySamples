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
     *   A = 2.10
     *   B = 1.20
     */

    data_type A = 2.1;
    data_type B = 1.2;
    data_type c = 2.1;
    data_type s = 1.2;

    printf("A\n");
    std::printf("%0.2f\n", A);
    printf("=====\n");

    printf("B\n");
    std::printf("%0.2f\n", B);
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasDrotg(cublasH, &A, &B, &c, &s));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   A = 2.42
     *   B = 0.50
     */

     printf("A\n");
     std::printf("%0.2f\n", A);
     printf("=====\n");
 
     printf("B\n");
     std::printf("%0.2f\n", B);
     printf("=====\n");

    /* free resources */
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}