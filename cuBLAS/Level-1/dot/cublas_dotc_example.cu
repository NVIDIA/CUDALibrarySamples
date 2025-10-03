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

using data_type = cuDoubleComplex;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    /*
     *   A = | 1.1 + 1.2j | 2.3 + 2.4j | 3.5 + 3.6j | 4.7 + 4.8j |
     *   B = | 5.1 + 5.2j | 6.3 + 6.4j | 7.5 + 7.6j | 8.7 + 8.8j |
     */

    const std::vector<data_type> A = {{1.1, 1.2}, {2.3, 2.4}, {3.5, 3.6}, {4.7, 4.8}};
    const std::vector<data_type> B = {{5.1, 5.2}, {6.3, 6.4}, {7.5, 7.6}, {8.7, 8.8}};
    const int incx = 1;
    const int incy = 1;

    data_type result = {0.0, 0.0};

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;

    printf("A\n");
    print_vector(A.size(), A.data());
    printf("=====\n");

    printf("B\n");
    print_vector(B.size(), B.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasZdotc(cublasH, A.size(), d_A, incx, d_B, incy, &result));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   result = 178.44+-1.60j
     */

    printf("Result\n");
    printf("%0.2f+%0.2fj\n", result.x, result.y);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}