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

    const int n = 2;

    /*
     *   AP = | 1.1 + 1.2j | 2.3 + 2.4j |
     *        |            | 4.7 + 4.8j |
     *   x  = | 5.1 + 6.2j | 7.3 + 8.4j |
     */

    std::vector<data_type> AP = {{1.1, 1.2}, {3.5, 3.6}, {4.7, 4.8}};
    const std::vector<data_type> x = {{5.1, 6.2}, {7.3, 8.4}};

    const double alpha = 1.0;
    const int incx = 1;

    data_type *d_AP = nullptr;
    data_type *d_x = nullptr;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    printf("AP\n");
    print_packed_matrix(uplo, n, AP.data());
    printf("=====\n");

    printf("x\n");
    print_vector(x.size(), x.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_AP), sizeof(data_type) * AP.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(data_type) * x.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_AP, AP.data(), sizeof(data_type) * AP.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(data_type) * x.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasZhpr(cublasH, uplo, n, &alpha, d_x, incx, d_AP));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(AP.data(), d_AP, sizeof(data_type) * AP.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   AP = | 65.55 +   0.00j  92.81 +   6.02j |
     *        |                 128.55 +   0.00j |
     */

    printf("AP\n");
    print_packed_matrix(uplo, n, AP.data());
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_AP));
    CUDA_CHECK(cudaFree(d_x));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}