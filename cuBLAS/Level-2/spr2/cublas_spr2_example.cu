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

    const int n = 2;

    /*
     *   AP = | 1.0 3.0 |
     *        |     4.0 |
     *   x  = | 5.0 6.0 |
     *   y  = | 7.0 8.0 |
     */

    std::vector<data_type> AP = {1.0, 3.0, 4.0};
    const std::vector<data_type> x = {5.0, 6.0};
    const std::vector<data_type> y = {7.0, 8.0};
    const data_type alpha = 1.0;
    const int incx = 1;
    const int incy = 1;

    data_type *d_AP = nullptr;
    data_type *d_x = nullptr;
    data_type *d_y = nullptr;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    printf("AP\n");
    print_packed_matrix(uplo, n, AP.data());
    printf("=====\n");

    printf("x\n");
    print_vector(x.size(), x.data());
    printf("=====\n");

    printf("y\n");
    print_vector(y.size(), y.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_AP), sizeof(data_type) * AP.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(data_type) * x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(data_type) * y.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_AP, AP.data(), sizeof(data_type) * AP.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(data_type) * x.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y, y.data(), sizeof(data_type) * y.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasDspr2(cublasH, uplo, n, &alpha, d_x, incx, d_y, incy, d_AP));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(AP.data(), d_AP, sizeof(data_type) * AP.size(),
                               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   AP = | 71.0  85.0 |
     *        |      100.0 |
     */

    printf("AP\n");
    print_packed_matrix(uplo, n, AP.data());
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_AP));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}