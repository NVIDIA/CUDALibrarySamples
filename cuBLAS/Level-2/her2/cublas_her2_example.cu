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

    const int m = 2;
    const int n = 2;
    const int lda = m;

    /*
     *   A = | 1.1 + 1.2j | 2.3 + 2.4j |
     *       | 3.5 + 3.6j | 4.7 + 4.8j |
     *   x = | 5.1 + 6.2j | 7.3 + 8.4j |
     *   y = | 1.1 + 2.2j | 3.3 + 4.4j |
     */

    std::vector<data_type> A = {{1.1, 1.2}, {3.5, 3.6}, {2.3, 2.4}, {4.7, 4.8}};
    const std::vector<data_type> x = {{5.1, 6.2}, {7.3, 8.4}};
    const std::vector<data_type> y = {{1.1, 2.2}, {3.3, 4.4}};
    const data_type alpha = {1.0, 1.0};
    const int incx = 1;
    const int incy = 1;

    data_type *d_A = nullptr;
    data_type *d_x = nullptr;
    data_type *d_y = nullptr;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    printf("A\n");
    print_matrix(m, n, A.data(), lda);
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
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(data_type) * x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(data_type) * y.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, x.data(), sizeof(data_type) * x.size(), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y, y.data(), sizeof(data_type) * y.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasZher2(cublasH, uplo, n, &alpha, d_x, incx, d_y, incy, d_A, lda));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(A.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   A = | 48.40 + 0.00j |  81.72 + 24.84j |
     *       |  3.50 + 3.60j | 135.60 + 0.00j |
     */

    printf("A\n");
    print_matrix(m, n, A.data(), lda);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}