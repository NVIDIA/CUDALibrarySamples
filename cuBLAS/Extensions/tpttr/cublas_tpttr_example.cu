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

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = m;

    /*
     *   AP = | 1.0 | 2.0 |
     *        | 3.0 | 4.0 |
     */

    const std::vector<data_type> AP = {1.0, 3.0, 2.0, 4.0};
    std::vector<data_type> A(n * k);

    data_type *d_AP = nullptr;
    data_type *d_A = nullptr;

    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    printf("AP\n");
    print_matrix(m, k, AP.data(), lda);
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_AP), sizeof(data_type) * AP.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_AP, AP.data(), sizeof(data_type) * AP.size(),
                               cudaMemcpyHostToDevice, stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasDtpttr(cublasH, uplo, n, d_AP, d_A, lda));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(A.data(), d_A, sizeof(data_type) * A.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   A = | 1.0 | 3.0 |
     *       | 0.0 | 2.0 |
     */

    printf("A\n");
    print_matrix(m, n, A.data(), lda);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_AP));
    CUDA_CHECK(cudaFree(d_A));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}