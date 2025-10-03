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

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;

    using data_type = double;

    const int m = 3;
    const int lda = m;
    /*
     *       | 3.5 0.5 0.0 |
     *   A = | 0.5 3.5 0.0 |
     *       | 0.0 0.0 2.0 |
     *
     */
    const std::vector<data_type> A = {3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};
    const std::vector<data_type> lambda = {2.0, 3.0, 4.0};

    std::vector<data_type> V(lda * m, 0); // eigenvectors
    std::vector<data_type> W(m, 0);       // eigenvalues

    data_type *d_A = nullptr;
    data_type *d_W = nullptr;
    data_type vl;
    data_type vu;
    int64_t h_meig = 0;
    int *d_info = nullptr;

    int info = 0;

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr; /* device workspace */
    size_t workspaceInBytesOnHost = 0; /* size of workspace */
    void *h_work = nullptr; /* host workspace for */

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, m, A.data(), lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(data_type) * W.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));

    // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolverEigRange_t range = CUSOLVER_EIG_RANGE_ALL;

    CUSOLVER_CHECK(cusolverDnXsyevdx_bufferSize(
        cusolverH, params, jobz, range, uplo, m, traits<data_type>::cuda_data_type, d_A, lda, &vl,
        &vu, 0L, 0L, &h_meig, traits<data_type>::cuda_data_type, d_W,
        traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    // step 4: compute spectrum
    CUSOLVER_CHECK(cusolverDnXsyevdx(
        cusolverH, params, jobz, range, uplo, m, traits<data_type>::cuda_data_type, d_A, lda, &vl,
        &vu, 0L, 0L, &h_meig, traits<data_type>::cuda_data_type, d_W, traits<data_type>::cuda_data_type,
        d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));

    CUDA_CHECK(cudaMemcpyAsync(V.data(), d_A, sizeof(data_type) * V.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(W.data(), d_W, sizeof(data_type) * W.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xsyevdx: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    std::printf("eigenvalue = (matlab base-1), ascending order\n");
    int idx = 1;
    for (auto const &i : W) {
        std::printf("W[%i] = %E\n", idx, i);
        idx++;
    }

    std::printf("V = (matlab base-1)\n");
    print_matrix(m, m, V.data(), lda);
    std::printf("=====\n");

    // step 4: check eigenvalues
    data_type lambda_sup = 0;
    for (int i = 0; i < m; i++) {
        data_type error = fabs(lambda[i] - W[i]);
        lambda_sup = (lambda_sup > error) ? lambda_sup : error;
    }
    std::printf("|lambda - W| = %E\n", lambda_sup);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}