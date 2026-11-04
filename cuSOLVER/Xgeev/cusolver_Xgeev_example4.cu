/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cusolverDn.h>

#include "cusolver_utils.h"


using data_type = cuDoubleComplex;


int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;

    int64_t n;
    int64_t lda; // lda >= n, we set lda = n

    /* Square sample matrices (download commands in the README) */
    const char *sample_matrices[] = {
        "conf5_0-4x4-22.mtx", // well-separated eigenvalues,
                              // moderate departure of normality
        "cavity24.mtx"        // clustered eigenvalues
    };

    /* Compute only right eigenvectors. */
    cusolverEigMode_t jobvl = CUSOLVER_EIG_MODE_NOVECTOR;
    cusolverEigMode_t jobvr = CUSOLVER_EIG_MODE_VECTOR;
    const int64_t ldvl = 1; // ldvl >= 1 if jobvl = CUSOLVER_EIG_MODE_NOVECTOR
    int64_t ldvr;           // ldvr >= n if jobvr = CUSOLVER_EIG_MODE_VECTOR

    std::vector<data_type> A;

    data_type *d_A = nullptr;
    data_type *d_W = nullptr;
    data_type *d_VR = nullptr;
    data_type *d_VL = nullptr;
    data_type *d_R = nullptr; // for validation R = VR * diag(W) - A * VR

    int *d_info = nullptr;
    int info = 0;
    const data_type h_minus_one = data_type{-1.0,0.0};
    const data_type h_one = data_type{1.0,0.0};
    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    void *d_work = nullptr;
    void *h_work = nullptr;

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: select mixed precision with iterative refinement as the algorithm */
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GEEV, CUSOLVER_ALG_1));

    for (const char *mtx_file : sample_matrices) {
        /* step 3: prepare input data */
        // A is n-by-n with leading dimension lda
        read_mtx_and_convert_to_dense(mtx_file, n, n, A, lda);
        std::printf("%s: n = %ld\n", mtx_file, n);
        ldvr = n; // VR is n-by-n with leading dimension ldvr

        /* step 4: allocate device memory and copy A to device */
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(data_type) * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VR), sizeof(data_type) * ldvr * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(data_type) * n * n));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
        CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                                   cudaMemcpyHostToDevice, stream));

        /* step 5: query size of workspace and allocate host and device buffers */
        CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
            cusolverH,
            params,
            jobvl,
            jobvr,
            n,
            traits<data_type>::cuda_data_type,
            d_A,
            lda,
            traits<data_type>::cuda_data_type,
            d_W,
            traits<data_type>::cuda_data_type,
            d_VL,
            ldvl,
            traits<data_type>::cuda_data_type,
            d_VR,
            ldvr,
            traits<data_type>::cuda_data_type,
            &workspaceInBytesOnDevice,
            &workspaceInBytesOnHost));

        CUDA_CHECK(cudaMallocHost(&h_work, workspaceInBytesOnHost)); // pinned host memory for best performance
        CUDA_CHECK(cudaMalloc(&d_work, workspaceInBytesOnDevice));

        /* step 6: compute eigenvalues and eigenvectors */
        CUSOLVER_CHECK(cusolverDnXgeev(
            cusolverH,
            params,
            jobvl, jobvr,
            n,
            traits<data_type>::cuda_data_type,
            d_A,
            lda,
            traits<data_type>::cuda_data_type,
            d_W,
            traits<data_type>::cuda_data_type,
            d_VL,
            ldvl,
            traits<data_type>::cuda_data_type,
            d_VR,
            ldvr,
            traits<data_type>::cuda_data_type,
            d_work,
            workspaceInBytesOnDevice,
            h_work,
            workspaceInBytesOnHost,
            d_info));

        CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int),
            cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (0 > info) {
            std::printf("%d-th parameter is wrong\n", -info);
        } else if (0 < info) {
            std::printf("info = %d : Xgeev did not converge\n", info);
        } else {
            std::printf("info = %d : Xgeev converged\n", info);

            /* step 7: verify the results */
            // Xgeev overwrites d_A. Reset d_A to the original input matrix.
            CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                                       cudaMemcpyHostToDevice, stream));

            // R = VR * diag(W)
            CUBLAS_CHECK(cublasZdgmm(cublasH, CUBLAS_SIDE_RIGHT, n, n, d_VR, ldvr,
                d_W, 1, d_R, n));
            // R = R - A * VR
            CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n, &h_minus_one, d_A, lda, d_VR, ldvr, &h_one, d_R, n));
            double dR_nrm = 0.0;
            CUBLAS_CHECK(cublasDznrm2(cublasH, n * n, d_R, 1, &dR_nrm));
            double dA_nrm = 0.0;
            CUBLAS_CHECK(cublasDznrm2(cublasH, n * lda, d_A, 1, &dA_nrm));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("|A*VR - VR*diag(W)|/|A| = %E\n", dR_nrm / dA_nrm);
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_W));
        CUDA_CHECK(cudaFree(d_VR));
        CUDA_CHECK(cudaFree(d_R));
        CUDA_CHECK(cudaFree(d_info));
        CUDA_CHECK(cudaFree(d_work));
        CUDA_CHECK(cudaFreeHost(h_work));
    }

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
