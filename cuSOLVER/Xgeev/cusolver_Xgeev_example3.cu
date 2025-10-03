/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;

    using data_type = cuDoubleComplex;

    const int64_t n = 3;
    const int64_t lda = 3; // lda >= n
    const int64_t ldr = n; // for validation

    /*
     *  A = | 2.0 + 1.0j | -1.0 + 0.0j | 1.0 + 2.0j |
     *      | 2.0 + 1.0j | -3.0 + 1.0j | 2.0 + 3.0j |
     *      | 1.0 + 2.0j | -1.0 + 2.0j | 0.0 + 1.0j |
     */
    std::vector<data_type> A(n*lda);
    A[0] = data_type{ 2.,1.};
    A[1] = data_type{ 2.,1.};
    A[2] = data_type{1., 2.};
    A[0 + lda] = data_type{ -1.,0.};
    A[1 + lda] = data_type{ -3.,1.};
    A[2 + lda] = data_type{ -1.,2.};
    A[0 + 2*lda] = data_type{ 1.,2.};
    A[1 + 2*lda] = data_type{ 2.,3.};
    A[2 + 2*lda] = data_type{ 0.,1.};

    std::vector<data_type> W(n);

    /* Compute only right eigenvectors.*/
    cusolverEigMode_t jobvl = CUSOLVER_EIG_MODE_NOVECTOR;
    cusolverEigMode_t jobvr = CUSOLVER_EIG_MODE_VECTOR;
    const int64_t ldvl = 1; // ldvl >= 1 if jobvl = CUSOLVER_EIG_MODE_NOVECTOR
    const int64_t ldvr = 3; // ldvr >= n if jobvr = CUSOLVER_EIG_MODE_VECTOR
    std::vector<data_type> VR(ldvr * n);

    data_type *d_A = nullptr;
    data_type *d_W = nullptr;
    data_type *d_VR = nullptr;
    data_type *d_VL = nullptr;
    data_type *d_R = nullptr; // for validation R = A * VR - VR * W

    int *d_info = nullptr;
    int info = 0;
    const data_type h_minus_one = data_type{-1.0,0.0};
    const data_type h_one = data_type{1.0,0.0};
    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    void *d_work = nullptr;
    void *h_work = nullptr;

    std::printf("A = (matlab base -1) \n");
    print_matrix(n, n, A.data(), lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(data_type) * W.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VR), sizeof(data_type) * VR.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_R), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                               cudaMemcpyHostToDevice, stream));

    /* step 3: query size of workspace and allocate host and device buffers*/
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

    /* step 4: compute eigenvalues and eigenvectors */
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

    CUDA_CHECK(cudaMemcpyAsync(VR.data(), d_VR, sizeof(data_type) * VR.size(),
               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(W.data(), d_W, sizeof(data_type) * W.size(),
               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int),
               cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
    } else if (0 < info) {
        std::printf("info = %d : Xgeev did not converge \n", info);
    } else {
        std::printf("info = %d : Xgeev converged\n", info);

        std::printf("eigenvalues = (matlab base -1) \n");
        print_matrix(n, 1, W.data(), 1);
        std::printf("=====\n");

        std::printf("VR = \n");
        print_matrix(n, n, VR.data(), ldvr);
        std::printf("=====\n");

        /* step 5: Verify the results */
        // Xgeev overwrites d_A. Reset d_A to the original input matrix.
        CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                                cudaMemcpyHostToDevice, stream));
        // R = VR * diag(W)
        CUBLAS_CHECK(cublasZdgmm(cublasH, CUBLAS_SIDE_RIGHT, n, n, d_VR, ldvr,
            d_W, 1, d_R, ldr));
        // R = R - A * VR
        CUBLAS_CHECK(cublasZgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, n, &h_minus_one, d_A, lda, d_VR, ldvr, &h_one, d_R, ldr));
        double dR_nrm = 0.0;
        CUBLAS_CHECK(cublasDznrm2_v2(cublasH, n*ldr, d_R, 1, &dR_nrm));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::printf("|A*VR - VR*diag(W)| = %E \n", dR_nrm);
    }

    return EXIT_SUCCESS;
}