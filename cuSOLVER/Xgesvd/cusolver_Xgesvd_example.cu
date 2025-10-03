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
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;

    using data_type = double;

    const int64_t m = 3;
    const int64_t n = 2;
    const int64_t lda = m;  // lda >= m
    const int64_t ldu = m;  // ldu >= m
    const int64_t ldvt = n; // ldvt >= n if jobu = 'A'

    /*
     *       | 1 2 |
     *   A = | 4 5 |
     *       | 2 1 |
     */

    const std::vector<data_type> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    std::vector<data_type> U(ldu * m, 0);
    std::vector<data_type> VT(ldvt * n, 0);
    std::vector<data_type> S(n, 0);
    std::vector<data_type> S_exact = {7.065283497082729,
                                      1.040081297712078}; // exact singular values

    data_type *d_A = nullptr;
    data_type *d_S = nullptr;  // singular values
    data_type *d_U = nullptr;  // left singular vectors
    data_type *d_VT = nullptr; // right singular vectors
    int *d_info = nullptr;
    data_type *d_work = nullptr;
    data_type *h_work = nullptr;
    data_type *d_rwork = nullptr;
    data_type *d_W = nullptr;  // W = S*VT

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    int info = 0;
    const data_type h_one = 1;
    const data_type h_minus_one = -1;

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, n, A.data(), lda);
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
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(data_type) * S.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(data_type) * U.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VT), sizeof(data_type) * VT.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(data_type) * lda * n));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * lda * n, cudaMemcpyHostToDevice,
                               stream));

    signed char jobu = 'A';  // all m columns of U
    signed char jobvt = 'A'; // all n rows of VT

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnXgesvd_bufferSize(
        cusolverH,
        params,
        jobu,
        jobvt,
        m,
        n,
        traits<data_type>::cuda_data_type,
        d_A,
        lda,
        traits<data_type>::cuda_data_type,
        d_S,
        traits<data_type>::cuda_data_type,
        d_U,
        ldu,
        traits<data_type>::cuda_data_type,
        d_VT,
        ldvt,
        traits<data_type>::cuda_data_type,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<data_type *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    /* step 4: compute SVD */
    CUSOLVER_CHECK(cusolverDnXgesvd(
        cusolverH,
        params,
        jobu,
        jobvt,
        m,
        n,
        traits<data_type>::cuda_data_type,
        d_A,
        lda,
        traits<data_type>::cuda_data_type,
        d_S,
        traits<data_type>::cuda_data_type,
        d_U,
        ldu,
        traits<data_type>::cuda_data_type,
        d_VT,
        ldvt,
        traits<data_type>::cuda_data_type,
        d_work,
        workspaceInBytesOnDevice,
        h_work,
        workspaceInBytesOnHost,
        d_info));

    CUDA_CHECK(cudaMemcpyAsync(U.data(), d_U, sizeof(data_type) * U.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(VT.data(), d_VT, sizeof(data_type) * VT.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(S.data(), d_S, sizeof(data_type) * S.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xgesvd: info = %d\n", info);
    if (0 == info) {
        std::printf("Xgesvd converges \n");
    } else if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    } else {
        std::printf("WARNING: info = %d : Xgesvd does not converge \n", info);
    }
    std::printf("=====\n");

    std::printf("S = (matlab base-1)\n");
    print_matrix(n, 1, S.data(), lda);
    std::printf("=====\n");

    std::printf("U = (matlab base-1)\n");
    print_matrix(m, m, U.data(), ldu);
    std::printf("=====\n");

    std::printf("VT = (matlab base-1)\n");
    print_matrix(n, n, VT.data(), ldvt);
    std::printf("=====\n");

    // step 5: measure error of singular value
    double ds_sup = 0;
    for (int j = 0; j < n; j++) {
        double err = fabs(S[j] - S_exact[j]);
        ds_sup = (ds_sup > err) ? ds_sup : err;
    }
    std::printf("|S - S_exact| = %E \n", ds_sup);

    // step 6: |A - U*S*VT|
    // W = S*VT
    CUBLAS_CHECK(cublasDdgmm(cublasH, CUBLAS_SIDE_LEFT, n, n, d_VT, ldvt, d_S, 1, d_W, lda));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));

    CUBLAS_CHECK(cublasDgemm_v2(cublasH,
                                CUBLAS_OP_N,  // U
                                CUBLAS_OP_N,  // W
                                m,            // number of rows of A
                                n,            // number of columns of A
                                n,            // number of columns of U
                                &h_minus_one, /* host pointer */
                                d_U,          // U
                                ldu,
                                d_W,         // W
                                lda, &h_one, /* hostpointer */
                                d_A, lda));

    double dR_fro = 0.0;
    CUBLAS_CHECK(cublasDnrm2_v2(cublasH, A.size(), d_A, 1, &dR_fro));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("|A - U*S*VT| = %E \n", dR_fro);

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_VT));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_rwork));
    CUDA_CHECK(cudaFree(d_W));
    free(h_work);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}