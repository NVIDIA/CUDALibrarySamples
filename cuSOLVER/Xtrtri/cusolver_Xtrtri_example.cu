/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

template <typename T>
void trtri(cusolverDnHandle_t handle, cusolver_int_t n, T *d_A, cusolver_int_t lda,
           cublasFillMode_t uplo, cublasDiagType_t diag, int *d_info) {
    void *d_work = nullptr;
    size_t d_lwork = 0;
    void *h_work = nullptr;
    size_t h_lwork = 0;

    try {
        printf("Quering required device and host workspace size...\n");
        CUSOLVER_CHECK(cusolverDnXtrtri_bufferSize(handle, uplo, diag, n, traits<T>::cuda_data_type,
                                                   reinterpret_cast<void *>(d_A), lda, &d_lwork,
                                                   &h_lwork));

        printf("Allocating required device workspace...\n");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), d_lwork));

        printf("Allocating required host workspace...\n");
        if (h_lwork) {
            h_work = malloc(h_lwork);
            if (h_work == nullptr) {
                throw std::bad_alloc();
            }
        }

        printf("Computing the inverse of a %s triangular matrix...\n",
               (uplo == CUBLAS_FILL_MODE_UPPER ? "upper" : "lower"));
        CUSOLVER_CHECK(cusolverDnXtrtri(handle, uplo, diag, n, traits<T>::cuda_data_type, d_A, lda,
                                        d_work, d_lwork, h_work, h_lwork, d_info));
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
    }

    CUDA_CHECK(cudaFree(d_work));
    free(h_work);
}

// calculate |I - A * A^-1| and compare with eps
template <typename T>
void residual_check(cusolver_int_t n, T *d_A, T *d_A_inv, cusolver_int_t lda, double eps) {
    // create identity matrix
    T *h_A_res = (T *)calloc(n * lda, sizeof(T));

    for (cusolver_int_t i = 0; i < n; i++) {
        h_A_res[i * lda + i] = T(1);
    }

    T alpha = -1;
    T beta = 1;

    T *d_A_res;
    CUDA_CHECK(cudaMalloc(&d_A_res, sizeof(T) * n * lda));
    CUDA_CHECK(cudaMemcpy(d_A_res, h_A_res, sizeof(T) * n * lda, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                              reinterpret_cast<void *>(&alpha), reinterpret_cast<void *>(d_A),
                              traits<T>::cuda_data_type, lda, reinterpret_cast<void *>(d_A_inv),
                              traits<T>::cuda_data_type, lda, reinterpret_cast<void *>(&beta),
                              reinterpret_cast<void *>(d_A_res), traits<T>::cuda_data_type, lda,
                              traits<T>::cuda_data_type, CUBLAS_GEMM_DEFAULT));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_A_res, d_A_res, sizeof(T) * n * lda, cudaMemcpyDeviceToHost));

    double A_res_norm = 0.0;
    for (cusolver_int_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (cusolver_int_t j = 0; j < n; j++) {
            sum += traits<T>::abs(h_A_res[i + j * lda]);
        }
        A_res_norm = std::max(A_res_norm, sum);
    }

    printf("Check: %s\n", (A_res_norm > eps ? "FAILED" : "PASSED"));

    CUBLAS_CHECK(cublasDestroy(handle));

    free(h_A_res);
    CUDA_CHECK(cudaFree(d_A_res));
}

int main(int argc, char *argv[]) {
    using data_type = double;
    const double eps = 1.e-15;

    cusolverDnHandle_t handle;

    cudaStream_t stream;

    cusolver_int_t n = 1000;
    cusolver_int_t lda = n + 1;

    data_type *d_A = nullptr;
    data_type *d_A_inv = nullptr;
    data_type *h_A = nullptr;
    int *d_info;
    int h_info;

    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
    const cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

    printf("Generating random diagonal dominant matrix...\n");
    generate_random_matrix<data_type>(n, lda, &h_A, &lda);
    make_diag_dominant_matrix<data_type>(n, lda, h_A, lda);

    // zero lower triangle
    for (cusolver_int_t j = 0; j < n; j++) {
        for (cusolver_int_t i = j + 1; i < n; i++) {
            h_A[j * lda + i] = 0;
        }
    }

    printf("Initializing required CUDA and cuSOLVER miscelaneous variables...\n");
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

    printf("Allocating required device memory...\n");
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * lda * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A_inv), sizeof(data_type) * lda * n));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    printf("Copying input data to the device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(data_type) * lda * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_inv, d_A, sizeof(data_type) * lda * n, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));

    trtri(handle, n, d_A_inv, lda, uplo, diag, d_info);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("Copying information back to the host...\n");
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Checking returned information...\n");
    if (h_info > 0) {
        fprintf(stderr, "warning: leading minor of order %d is not p.d.\n", h_info);
    } else if (h_info < 0) {
        fprintf(stderr, "error: %d-th argument had an illegal value\n", h_info);
    }

    printf("Verifying results...\n");
    residual_check(n, d_A, d_A_inv, lda, eps);

    printf("Destroying CUDA and cuSOLVER miscelaneous variables...\n");
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    printf("Freeing memory...\n");
    free(h_A);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_A_inv));
    CUDA_CHECK(cudaFree(d_info));

    printf("Done...\n");

    return EXIT_SUCCESS;
}
