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
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    using data_type = double;

    const int64_t m = 3;
    const int64_t n = 2;
    const int64_t lda = m;
    /*
     *       | 1 2  |
     *   A = | 4 5  |
     *       | 2 1  |
     */

    const std::vector<data_type> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    std::vector<data_type> U(lda * m, 0);
    std::vector<data_type> V(lda * n, 0);
    std::vector<data_type> S(n, 0);
    std::vector<data_type> S_exact = {7.065283497082729, 1.040081297712078};

    data_type *d_A = nullptr;
    data_type *d_S = nullptr;
    data_type *d_U = nullptr;
    data_type *d_V = nullptr;
    int *d_info = nullptr;
    data_type *d_W = nullptr; // W = S*VT

    int info = 0;
    const double h_one = 1;
    const double h_minus_one = -1;

    size_t d_lwork = 0;     /* size of workspace */
    void *d_work = nullptr; /* device workspace for getrf */
    size_t h_lwork = 0;     /* size of workspace */
    void *h_work = nullptr; /* host workspace for getrf */

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, n, A.data(), lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(data_type) * S.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(data_type) * U.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(data_type) * V.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(data_type) * lda * n));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    // const int econ = 0; /* compute 3-by-3 U  */
    const int econ = 1; /* compute 3-by-2 U  */
    double h_err_sigma;

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
        cusolverH, NULL,                                     /* params */
        jobz, econ, m, n, traits<data_type>::cuda_data_type, /* dataTypeA */
        d_A, lda, traits<data_type>::cuda_data_type,         /* dataTypeS */
        d_S, traits<data_type>::cuda_data_type,              /* dataTypeU */
        d_U, lda,                                            /* ldu */
        traits<data_type>::cuda_data_type,                   /* dataTypeV */
        d_V, lda,                                            /* ldv */
        traits<data_type>::cuda_data_type,                   /* computeType */
        &d_lwork, &h_lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));

    if (0 < h_lwork) {
        h_work = reinterpret_cast<void *>(malloc(h_lwork));
        if (d_work == nullptr) {
            throw std::runtime_error("Error: d_work not allocated.");
        }
    }

    /* step 4: compute SVD */
    CUSOLVER_CHECK(cusolverDnXgesvdp(cusolverH, NULL, /* params */
                                     jobz, econ, m, n,
                                     traits<data_type>::cuda_data_type,           /* dataTypeA */
                                     d_A, lda, traits<data_type>::cuda_data_type, /* dataTypeS */
                                     d_S, traits<data_type>::cuda_data_type,      /* dataTypeU */
                                     d_U, lda,                                    /* ldu */
                                     traits<data_type>::cuda_data_type,           /* dataTypeV */
                                     d_V, lda,                                    /* ldv */
                                     traits<data_type>::cuda_data_type,           /* computeType */
                                     d_work, d_lwork, h_work, h_lwork, d_info, &h_err_sigma));

    CUDA_CHECK(cudaMemcpyAsync(U.data(), d_U, sizeof(data_type) * U.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(V.data(), d_V, sizeof(data_type) * V.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(S.data(), d_S, sizeof(data_type) * S.size(), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xgesvdp: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    std::printf("=====\n");

    std::printf("S = (matlab base-1)\n");
    print_matrix(n, 1, S.data(), lda);
    std::printf("=====\n");

    std::printf("U = (matlab base-1)\n");
    print_matrix(m, (econ) ? n : m, U.data(), lda);
    std::printf("=====\n");

    std::printf("V = (matlab base-1)\n");
    print_matrix(n, n, V.data(), lda);
    std::printf("=====\n");

    // step 5: measure error of singular value
    double ds_sup = 0;
    for (int j = 0; j < n; j++) {
        double err = fabs(S[j] - S_exact[j]);
        ds_sup = (ds_sup > err) ? ds_sup : err;
    }
    std::printf("|S - S_exact| = %E \n", ds_sup);

    /* step 6: |A - U*S*V**T| */
    /* W = V*S */
    CUBLAS_CHECK(cublasDdgmm(cublasH, CUBLAS_SIDE_RIGHT, n, n, d_V, lda, d_S, 1, d_W, lda));

    /* A := -U*W**T + A */
    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));

    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, m, /* number of rows of A */
                             n,                                    /* number of columns of A */
                             n,                                    /* number of columns of U  */
                             &h_minus_one, d_U, lda, d_W, lda, &h_one, d_A, lda));

    double dR_fro = 0.0;
    CUBLAS_CHECK(cublasDnrm2(cublasH, A.size(), d_A, 1, &dR_fro));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("|A - U*S*V**T| = %E \n", dR_fro);
    std::printf("h_err_sigma = %E \n", h_err_sigma);
    std::printf("h_err_sigma is 0 if the singular value of A is not close to zero\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(h_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
