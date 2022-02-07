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
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"

int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params_gesvdr = NULL;

    using data_type = double;

    /* Input matrix dimensions */
    const int64_t m = 5;
    const int64_t n = 5;
    const int64_t lda = m;
    const int64_t ldu = m;
    const int64_t ldv = n;

    /* rank of matrix A */
    const int64_t min_mn = std::min(m, n);

    /* Compute left/right eigenvectors */
    signed char jobu = 'S';
    signed char jobv = 'S';

    /* Number of iterations */
    const int64_t iters = 2;
    const int64_t rank = std::min(2, *reinterpret_cast<int *>(const_cast<int64_t *>(&n)));
    const int64_t p = std::min(2, static_cast<int>(n - rank));

    std::printf("%lu, %lu\n", rank, p);

    const std::vector<data_type> A = {0.76420743, 0.61411544, 0.81724151, 0.42040879, 0.03446089,
                                      0.03697287, 0.85962444, 0.67584086, 0.45594666, 0.02074835,
                                      0.42018265, 0.39204509, 0.12657948, 0.90250559, 0.23076218,
                                      0.50339844, 0.92974961, 0.21213988, 0.63962457, 0.58124562,
                                      0.58325673, 0.11589871, 0.39831112, 0.21492685, 0.00540355};
    const std::vector<data_type> S_ref{2.36539241, 0.81117785, 0.68562255, 0.41390509, 0.01519322};
    std::vector<data_type> S_gpu(m, 0);

    data_type *d_A = nullptr;
    data_type *d_U = nullptr;
    data_type *d_S = nullptr;
    data_type *d_V = nullptr;
    int *d_info = nullptr;
    int info = 0;

    size_t d_lwork = 0;     /* size of workspace */
    void *d_work = nullptr; /* device workspace for getrf */
    size_t h_lwork = 0;     /* size of workspace */
    void *h_work = nullptr; /* host workspace for getrf */

    std::printf("A = (matlab base-1)\n");
    print_matrix(m, n, A.data(), lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    CUSOLVER_CHECK(cusolverDnCreateParams(&params_gesvdr));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(data_type) * ldu * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(data_type) * ldv * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(data_type) * S_ref.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * lda * n, cudaMemcpyHostToDevice,
                               stream));

    std::printf("m = %ld, n = %ld, rank = %ld, p = %ld, iters = %ld\n", m, n, rank, p, iters);
    if ((rank + p) > n) {
        throw std::runtime_error("Error: (rank + p) > n ");
    }

    /* step 3: query working space of SVD */
    CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
        cusolverH, params_gesvdr, jobu, jobv, m, n, rank, p, iters,
        traits<data_type>::cuda_data_type, d_A, lda, traits<data_type>::cuda_data_type, d_S,
        traits<data_type>::cuda_data_type, d_U, ldu, traits<data_type>::cuda_data_type, d_V, ldv,
        traits<data_type>::cuda_data_type, &d_lwork, &h_lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));

    if (0 < h_lwork) {
        h_work = reinterpret_cast<void *>(malloc(h_lwork));
        if (d_work == nullptr) {
            throw std::runtime_error("Error: d_work not allocated.");
        }
    }

    /* step 4: compute SVD */
    CUSOLVER_CHECK(cusolverDnXgesvdr(
        cusolverH, params_gesvdr, jobu, jobv, m, n, rank, p, iters,
        traits<data_type>::cuda_data_type, d_A, lda, traits<data_type>::cuda_data_type, d_S,
        traits<data_type>::cuda_data_type, d_U, ldu, traits<data_type>::cuda_data_type, d_V, ldv,
        traits<data_type>::cuda_data_type, d_work, d_lwork, h_work, h_lwork, d_info));

    CUDA_CHECK(cudaMemcpyAsync(S_gpu.data(), d_S, sizeof(data_type) * S_gpu.size(),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* check info value */
    std::printf("after Xgesvdr: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    double max_err = 0;
    double max_relerr = 0;
    for (int i = 0; i < rank; i++) {
        const double lambda_ref = S_ref[i];
        const double lambda_gpu = S_gpu[i];
        const double AbsErr = fabs(lambda_ref - lambda_gpu);
        const double RelErr = AbsErr / lambda_ref;

        max_err = std::max(max_err, AbsErr) ? max_err : AbsErr;
        max_relerr = std::max(max_relerr, RelErr) ? max_relerr : RelErr;

        std::printf("S_ref[%d]=%f  S_gpu=[%d]=%f  AbsErr=%E  RelErr=%E\n", i, lambda_ref, i,
                    lambda_gpu, AbsErr, RelErr);
    }
    std::printf("\n");

    double eps = 1.E-8;
    std::printf("max_err = %E, max_relerr = %E, eps = %E\n", max_err, max_relerr, eps);

    if (max_relerr > eps) {
        std::printf("Error: max_relerr is bigger than eps\n");
        std::printf("try to increase oversampling or iters\n");
        std::printf("otherwise, reduce eps\n");
    } else {
        std::printf("Success: max_relerr is smaller than eps\n");
    }

    /* free resources */
    free(h_work);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroyParams(params_gesvdr));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
