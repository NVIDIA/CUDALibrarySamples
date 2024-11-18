/*
 * Copyright 2024 NVIDIA Corporation.  All rights reserved.
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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"


int main(int argc, char *argv[]) {
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;

    using real_type = double;
    using cmplx_type = cuDoubleComplex;

    const int64_t n = 3;
    const int64_t lda = 3; // lda >= n

    /*
     *  A = | 1.0 | 2.0 | -3.0 |
     *      | 7.0 | 4.0 | -2.0 |
     *      | 4.0 | 2.0 |  1.0 |
     */
    const std::vector<real_type> A = {1.0, 7.0, 4.0, 2.0, 4.0, 2.0, -3.0, -2.0, 1.0};

    // The eigenvalues are promoted to complex.
    std::vector<cmplx_type> W(n);

    /* Compute only right eigenvectors.*/
    cusolverEigMode_t jobvl = CUSOLVER_EIG_MODE_NOVECTOR;
    cusolverEigMode_t jobvr = CUSOLVER_EIG_MODE_VECTOR;
    const int64_t ldvl = 1; // ldvl >= 1 if jobvl = CUSOLVER_EIG_MODE_NOVECTOR
    const int64_t ldvr = 3; // ldvr >= n if jobvr = CUSOLVER_EIG_MODE_VECTOR
    std::vector<real_type> VR(ldvr * n);

    real_type *d_A = nullptr;
    cmplx_type *d_W = nullptr;
    real_type *d_VR = nullptr;
    real_type *d_VL = nullptr;

    int *d_info = nullptr;
    int info = 0;
    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    void *d_work = nullptr;
    void *h_work = nullptr;

    std::printf("A = (matlab base -1) \n");
    print_matrix(n, n, A.data(), lda);
    std::printf("=====\n");

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(real_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(cmplx_type) * W.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VR), sizeof(real_type) * VR.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(real_type) * A.size(),
                               cudaMemcpyHostToDevice, stream));

    /* step 3: query size of workspace and allocate host and device buffers*/
    CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
        cusolverH,
        params,
        jobvl,
        jobvr,
        n,
        traits<real_type>::cuda_data_type,
        d_A,
        lda,
        traits<cmplx_type>::cuda_data_type,
        d_W,
        traits<real_type>::cuda_data_type,
        d_VL,
        ldvl,
        traits<real_type>::cuda_data_type,
        d_VR,
        ldvr,
        traits<real_type>::cuda_data_type,
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
        traits<real_type>::cuda_data_type,
        d_A,
        lda,
        traits<cmplx_type>::cuda_data_type,
        d_W,
        traits<real_type>::cuda_data_type,
        d_VL,
        ldvl,
        traits<real_type>::cuda_data_type,
        d_VR,
        ldvr,
        traits<real_type>::cuda_data_type,
        d_work,
        workspaceInBytesOnDevice,
        h_work,
        workspaceInBytesOnHost,
        d_info));

    CUDA_CHECK(cudaMemcpyAsync(VR.data(), d_VR, sizeof(real_type) * VR.size(),
               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(W.data(), d_W, sizeof(cmplx_type) * W.size(),
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
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_VR));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFreeHost(h_work));
    CUDA_CHECK(cudaFree(d_work));
    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}