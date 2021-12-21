/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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
#include <cusparse.h>

#define CHECK_CUDA(func)                                                                           \
    {                                                                                              \
        cudaError_t status = (func);                                                               \
        if (status != cudaSuccess) {                                                               \
            std::printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,              \
                        cudaGetErrorString(status), status);                                       \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    }

#define CHECK_CUSPARSE(func)                                                                       \
    {                                                                                              \
        cusparseStatus_t status = (func);                                                          \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                   \
            std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__,          \
                        cusparseGetErrorString(status), status);                                   \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    }

#define CHECK_CUBLAS(func)                                                                         \
    {                                                                                              \
        cublasStatus_t status = (func);                                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            std::printf("CUBLAS API failed at line %d with error: (%d)\n", __LINE__, status);      \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    }

/*
 * compute | b - A*x|_inf
 */
template <typename T>
void residaul_eval(int n, const T *ds, const T *dl, const T *d, const T *du, const T *dw,
                   const T *b, const T *x, T *r_nrminf_ptr) {
    T r_nrminf = 0;
    for (int i = 0; i < n; i++) {
        T dot = 0;
        if (i > 1) {
            dot += ds[i] * x[i - 2];
        }
        if (i > 0) {
            dot += dl[i] * x[i - 1];
        }
        dot += d[i] * x[i];
        if (i < (n - 1)) {
            dot += du[i] * x[i + 1];
        }
        if (i < (n - 2)) {
            dot += dw[i] * x[i + 2];
        }
        T ri = b[i] - dot;
        r_nrminf = (r_nrminf > fabs(ri)) ? r_nrminf : fabs(ri);
    }

    *r_nrminf_ptr = r_nrminf;
}

int main(void) {
    cusparseHandle_t cusparseH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int n = 4;
    const int batchSize = 2;

    /*
     *      |  1    8   13   0  |       | 1 |       | -0.0592 |
     *  A1 =|  5    2    9  14  |, b1 = | 2 |, x1 = |  0.3428 |
     *      | 11    6    3  10  |       | 3 |       | -0.1295 |
     *      |  0   12    7   4  |       | 4 |       |  0.1982 |
     *
     *      | 15   22   27   0  |       | 5 |       | -0.0012 |
     *  A2 =| 19   16   23  28  |, b2 = | 6 |, x2 = |  0.2792 |
     *      | 25   20   17  24  |       | 7 |       | -0.0416 |
     *      |  0   26   21  18  |       | 8 |       |  0.0898 |
     */

    /*
     * A = (ds, dl, d, du, dw), B and X are in aggregate format
     */
    const std::vector<float> ds = {0, 0, 11, 12, 0, 0, 25, 26};
    const std::vector<float> dl = {0, 5, 6, 7, 0, 19, 20, 21};
    const std::vector<float> d = {1, 2, 3, 4, 15, 16, 17, 18};
    const std::vector<float> du = {8, 9, 10, 0, 22, 23, 24, 0};
    const std::vector<float> dw = {13, 14, 0, 0, 27, 28, 0, 0};
    const std::vector<float> B = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> X(n * batchSize, 0); /* Xj = Aj \ Bj */

    /* device memory
     * (d_ds0, d_dl0, d_d0, d_du0, d_dw0) is aggregate format
     * (d_ds, d_dl, d_d, d_du, d_dw) is interleaved format
     */
    float *d_ds0 = nullptr;
    float *d_dl0 = nullptr;
    float *d_d0 = nullptr;
    float *d_du0 = nullptr;
    float *d_dw0 = nullptr;
    float *d_ds = nullptr;
    float *d_dl = nullptr;
    float *d_d = nullptr;
    float *d_du = nullptr;
    float *d_dw = nullptr;
    float *d_B = nullptr;
    float *d_X = nullptr;

    size_t lworkInBytes = 0;
    char *d_work = nullptr;

    const float h_one = 1;
    const float h_zero = 0;

    int algo = 0; /* QR factorization */

    /* step 1: create cusolver handle, bind a stream */
    CHECK_CUSPARSE(cusparseCreate(&cusparseH));
    CHECK_CUBLAS(cublasCreate(&cublasH));

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUSPARSE(cusparseSetStream(cusparseH, stream));
    CHECK_CUBLAS(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    /* step 2: allocate device memory */
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_ds0), sizeof(float) * ds.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_dl0), sizeof(float) * dl.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_d0), sizeof(float) * d.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_du0), sizeof(float) * du.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_dw0), sizeof(float) * dw.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_ds), sizeof(float) * ds.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_dl), sizeof(float) * dl.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_d), sizeof(float) * d.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_du), sizeof(float) * du.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_dw), sizeof(float) * dw.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * B.size()));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(float) * X.size()));

    /* step 3: prepare data in device, interleaved format */
    CHECK_CUDA(cudaMemcpyAsync(d_ds0, ds.data(), sizeof(float) * ds.size(), cudaMemcpyHostToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(d_dl0, dl.data(), sizeof(float) * dl.size(), cudaMemcpyHostToDevice,
                               stream));
    CHECK_CUDA(
        cudaMemcpyAsync(d_d0, d.data(), sizeof(float) * d.size(), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_du0, du.data(), sizeof(float) * du.size(), cudaMemcpyHostToDevice,
                               stream));
    CHECK_CUDA(cudaMemcpyAsync(d_dw0, dw.data(), sizeof(float) * dw.size(), cudaMemcpyHostToDevice,
                               stream));
    CHECK_CUDA(
        cudaMemcpyAsync(d_B, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice, stream));

    /* convert ds to interleaved format ds = transpose(ds0) */
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, /* transa */
                             CUBLAS_OP_T,          /* transb, don't care */
                             batchSize,            /* number of rows of ds */
                             n,                    /* number of columns of ds */
                             &h_one, d_ds0,        /* ds0 is n-by-batchSize */
                             n,                    /* leading dimension of ds0 */
                             &h_zero, nullptr, n,  /* don't care */
                             d_ds,                 /* ds is batchSize-by-n */
                             batchSize));          /* leading dimension of ds */

    /* convert dl to interleaved format  dl = transpose(dl0)
     */
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, /* transa */
                             CUBLAS_OP_T,          /* transb, don't care */
                             batchSize,            /* number of rows of dl */
                             n,                    /* number of columns of dl */
                             &h_one, d_dl0,        /* dl0 is n-by-batchSize */
                             n,                    /* leading dimension of dl0 */
                             &h_zero, nullptr, n,  /* don't care */
                             d_dl,                 /* dl is batchSize-by-n */
                             batchSize             /* leading dimension of dl */
                             ));

    /* convert d to interleaved format d = transpose(d0)
     */
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, /* transa */
                             CUBLAS_OP_T,          /* transb, don't care */
                             batchSize,            /* number of rows of d */
                             n,                    /* number of columns of d */
                             &h_one, d_d0,         /* d0 is n-by-batchSize */
                             n,                    /* leading dimension of d0 */
                             &h_zero, nullptr, n,  /* don't care */
                             d_d,                  /* d is batchSize-by-n */
                             batchSize             /* leading dimension of d */
                             ));

    /* convert du to interleaved format du = transpose(du0)
     */
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, /* transa */
                             CUBLAS_OP_T,          /* transb, don't care */
                             batchSize,            /* number of rows of du */
                             n,                    /* number of columns of du */
                             &h_one, d_du0,        /* du0 is n-by-batchSize */
                             n,                    /* leading dimension of du0 */
                             &h_zero, nullptr, n,  /* don't care */
                             d_du,                 /* du is batchSize-by-n */
                             batchSize             /* leading dimension of du */
                             ));

    /* convert dw to interleaved format dw = transpose(dw0)
     */
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, /* transa */
                             CUBLAS_OP_T,          /* transb, don't care */
                             batchSize,            /* number of rows of dw */
                             n,                    /* number of columns of dw */
                             &h_one, d_dw0,        /* dw0 is n-by-batchSize */
                             n,                    /* leading dimension of dw0 */
                             &h_zero, nullptr, n,  /* don't care */
                             d_dw,                 /* dw is batchSize-by-n */
                             batchSize             /* leading dimension of dw */
                             ));

    /* convert B to interleaved format X = transpose(B)
     */
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, /* transa */
                             CUBLAS_OP_T,          /* transb, don't care */
                             batchSize,            /* number of rows of X */
                             n,                    /* number of columns of X */
                             &h_one, d_B,          /* B is n-by-batchSize */
                             n,                    /* leading dimension of B */
                             &h_zero, nullptr, n,  /* don't care */
                             d_X,                  /* X is batchSize-by-n */
                             batchSize             /* leading dimension of X */
                             ));

    /* step 4: prepare workspace */
    CHECK_CUSPARSE(cusparseSgpsvInterleavedBatch_bufferSizeExt(
        cusparseH, algo, n, d_ds, d_dl, d_d, d_du, d_dw, d_X, batchSize, &lworkInBytes));

    std::printf("lworkInBytes = %lld \n", static_cast<long long>(lworkInBytes));

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_work), lworkInBytes));

    /* step 5: solve Aj*xj = bj */
    CHECK_CUSPARSE(cusparseSgpsvInterleavedBatch(cusparseH, algo, n, d_ds, d_dl, d_d, d_du, d_dw,
                                                 d_X, batchSize, d_work));

    /* step 6: convert X back to aggregate format  */
    /* B = transpose(X) */
    CHECK_CUBLAS(cublasSgeam(cublasH, CUBLAS_OP_T, /* transa */
                             CUBLAS_OP_T,          /* transb, don't care */
                             n,                    /* number of rows of B */
                             batchSize,            /* number of columns of B */
                             &h_one, d_X,          /* X is batchSize-by-n */
                             batchSize,            /* leading dimension of X */
                             &h_zero, nullptr, n,  /* don't cae */
                             d_B,                  /* B is n-by-batchSize */
                             n                     /* leading dimension of B */
                             ));

    /* step 7: residual evaluation */
    CHECK_CUDA(
        cudaMemcpyAsync(X.data(), d_B, sizeof(float) * X.size(), cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    /* step 8: Check results */
    printf("==== x1 = inv(A1)*b1 \n");
    for (int j = 0; j < n; j++) {
        printf("x1[%d] = %f\n", j, X[j]);
    }

    float r1_nrminf;
    residaul_eval(n, ds.data(), dl.data(), d.data(), du.data(), dw.data(), B.data(), X.data(),
                  &r1_nrminf);
    printf("|b1 - A1*x1| = %E\n", r1_nrminf);

    printf("\n==== x2 = inv(A2)*b2 \n");
    for (int j = 0; j < n; j++) {
        printf("x2[%d] = %f\n", j, X[n + j]);
    }

    float r2_nrminf;
    residaul_eval(n, ds.data() + n, dl.data() + n, d.data() + n, du.data() + n, dw.data() + n,
                  B.data() + n, X.data() + n, &r2_nrminf);
    printf("|b2 - A2*x2| = %E\n", r2_nrminf);

    /* free resources */
    CHECK_CUDA(cudaFree(d_ds0));
    CHECK_CUDA(cudaFree(d_dl0));
    CHECK_CUDA(cudaFree(d_d0));
    CHECK_CUDA(cudaFree(d_du0));
    CHECK_CUDA(cudaFree(d_dw0));
    CHECK_CUDA(cudaFree(d_ds));
    CHECK_CUDA(cudaFree(d_dl));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFree(d_du));
    CHECK_CUDA(cudaFree(d_dw));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_X));

    CHECK_CUSPARSE(cusparseDestroy(cusparseH));
    CHECK_CUBLAS(cublasDestroy(cublasH));

    CHECK_CUDA(cudaStreamDestroy(stream));

    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
