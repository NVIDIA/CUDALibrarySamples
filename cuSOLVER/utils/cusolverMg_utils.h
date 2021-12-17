/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include "cusolver_utils.h"

#ifndef IDX2F
#define IDX2F(i, j, lda) ((((j)-1) * (static_cast<size_t>(lda))) + ((i)-1))
#endif /* IDX2F */

#ifndef IDX1F
#define IDX1F(i) ((i)-1)
#endif /* IDX1F */

/*
 * nbGpus : (int) number of gpus in deviceList array.
 * deviceList : (*int) list of device ids.
 *
 * The function restores the input device before leaving.
 */
static void enablePeerAccess(const int nbGpus, const int *deviceList) {
    int currentDevice = 0;
    CUDA_CHECK(cudaGetDevice(&currentDevice));

    /* Remark: access granted by this cudaDeviceEnablePeerAccess is unidirectional */
    /* Rows and columns represents a connectivity matrix between GPUs in the system */
    for (int row = 0; row < nbGpus; row++) {
        CUDA_CHECK(cudaSetDevice(row));
        for (int col = 0; col < nbGpus; col++) {
            if (row != col) {
                int canAccessPeer = 0;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, row, col));
                if (canAccessPeer) {
                    std::printf("\t Enable peer access from gpu %d to gpu %d\n", row, col);
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(col, 0));
                }
            }
        }
    }
    CUDA_CHECK(cudaSetDevice(currentDevice));
}

static void workspaceFree(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                          void **array_d_work                    /* <t> num_devices, host array */
                          /* array_d_work[j] points to device workspace of device j */
) {
    int currentDev = 0; /* record current device ID */
    CUDA_CHECK(cudaGetDevice(&currentDev));

    for (int idx = 0; idx < num_devices; idx++) {
        int deviceId = deviceIdA[idx];
        /* WARNING: we need to set device before any runtime API */
        CUDA_CHECK(cudaSetDevice(deviceId));

        if (NULL != array_d_work[idx]) {
            CUDA_CHECK(cudaFree(array_d_work[idx]));
        }
    }
    CUDA_CHECK(cudaSetDevice(currentDev));
}

static void workspaceAlloc(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                           size_t sizeInBytes,                    /* number of bytes per device */
                           void **array_d_work                    /* <t> num_devices, host array */
                           /* array_d_work[j] points to device workspace of device j */
) {
    int currentDev = 0; /* record current device ID */
    CUDA_CHECK(cudaGetDevice(&currentDev));

    for (int idx = 0; idx < num_devices; idx++) {
        int deviceId = deviceIdA[idx];
        /* WARNING: we need to set device before any runtime API */
        CUDA_CHECK(cudaSetDevice(deviceId));

        void *d_workspace = NULL;

        CUDA_CHECK(cudaMalloc(&d_workspace, sizeInBytes));
        array_d_work[idx] = d_workspace;
    }
    CUDA_CHECK(cudaSetDevice(currentDev));
}

/* create a empty matrix A with A := 0 */
template <typename T_ELEM>
void createMat(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
               int N_A,                               /* number of columns of global A */
               int T_A,                               /* number of columns per column tile */
               int LLD_A,                             /* leading dimension of local A */
               T_ELEM **array_d_A /* host pointer array of dimension num_devices */
                                  // std::vector<T_ELEM*>array_d_A
) {
    int currentDev = 0; /* record current device id */
    CUDA_CHECK(cudaGetDevice(&currentDev));
    CUDA_CHECK(cudaDeviceSynchronize());
    const int A_num_blks = (N_A + T_A - 1) / T_A;
    const int max_A_num_blks_per_device = (A_num_blks + num_devices - 1) / num_devices;
    /* Allocate base pointers */
    for (int p = 0; p < num_devices; p++) {
        CUDA_CHECK(cudaSetDevice(deviceIdA[p]));
        /* Allocate max_A_num_blks_per_device blocks per device */
        CUDA_CHECK(
            cudaMalloc(&(array_d_A[p]), sizeof(T_ELEM) * LLD_A * T_A * max_A_num_blks_per_device));
        /* A := 0 */
        CUDA_CHECK(
            cudaMemset(array_d_A[p], 0, sizeof(T_ELEM) * LLD_A * T_A * max_A_num_blks_per_device));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(currentDev));
}

static void destroyMat(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                       int N_A,                               /* number of columns of global A */
                       int T_A,          /* number of columns per column tile */
                       void **array_d_A) /* host pointer array of dimension num_devices */
{
    int currentDev = 0; /* record current device id */
    CUDA_CHECK(cudaGetDevice(&currentDev));

    for (int p = 0; p < num_devices; p++) {
        CUDA_CHECK(cudaSetDevice(deviceIdA[p]));

        if (NULL != array_d_A[p]) {
            CUDA_CHECK(cudaFree(array_d_A[p]));
        }
    }
    CUDA_CHECK(cudaSetDevice(currentDev));
}

template <typename T_ELEM>
static void
mat_pack2unpack(int num_devices, int N_A,   /* number of columns of global A */
                int T_A,                    /* number of columns per column tile */
                int LLD_A,                  /* leading dimension of local A */
                T_ELEM **array_d_A_packed,  /* host pointer array of dimension num_devices */
                                            /* output */
                T_ELEM **array_d_A_unpacked /* host pointer array of dimension num_blks */
) {
    const int num_blks = (N_A + T_A - 1) / T_A;

    for (int p_a = 0; p_a < num_devices; p_a++) {
        T_ELEM *d_A = array_d_A_packed[p_a];
        int nz_blks = 0;
        for (int JA_blk_id = p_a; JA_blk_id < num_blks; JA_blk_id += num_devices) {
            array_d_A_unpacked[JA_blk_id] = d_A + static_cast<size_t>(LLD_A) * T_A * nz_blks;
            nz_blks++;
        }
    }
}

/*
 *  A(IA:IA+M-1, JA:JA+N-1) := B(1:M, 1:N)
 */
template <typename T_ELEM>
static void memcpyH2D(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                      int M,                                 /* number of rows in local A, B */
                      int N,                                 /* number of columns in local A, B */
                                                             /* input */
                      const T_ELEM *h_B, /* host array, h_B is M-by-N with leading dimension ldb  */
                      int ldb,
                      /* output */
                      int N_A,                   /* number of columns of global A */
                      int T_A,                   /* number of columns per column tile */
                      int LLD_A,                 /* leading dimension of local A */
                      T_ELEM **array_d_A_packed, /* host pointer array of dimension num_devices */
                      int IA,                    /* base-1 */
                      int JA                     /* base-1 */
) {
    int currentDev = 0; /* record current device id */

    /*  Quick return if possible */
    if ((0 >= M) || (0 >= N)) {
        return;
    }

    /* consistent checking */
    if (ldb < M) {
        throw std::runtime_error("Consistency Error.");
    }

    CUDA_CHECK(cudaGetDevice(&currentDev));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int num_blks = (N_A + T_A - 1) / T_A;

    std::vector<T_ELEM *> array_d_A_unpacked(num_blks);

    mat_pack2unpack<T_ELEM>(num_devices, N_A,         /* number of columns of global A */
                            T_A,                      /* number of columns per column tile */
                            LLD_A,                    /* leading dimension of local A */
                            array_d_A_packed,         /* host pointer array of size num_devices */
                                                      /* output */
                            array_d_A_unpacked.data() /* host pointer arrya of size num_blks */
    );

    /* region of interest is A(IA:IA+N-1, JA:JA+N-1) */
    const int N_hat = (JA - 1) + N; /* JA is base-1 */

    const int JA_start_blk_id = (JA - 1) / T_A;
    const int JA_end_blk_id = (N_hat - 1) / T_A;

    for (int p_a = 0; p_a < num_devices; p_a++) {
        /* region of interest: JA_start_blk_id:1:JA_end_blk_id */
        for (int JA_blk_id = p_a; JA_blk_id <= JA_end_blk_id; JA_blk_id += num_devices) {
            if (JA_blk_id < JA_start_blk_id) {
                continue;
            }
            /*
             * process column block of A
             *       A(A_start_row:M_A, A_start_col : (A_start_col + IT_A-1) )
             */
            const int IBX_A = (1 + JA_blk_id * T_A);     /* base-1 */
            const int A_start_col = std::max(JA, IBX_A); /* base-1 */
            const int A_start_row = IA;                  /* base-1 */

            const int bdd = std::min(N_hat, (IBX_A + T_A - 1));
            const int IT_A = std::min(T_A, (bdd - A_start_col + 1));

            const int loc_A_start_row = A_start_row;               /* base-1 */
            const int loc_A_start_col = (A_start_col - IBX_A) + 1; /* base-1 */

            T_ELEM *d_A =
                array_d_A_unpacked[JA_blk_id] + IDX2F(loc_A_start_row, loc_A_start_col, LLD_A);
            const T_ELEM *h_A = h_B + IDX2F(A_start_row - IA + 1, A_start_col - JA + 1, ldb);

            CUDA_CHECK(cudaMemcpy2D(d_A,                                              /* dst */
                                    static_cast<size_t>(LLD_A) * sizeof(T_ELEM), h_A, /* src */
                                    static_cast<size_t>(ldb) * sizeof(T_ELEM),
                                    static_cast<size_t>(M) * sizeof(T_ELEM),
                                    static_cast<size_t>(IT_A), cudaMemcpyHostToDevice));
        } /* for each tile per device */
    }     /* for each device */
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(currentDev));
}

/*
 *  B(1:M, 1:N) := A(IA:IA+M-1, JA:JA+N-1)
 */
template <typename T_ELEM>
static void memcpyD2H(int num_devices, const int *deviceIdA, /* <int> dimension num_devices */
                      int M,                                 /* number of rows in local A, B */
                      int N,                                 /* number of columns in local A, B */
                                                             /* input */
                      int N_A,                               /* number of columns of global A */
                      int T_A,                               /* number of columns per column tile */
                      int LLD_A,                             /* leading dimension of local A */
                      T_ELEM **array_d_A_packed, /* host pointer array of dimension num_devices */
                      int IA,                    /* base-1 */
                      int JA,                    /* base-1 */
                                                 /* output */
                      T_ELEM *h_B, /* host array, h_B is M-by-N with leading dimension ldb  */
                      int ldb) {
    int currentDev = 0; /* record current device id */

    /*  Quick return if possible */
    if ((0 >= M) || (0 >= N)) {
        return;
    }

    /* consistent checking */
    if (ldb < M) {
        throw std::runtime_error("Consistency Error.");
    }

    CUDA_CHECK(cudaGetDevice(&currentDev));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int num_blks = (N_A + T_A - 1) / T_A;
    std::vector<T_ELEM *> array_d_A_unpacked(num_blks);

    mat_pack2unpack<T_ELEM>(num_devices, N_A,         /* number of columns of global A */
                            T_A,                      /* number of columns per column tile */
                            LLD_A,                    /* leading dimension of local A */
                            array_d_A_packed,         /* host pointer array of size num_devices */
                            array_d_A_unpacked.data() /* host pointer arrya of size num_blks */
    );
    /* region of interest is A(IA:IA+N-1, JA:JA+N-1) */
    const int N_hat = (JA - 1) + N; /* JA is base-1 */
    const int JA_start_blk_id = (JA - 1) / T_A;
    const int JA_end_blk_id = (N_hat - 1) / T_A;
    for (int p_a = 0; p_a < num_devices; p_a++) {
        /* region of interest: JA_start_blk_id:1:JA_end_blk_id */
        for (int JA_blk_id = p_a; JA_blk_id <= JA_end_blk_id; JA_blk_id += num_devices) {
            if (JA_blk_id < JA_start_blk_id) {
                continue;
            }
            /* process column block, A(A_start_row:M_A, A_start_col : (A_start_col + IT_A-1) ) */
            const int IBX_A = (1 + JA_blk_id * T_A);     /* base-1 */
            const int A_start_col = std::max(JA, IBX_A); /* base-1 */
            const int A_start_row = IA;                  /* base-1 */
            const int bdd = std::min(N_hat, (IBX_A + T_A - 1));
            const int IT_A = std::min(T_A, (bdd - A_start_col + 1));
            const int loc_A_start_row = A_start_row;               /* base-1 */
            const int loc_A_start_col = (A_start_col - IBX_A) + 1; /* base-1 */
            const T_ELEM *d_A =
                array_d_A_unpacked[JA_blk_id] + IDX2F(loc_A_start_row, loc_A_start_col, LLD_A);
            T_ELEM *h_A = h_B + IDX2F(A_start_row - IA + 1, A_start_col - JA + 1, ldb);
            CUDA_CHECK(cudaMemcpy2D(h_A,                                            /* dst */
                                    static_cast<size_t>(ldb) * sizeof(T_ELEM), d_A, /* src */
                                    static_cast<size_t>(LLD_A) * sizeof(T_ELEM),
                                    static_cast<size_t>(M) * sizeof(T_ELEM),
                                    static_cast<size_t>(IT_A), cudaMemcpyDeviceToHost));
        } /* for each tile per device */
    }     /* for each device */
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaSetDevice(currentDev));
}
