/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#error                                                                                   \
    "This sample does not support Windows right now (replace dlopen()/dlsym() with equivalent Windows API functions)"
#endif

#include <dlfcn.h> // required for dlopen and dlsym; only works on Linux

#include "cudss_distributed_interface.h"
#include <library_types.h>

#define CUDSS_DISTRIBUTED_INTERFACE_MIN_MAJOR_VERSION 0
#define CUDSS_DISTRIBUTED_INTERFACE_MIN_MINOR_VERSION 8
#define CUDSS_DISTRIBUTED_INTERFACE_MIN_PATCH_VERSION 0

#ifndef USE_MPI
#error "This example needs to be compiled with MPI (and optionally with NCCL as well)"
#endif

#include "mpi.h"
#ifdef USE_NCCL
#include "nccl.h"
#endif
#if !defined(USE_OPENMPI) && !defined(USE_NCCL)
#error "Either USE_OPENMPI or USE_NCCL must be defined"
#endif
#if defined(USE_OPENMPI) && defined(USE_NCCL)
#error "Exactly one of USE_OPENMPI and USE_NCCL must be defined"
#endif


void fill_host_data(int *data_h, int number_elements, int seed) {
    for (int i = 0; i < number_elements; ++i) {
        data_h[i] = i + seed;
    }
}

/* Returns 1 if data matches fill_host_data(seed), 0 on first mismatch. */
int validate_host_data(const int *data_h, int number_elements, int seed) {
    for (int i = 0; i < number_elements; ++i) {
        if (data_h[i] != i + seed) {
            return 0;
        }
    }
    return 1;
}

static void print_usage(int rank, const char *prog) {
    if (rank != 0)
        return;
    printf("Usage: mpirun -np <N> %s <backend> <comm_layer.so>\n", prog);
    printf("  backend        \"openmpi\" or \"nccl\"\n");
    printf("  comm_layer.so  Full path to communication layer library\n");
    printf("  For openmpi backend, OpenMPI must be built with CUDA-aware support.\n");
    printf("Example: mpirun -np 2 %s openmpi /path/to/libcudss_commlayer_openmpi.so\n",
           prog);
    fflush(stdout);
}

typedef struct {
    void        *commIfaceLib;
    cudaStream_t stream;
    MPI_Comm    *mpi_comm;
#ifdef USE_NCCL
    ncclComm_t *nccl_comm;
#endif
    int *data_h;
} cleanup_ctx_t;

/* Releases all acquired resources and calls MPI_Finalize.
 * Safe to call with a zero-initialized struct: all fields are NULL-checked before use. */
static void do_finalize_and_cleanup(cleanup_ctx_t *c) {
    if (c->commIfaceLib != NULL)
        dlclose(c->commIfaceLib);
    if (c->stream != NULL)
        cudaStreamDestroy(c->stream);
    if (c->mpi_comm != NULL)
        free(c->mpi_comm);
#ifdef USE_NCCL
    if (c->nccl_comm != NULL) {
        ncclCommDestroy(*c->nccl_comm);
        free(c->nccl_comm);
    }
#endif
    if (c->data_h != NULL)
        free(c->data_h);
    MPI_Finalize();
}

typedef struct {
    int          rank, size;
    MPI_Comm    *mpi_comm;
    MPI_Comm     host_mpi_comm; /* MPI_COMM_WORLD copy for host-side APIs with NCCL */
    void        *comm;
    char         comm_backend_name[1024];
    int         *data_h;
    int          number_elements;
    int          seed;
    cudaStream_t stream;
#ifdef USE_NCCL
    ncclComm_t *nccl_comm;
#endif
    cudssDistributedInterface_t *commIface;
    void                        *commIfaceLib;
} test_context_t;

/* For the NCCL backend, the comm layer still requires an MPI_Comm for host APIs. */
static inline void *host_comm_world(test_context_t *ctx) {
    if (strcmp(ctx->comm_backend_name, "nccl") == 0)
        return (void *)&ctx->host_mpi_comm;
    return ctx->comm;
}

/* Fatal infrastructure errors (CUDA/MPI/NCCL): print from the detecting rank and
 * abort all ranks immediately via MPI_Abort, avoiding hangs on partial failures. */
#define CUDA_CALL_AND_CHECK(call, msg)                                                   \
    do {                                                                                 \
        cudaError_t _e = (call);                                                         \
        if (_e != cudaSuccess) {                                                         \
            printf("Example FAILED: CUDA error = %d, %s\n", _e, (msg));                  \
            fflush(stdout);                                                              \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                     \
        }                                                                                \
    } while (0)
#define MPI_CALL_AND_CHECK(call, msg)                                                    \
    do {                                                                                 \
        int _e = (call);                                                                 \
        if (_e != 0) {                                                                   \
            printf("Example FAILED: MPI error = %d, %s\n", _e, (msg));                   \
            fflush(stdout);                                                              \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                     \
        }                                                                                \
    } while (0)
#ifdef USE_NCCL
#define NCCL_CALL_AND_CHECK(call, msg)                                                   \
    do {                                                                                 \
        ncclResult_t _r = (call);                                                        \
        if (_r != ncclSuccess) {                                                         \
            printf("Example FAILED: NCCL error = %d, %s\n", _r, (msg));                  \
            fflush(stdout);                                                              \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                     \
        }                                                                                \
    } while (0)
#endif

/* Reduces local_err across all ranks via MPI_Allreduce and prints the result from
 * rank 0. Returns 1 if all ranks passed, 0 if any rank reported an error. */
static int report_result(test_context_t *ctx, int local_err, const char *test_name) {
    int any_err = 0;
    MPI_CALL_AND_CHECK(
        MPI_Allreduce(&local_err, &any_err, 1, MPI_INT, MPI_SUM, *ctx->mpi_comm),
        "MPI_Allreduce in report_result failed");
    if (ctx->rank == 0)
        printf("Test [%s] %s\n", test_name, any_err != 0 ? "FAILED" : "PASSED");
    fflush(stdout);
    return (any_err == 0) ? 1 : 0;
}

static int test_init_env_comm(test_context_t *ctx) {
    /* Reaching this point means MPI, CUDA, and the comm backend all initialized
     * successfully; any failure there would have aborted the run already. */
    if (ctx->rank == 0)
        printf("Test [initialize environment and communicator] PASSED\n");
    fflush(stdout);
    return 1;
}

static int test_bcast_raw(test_context_t *ctx) {
    int  test_err = 0;
    int *data_d   = NULL;
    CUDA_CALL_AND_CHECK(cudaMalloc((void **)&data_d, ctx->number_elements * sizeof(int)),
                        "cudaMalloc(data_d) failed");
    if (ctx->rank == 0) {
        fill_host_data(ctx->data_h, ctx->number_elements, ctx->seed);
        CUDA_CALL_AND_CHECK(cudaMemcpy(data_d, ctx->data_h,
                                       ctx->number_elements * sizeof(int),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy HostToDevice (data_h -> data_d) failed");
    }
    memset(ctx->data_h, 0, ctx->number_elements * sizeof(int));

#ifdef USE_OPENMPI
    /* This test uses MPI_Bcast with a device pointer; requires CUDA-aware OpenMPI. */
    MPI_CALL_AND_CHECK(MPI_Bcast(data_d, ctx->number_elements * sizeof(int), MPI_BYTE, 0,
                                 (*ctx->mpi_comm)),
                       "MPI_Bcast(data_d) failed");
    CUDA_CALL_AND_CHECK(cudaMemcpy(ctx->data_h, data_d,
                                   ctx->number_elements * sizeof(int),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy DeviceToHost (data_d -> data_h) after Bcast failed");
    if (!validate_host_data(ctx->data_h, ctx->number_elements, ctx->seed))
        test_err = 1;
    memset(ctx->data_h, 0, ctx->number_elements * sizeof(int));
#endif
#ifdef USE_NCCL
    NCCL_CALL_AND_CHECK(ncclBcast(data_d, ctx->number_elements * sizeof(int), ncclChar, 0,
                                  (*ctx->nccl_comm), ctx->stream),
                        "ncclBcast(data_d) failed");
    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(ctx->stream),
                        "cudaStreamSynchronize after ncclBcast failed");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(ctx->data_h, data_d, ctx->number_elements * sizeof(int),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy DeviceToHost (data_d -> data_h) after ncclBcast failed");
    if (!validate_host_data(ctx->data_h, ctx->number_elements, ctx->seed))
        test_err = 1;
    memset(ctx->data_h, 0, ctx->number_elements * sizeof(int));
#endif

    CUDA_CALL_AND_CHECK(cudaFree(data_d), "cudaFree(data_d) failed");
    return report_result(ctx, test_err, "BcastRaw (direct MPI/NCCL, no comm layer)");
}

static int test_bcast_device(test_context_t *ctx) {
    /* Increment seed to distinguish this test's data from the previous Bcast test. */
    ctx->seed++;
    int  test_err = 0;
    int *data_d   = NULL;
    CUDA_CALL_AND_CHECK(cudaMalloc((void **)&data_d, ctx->number_elements * sizeof(int)),
                        "cudaMalloc(data_d) failed");
    if (ctx->rank == 0) {
        fill_host_data(ctx->data_h, ctx->number_elements, ctx->seed);
        CUDA_CALL_AND_CHECK(cudaMemcpy(data_d, ctx->data_h,
                                       ctx->number_elements * sizeof(int),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy HostToDevice (data_h -> data_d) failed");
    }
    memset(ctx->data_h, 0, ctx->number_elements * sizeof(int));

    if (ctx->comm != NULL) {
        if (ctx->commIface->cudssBcastDevice(data_d, ctx->number_elements, CUDSS_R_32I, 0,
                                             ctx->comm, ctx->stream) != 0)
            test_err = 1;
        CUDA_CALL_AND_CHECK(cudaStreamSynchronize(ctx->stream),
                            "cudaStreamSynchronize after BcastDevice failed");
        CUDA_CALL_AND_CHECK(
            cudaMemcpy(ctx->data_h, data_d, ctx->number_elements * sizeof(int),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy DeviceToHost (data_d -> data_h) after BcastDevice failed");
        if (!validate_host_data(ctx->data_h, ctx->number_elements, ctx->seed))
            test_err = 1;
        memset(ctx->data_h, 0, ctx->number_elements * sizeof(int));
    }

    CUDA_CALL_AND_CHECK(cudaFree(data_d), "cudaFree(data_d) failed");
    return report_result(ctx, test_err, "BcastDevice via comm layer");
}

static int test_comm_rank_size(test_context_t *ctx) {
    int rank_from_dev = -1, size_from_dev = -1;
    int rank_from_host = -1, size_from_host = -1;
    int test_err = 0;
    if (ctx->comm != NULL) {
        if (ctx->commIface->cudssCommRankDevice(ctx->comm, &rank_from_dev) != 0)
            test_err = 1;
        if (ctx->commIface->cudssCommSizeDevice(ctx->comm, &size_from_dev) != 0)
            test_err = 1;
        if (ctx->commIface->cudssCommRankHost(host_comm_world(ctx), &rank_from_host) != 0)
            test_err = 1;
        if (ctx->commIface->cudssCommSizeHost(host_comm_world(ctx), &size_from_host) != 0)
            test_err = 1;
        if (rank_from_dev != ctx->rank || size_from_dev != ctx->size ||
            rank_from_host != ctx->rank || size_from_host != ctx->size)
            test_err = 1;
    }
    return report_result(ctx, test_err, "CommRank/CommSize via comm layer");
}

static int test_bcast_host(test_context_t *ctx) {
    const int small_count       = 256;
    const int bcast_host_offset = 100; /* value at root: host_buf[i] = i + offset */
    int      *host_buf          = (int *)malloc(small_count * sizeof(int));
    int       test_err          = 0;
    if (host_buf == NULL) {
        test_err = 1;
    } else {
        if (ctx->rank == 0) {
            for (int i = 0; i < small_count; ++i)
                host_buf[i] = i + bcast_host_offset;
        } else {
            memset(host_buf, 0, small_count * sizeof(int));
        }
        if (ctx->comm != NULL &&
            ctx->commIface->cudssBcastHost(host_buf, small_count, CUDSS_R_32I, 0,
                                           host_comm_world(ctx), ctx->stream) != 0)
            test_err = 1;
        for (int i = 0; i < small_count && test_err == 0; ++i) {
            if (host_buf[i] != i + bcast_host_offset)
                test_err = 1;
        }
        free(host_buf);
    }
    return report_result(ctx, test_err, "BcastHost via comm layer");
}

static int test_reduce_allreduce(test_context_t *ctx) {
    const int reduce_count = 4;
    int       send_vals[4], recv_vals[4];
    double    send_f64[4], recv_f64[4];
    int       i;
    for (i = 0; i < reduce_count; ++i) {
        send_vals[i] = ctx->rank + i;
        send_f64[i]  = (double)(ctx->rank + i);
    }
    int test_err = 0;
    /* --- ReduceHost / AllreduceHost --- */
    if (ctx->comm != NULL) {
        memcpy(recv_vals, send_vals, sizeof(send_vals));
        if (ctx->commIface->cudssReduceHost(send_vals, recv_vals, reduce_count,
                                            CUDSS_R_32I, CUDSS_SUM, 0,
                                            host_comm_world(ctx), ctx->stream) != 0)
            test_err = 1;
        if (ctx->rank == 0) {
            for (i = 0; i < reduce_count; ++i) {
                int expected = 0;
                for (int r = 0; r < ctx->size; ++r)
                    expected += r + i;
                if (recv_vals[i] != expected)
                    test_err = 1;
            }
        }
        memcpy(recv_f64, send_f64, sizeof(send_f64));
        if (ctx->commIface->cudssAllreduceHost(send_f64, recv_f64, reduce_count,
                                               CUDSS_R_64F, CUDSS_MAX,
                                               host_comm_world(ctx), ctx->stream) != 0)
            test_err = 1;
        for (i = 0; i < reduce_count; ++i) {
            double expected = (double)(ctx->size - 1 + i);
            if (recv_f64[i] != expected)
                test_err = 1;
        }
    }
    if (report_result(ctx, test_err, "Reduce/Allreduce Host via comm layer") != 1)
        return 0;

    /* --- ReduceDevice --- */
    int *send_d = NULL, *recv_d = NULL;
    CUDA_CALL_AND_CHECK(cudaMalloc(&send_d, reduce_count * sizeof(int)),
                        "cudaMalloc(send_d) for Reduce failed");
    CUDA_CALL_AND_CHECK(cudaMalloc(&recv_d, reduce_count * sizeof(int)),
                        "cudaMalloc(recv_d) for Reduce failed");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(send_d, send_vals, reduce_count * sizeof(int), cudaMemcpyHostToDevice),
        "cudaMemcpy HostToDevice (send_vals -> send_d) failed");
    test_err = 0;
    if (ctx->comm != NULL &&
        ctx->commIface->cudssReduceDevice(send_d, recv_d, reduce_count, CUDSS_R_32I,
                                          CUDSS_MIN, 0, ctx->comm, ctx->stream) != 0)
        test_err = 1;
    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(ctx->stream),
                        "cudaStreamSynchronize after Reduce failed");
    /* Each rank sends [rank+0, rank+1, rank+2, rank+3]. CUDSS_MIN to root => root gets
     * [0,1,2,3]. */
    if (ctx->rank == 0) {
        CUDA_CALL_AND_CHECK(cudaMemcpy(recv_vals, recv_d, reduce_count * sizeof(int),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy DeviceToHost (recv_d -> recv_vals) failed");
        for (i = 0; i < reduce_count; ++i) {
            if (recv_vals[i] != i)
                test_err = 1;
        }
    }
    CUDA_CALL_AND_CHECK(cudaFree(send_d), "cudaFree(send_d) failed");
    CUDA_CALL_AND_CHECK(cudaFree(recv_d), "cudaFree(recv_d) failed");
    if (report_result(ctx, test_err, "ReduceDevice via comm layer") != 1)
        return 0;

    /* --- AllreduceDevice --- */
    int *allreduce_send = NULL, *allreduce_recv = NULL;
    CUDA_CALL_AND_CHECK(cudaMalloc(&allreduce_send, reduce_count * sizeof(int)),
                        "cudaMalloc(allreduce_send) failed");
    CUDA_CALL_AND_CHECK(cudaMalloc(&allreduce_recv, reduce_count * sizeof(int)),
                        "cudaMalloc(allreduce_recv) failed");
    CUDA_CALL_AND_CHECK(cudaMemcpy(allreduce_send, send_vals, reduce_count * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy HostToDevice (send_vals -> allreduce_send) failed");
    test_err = 0;
    if (ctx->comm != NULL && ctx->commIface->cudssAllreduceDevice(
                                 allreduce_send, allreduce_recv, reduce_count,
                                 CUDSS_R_32I, CUDSS_MIN, ctx->comm, ctx->stream) != 0)
        test_err = 1;
    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(ctx->stream),
                        "cudaStreamSynchronize after Allreduce failed");
    CUDA_CALL_AND_CHECK(cudaMemcpy(recv_vals, allreduce_recv, reduce_count * sizeof(int),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy DeviceToHost (allreduce_recv -> recv_vals) failed");
    /* Allreduce CUDSS_MIN => all ranks get [0,1,2,3]. */
    for (i = 0; i < reduce_count; ++i) {
        if (recv_vals[i] != i)
            test_err = 1;
    }
    CUDA_CALL_AND_CHECK(cudaFree(allreduce_send), "cudaFree(allreduce_send) failed");
    CUDA_CALL_AND_CHECK(cudaFree(allreduce_recv), "cudaFree(allreduce_recv) failed");
    return report_result(ctx, test_err, "AllreduceDevice via comm layer");
}

/* Send/Recv: rank 0 sends to rank 1; other ranks only participate in the final reduce. */
static int test_send_recv(test_context_t *ctx) {
    const int sendrecv_count = 8;
    const int tag_host = 42, tag_device = 43;
    const int send_host_base = 1000, send_device_base = 2000;
    int       test_err = 0;
    /* --- Send/Recv Host --- */
    if (ctx->comm == NULL) {
        test_err = 1;
    } else if (ctx->rank == 0) {
        int send_buf[8];
        for (int i = 0; i < sendrecv_count; ++i)
            send_buf[i] = send_host_base + i;
        if (ctx->commIface->cudssSendHost(send_buf, sendrecv_count, CUDSS_R_32I, 1,
                                          tag_host, host_comm_world(ctx),
                                          ctx->stream) != 0)
            test_err = 1;
    } else if (ctx->rank == 1) {
        int recv_buf[8];
        memset(recv_buf, 0, sizeof(recv_buf));
        if (ctx->commIface->cudssRecvHost(recv_buf, sendrecv_count, CUDSS_R_32I, 0,
                                          tag_host, host_comm_world(ctx),
                                          ctx->stream) != 0)
            test_err = 1;
        for (int i = 0; i < sendrecv_count; ++i) {
            if (recv_buf[i] != send_host_base + i)
                test_err = 1;
        }
    }
    if (report_result(ctx, test_err, "Send/Recv Host via comm layer") != 1)
        return 0;

    /* --- Send/Recv Device --- */
    int *send_d_buf = NULL, *recv_d_buf = NULL;
    int  send_h[8], recv_h[8];
    test_err = 0;
    CUDA_CALL_AND_CHECK(cudaMalloc(&send_d_buf, sendrecv_count * sizeof(int)),
                        "cudaMalloc(send_d_buf) for Send/Recv failed");
    CUDA_CALL_AND_CHECK(cudaMalloc(&recv_d_buf, sendrecv_count * sizeof(int)),
                        "cudaMalloc(recv_d_buf) for Send/Recv failed");
    if (ctx->comm == NULL) {
        test_err = 1;
    } else if (ctx->rank == 0) {
        int i;
        for (i = 0; i < sendrecv_count; ++i)
            send_h[i] = send_device_base + i;
        CUDA_CALL_AND_CHECK(cudaMemcpy(send_d_buf, send_h, sendrecv_count * sizeof(int),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy HostToDevice (send_h -> send_d_buf) failed");
        if (ctx->commIface->cudssSendDevice(send_d_buf, sendrecv_count, CUDSS_R_32I, 1,
                                            tag_device, ctx->comm, ctx->stream) != 0)
            test_err = 1;
        CUDA_CALL_AND_CHECK(cudaStreamSynchronize(ctx->stream),
                            "cudaStreamSynchronize after cudssSendDevice failed");
    } else if (ctx->rank == 1) {
        if (ctx->commIface->cudssRecvDevice(recv_d_buf, sendrecv_count, CUDSS_R_32I, 0,
                                            tag_device, ctx->comm, ctx->stream) != 0)
            test_err = 1;
        CUDA_CALL_AND_CHECK(cudaStreamSynchronize(ctx->stream),
                            "cudaStreamSynchronize after Send/Recv failed");
        CUDA_CALL_AND_CHECK(cudaMemcpy(recv_h, recv_d_buf, sendrecv_count * sizeof(int),
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy DeviceToHost (recv_d_buf -> recv_h) failed");
        for (int i = 0; i < sendrecv_count; ++i) {
            if (recv_h[i] != send_device_base + i)
                test_err = 1;
        }
    }
    CUDA_CALL_AND_CHECK(cudaFree(send_d_buf), "cudaFree(send_d_buf) failed");
    CUDA_CALL_AND_CHECK(cudaFree(recv_d_buf), "cudaFree(recv_d_buf) failed");
    return report_result(ctx, test_err, "Send/Recv Device via comm layer");
}

static int test_scatterv(test_context_t *ctx) {
    const int recv_count_per_rank = 4;
    const int scatterv_total      = ctx->size * recv_count_per_rank;
    const int scatterv_host_base = 2000, scatterv_device_base = 3000;
    int      *sendcounts = (int *)malloc((size_t)ctx->size * sizeof(int));
    int      *displs     = (int *)malloc((size_t)ctx->size * sizeof(int));

    int *recvbuf_h = (int *)malloc(recv_count_per_rank * sizeof(int));
    int *sendbuf_h = NULL;
    int  test_err  = (sendcounts == NULL || displs == NULL || recvbuf_h == NULL) ? 1 : 0;
    if (test_err) {
        if (ctx->rank == 0)
            printf("ScattervHost: allocation failed\n");
        free(sendcounts);
        free(displs);
        free(recvbuf_h);
        return report_result(ctx, 1, "ScattervHost via comm layer");
    }
    for (int r = 0; r < ctx->size; ++r) {
        sendcounts[r] = recv_count_per_rank;
        displs[r]     = r * recv_count_per_rank;
    }
    /* --- ScattervHost --- */
    if (ctx->rank == 0) {
        sendbuf_h = (int *)malloc(scatterv_total * sizeof(int));
        if (sendbuf_h) {
            for (int i = 0; i < scatterv_total; ++i)
                sendbuf_h[i] = scatterv_host_base + i;
        } else {
            test_err = 1;
        }
    }
    if (test_err == 0 && ctx->comm != NULL &&
        ctx->commIface->cudssScattervHost(sendbuf_h, sendcounts, displs, CUDSS_R_32I,
                                          recvbuf_h, recv_count_per_rank, CUDSS_R_32I, 0,
                                          host_comm_world(ctx), ctx->stream) != 0)
        test_err = 1;
    if (ctx->rank == 0 && sendbuf_h)
        free(sendbuf_h);
    for (int i = 0; i < recv_count_per_rank && test_err == 0; ++i) {
        if (recvbuf_h[i] != scatterv_host_base + ctx->rank * recv_count_per_rank + i)
            test_err = 1;
    }
    free(sendcounts);
    free(displs);
    free(recvbuf_h);
    if (report_result(ctx, test_err, "ScattervHost via comm layer") != 1)
        return 0;

    /* --- ScattervDevice --- */
    int *sendcounts_h = (int *)malloc((size_t)ctx->size * sizeof(int));
    int *displs_h     = (int *)malloc((size_t)ctx->size * sizeof(int));
    int *sendbuf_d    = NULL;
    int *recvbuf_d    = NULL;
    int *recv_h       = (int *)malloc(recv_count_per_rank * sizeof(int));
    if (sendcounts_h == NULL || displs_h == NULL || recv_h == NULL) {
        if (ctx->rank == 0)
            printf("ScattervDevice: allocation failed\n");
        free(sendcounts_h);
        free(displs_h);
        free(recv_h);
        return report_result(ctx, 1, "ScattervDevice via comm layer");
    }
    for (int r = 0; r < ctx->size; ++r) {
        sendcounts_h[r] = recv_count_per_rank;
        displs_h[r]     = r * recv_count_per_rank;
    }
    CUDA_CALL_AND_CHECK(cudaMalloc(&recvbuf_d, recv_count_per_rank * sizeof(int)),
                        "cudaMalloc(recvbuf_d) for Scatterv failed");
    test_err = 0;
    if (ctx->rank == 0) {
        CUDA_CALL_AND_CHECK(cudaMalloc(&sendbuf_d, scatterv_total * sizeof(int)),
                            "cudaMalloc(sendbuf_d) for Scatterv failed");
        int *send_h = (int *)malloc(scatterv_total * sizeof(int));
        if (send_h) {
            for (int i = 0; i < scatterv_total; ++i)
                send_h[i] = scatterv_device_base + i;
            CUDA_CALL_AND_CHECK(
                cudaMemcpy(sendbuf_d, send_h, scatterv_total * sizeof(int),
                           cudaMemcpyHostToDevice),
                "cudaMemcpy HostToDevice (send_h -> sendbuf_d) for Scatterv failed");
            free(send_h);
        } else {
            /* sendbuf_d remains uninitialized; collective still runs to avoid deadlock,
             * but test_err marks this run as failed. */
            test_err = 1;
        }
    }
    /* Non-root send buffer is ignored by the collective; pass recvbuf_d as a
     * non-NULL placeholder since the API does not guarantee NULL is accepted. */
    if (ctx->comm != NULL &&
        ctx->commIface->cudssScattervDevice(
            ctx->rank == 0 ? sendbuf_d : recvbuf_d, sendcounts_h, displs_h, CUDSS_R_32I,
            recvbuf_d, recv_count_per_rank, CUDSS_R_32I, 0, ctx->comm, ctx->stream) != 0)
        test_err = 1;
    CUDA_CALL_AND_CHECK(cudaStreamSynchronize(ctx->stream),
                        "cudaStreamSynchronize after Scatterv failed");
    CUDA_CALL_AND_CHECK(cudaMemcpy(recv_h, recvbuf_d, recv_count_per_rank * sizeof(int),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy DeviceToHost (recvbuf_d -> recv_h) failed");
    for (int i = 0; i < recv_count_per_rank && test_err == 0; ++i) {
        if (recv_h[i] != scatterv_device_base + ctx->rank * recv_count_per_rank + i)
            test_err = 1;
    }
    if (ctx->rank == 0 && sendbuf_d != NULL)
        CUDA_CALL_AND_CHECK(cudaFree(sendbuf_d), "cudaFree(sendbuf_d) failed");
    CUDA_CALL_AND_CHECK(cudaFree(recvbuf_d), "cudaFree(recvbuf_d) failed");
    free(sendcounts_h);
    free(displs_h);
    free(recv_h);
    return report_result(ctx, test_err, "ScattervDevice via comm layer");
}

static int test_comm_split_free(test_context_t *ctx) {
    /* Split output storage matches the comm layer implementation:
     * OpenMPI layer writes MPI_Comm; NCCL layer writes ncclComm_t (see
     * cudssCommSplitDevice). */
#ifdef USE_OPENMPI
    MPI_Comm newcomm_dev = MPI_COMM_NULL;
#endif
#ifdef USE_NCCL
    ncclComm_t newcomm_dev = NULL;
#endif
    MPI_Comm newcomm_host = MPI_COMM_NULL;
    int      color        = ctx->rank % 2;
    int      key          = ctx->rank;
    int      test_err     = 0;
    if (ctx->comm != NULL) {
        if (ctx->commIface->cudssCommSplitDevice(ctx->comm, color, key,
                                                 (void *)&newcomm_dev) != 0)
            test_err = 1;
        if (ctx->commIface->cudssCommSplitHost(host_comm_world(ctx), color, key,
                                               (void *)&newcomm_host) != 0)
            test_err = 1;
    }
    /* color = rank%2, key = rank: new_rank = rank/2,
     * new_size = (color==0) ? (size+1)/2 : size/2 */
    int expected_rank = ctx->rank / 2;
    int expected_size = (color == 0) ? (ctx->size + 1) / 2 : ctx->size / 2;

    /* Rank/Size/Free expect &handle (same convention as ctx->comm). */
#if defined(USE_OPENMPI)
    if (newcomm_dev != MPI_COMM_NULL) {
#elif defined(USE_NCCL)
    if (newcomm_dev != NULL) {
#endif
        int rdev = -1, sdev = -1;
        if (ctx->commIface->cudssCommRankDevice((void *)&newcomm_dev, &rdev) != 0)
            test_err = 1;
        if (ctx->commIface->cudssCommSizeDevice((void *)&newcomm_dev, &sdev) != 0)
            test_err = 1;
        if (rdev != expected_rank || sdev != expected_size)
            test_err = 1;
        if (ctx->commIface->cudssCommFreeDevice((void *)&newcomm_dev) != 0)
            test_err = 1;
    }
    if (newcomm_host != MPI_COMM_NULL) {
        int rhost = -1, shost = -1;
        if (ctx->commIface->cudssCommRankHost((void *)&newcomm_host, &rhost) != 0)
            test_err = 1;
        if (ctx->commIface->cudssCommSizeHost((void *)&newcomm_host, &shost) != 0)
            test_err = 1;
        if (rhost != expected_rank || shost != expected_size)
            test_err = 1;
        if (ctx->commIface->cudssCommFreeHost((void *)&newcomm_host) != 0)
            test_err = 1;
    }
    return report_result(ctx, test_err, "CommSplit/CommFree via comm layer");
}

int main(int argc, char *argv[]) {
    /* Initializing the communication backend
       Note: cuDSS can work with any CUDA-aware communication backend through the
       user-defined communication layers (see the documentation), but this example
       demonstrates the mode using pre-built communication layers for OpenMPI and
       NCCL.
       Therefore, as both of them rely on MPI_Init/MPI_Finalize, we call it here.
       For a different communication backend, a different initialization/cleanup
       APIs might be needed.
     */

    int mpi_error = MPI_Init(&argc, &argv);
    if (mpi_error != 0) {
        printf("Example FAILED: MPI_Init failed (error = %d)\n", mpi_error);
        return EXIT_FAILURE;
    }

    /* Initialize cleanup context immediately after MPI_Init so that
     * do_finalize_and_cleanup can be called safely from any subsequent failure path. */
    cleanup_ctx_t cleanup;
    memset(&cleanup, 0, sizeof(cleanup));

    int rank = 0, size = 1;
    mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (mpi_error != 0) {
        printf("Example FAILED: MPI_Comm_rank failed (error = %d)\n", mpi_error);
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (mpi_error != 0) {
        if (rank == 0)
            printf("Example FAILED: MPI_Comm_size failed (error = %d)\n", mpi_error);
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    if (rank == 0) {
        printf("---------------------------------------------------------\n");
        printf("This sample will be run with %d processes\n", size);
        printf("---------------------------------------------------------\n");
        fflush(stdout);
    }
    if (size < 2) {
        if (rank == 0) {
            printf("Error: This sample requires at least 2 processes (got %d).\n", size);
            fflush(stdout);
        }
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }

    /* Binding each process to a specific GPU device under the assumption
       that the number of processes does not exceed the number of devices */
    int         device_count = 0;
    cudaError_t cuda_error   = cudaGetDeviceCount(&device_count);
    if (cuda_error != cudaSuccess) {
        if (rank == 0)
            printf("Example FAILED: CUDA error = %d, cudaGetDeviceCount failed\n",
                   cuda_error);
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    if (device_count == 0) {
        if (rank == 0)
            printf("Error: no GPU devices have been found\n");
        fflush(stdout);
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    int device_id = rank % device_count;
    cuda_error    = cudaSetDevice(device_id);
    if (cuda_error != cudaSuccess) {
        if (rank == 0)
            printf("Example FAILED: CUDA error = %d, cudaSetDevice failed\n", cuda_error);
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }

    char comm_backend_name[1024];
    char comm_layer_libname[1024];
    if (argc < 3) {
        print_usage(rank, argc > 0 ? argv[0] : "test_communication_layer");
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage(rank, argv[0]);
        do_finalize_and_cleanup(&cleanup);
        return 0;
    }
    snprintf(comm_backend_name, sizeof(comm_backend_name), "%s", argv[1]);
    snprintf(comm_layer_libname, sizeof(comm_layer_libname), "%s", argv[2]);

    /* Validate that the backend name matches the compile-time backend selection. */
#ifdef USE_OPENMPI
    if (strcmp(comm_backend_name, "openmpi") != 0) {
        if (rank == 0) {
            printf("Error: this binary was compiled for the OpenMPI backend; "
                   "expected \"openmpi\", got \"%s\"\n",
                   comm_backend_name);
            print_usage(rank, argv[0]);
        }
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
#endif
#ifdef USE_NCCL
    if (strcmp(comm_backend_name, "nccl") != 0) {
        if (rank == 0) {
            printf("Error: this binary was compiled for the NCCL backend; "
                   "expected \"nccl\", got \"%s\"\n",
                   comm_backend_name);
            print_usage(rank, argv[0]);
        }
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
#endif

    if (rank == 0) {
        printf("Communication backend: %s\n", comm_backend_name);
        printf("Communication layer library: %s\n", comm_layer_libname);
        fflush(stdout);
    }

    const int number_elements = 256 * 1024 * 1024;
    int      *data_h          = NULL;
    data_h                    = (int *)malloc(number_elements * sizeof(int));
    if (data_h == NULL) {
        if (rank == 0)
            printf("Failed to allocate host memory. exiting...\nTest FAILED\n");
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    cleanup.data_h = data_h;

    cudaStream_t stream = NULL;
    cuda_error          = cudaStreamCreate(&stream);
    if (cuda_error != cudaSuccess) {
        if (rank == 0)
            printf("Example FAILED: CUDA error = %d, cudaStreamCreate failed\n",
                   cuda_error);
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    cleanup.stream = stream;

    MPI_Comm *mpi_comm = NULL;
    mpi_comm           = (MPI_Comm *)malloc(sizeof(MPI_Comm));
    if (mpi_comm == NULL) {
        if (rank == 0)
            printf("Failed to allocate MPI_Comm. exiting...\nTest FAILED\n");
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    mpi_comm[0]      = MPI_COMM_WORLD;
    cleanup.mpi_comm = mpi_comm;

#ifdef USE_NCCL
    ncclComm_t *nccl_comm = NULL;
    if (strcmp(comm_backend_name, "nccl") == 0) {
        nccl_comm = (ncclComm_t *)malloc(sizeof(ncclComm_t));
        if (nccl_comm == NULL) {
            printf("Failed to allocate ncclComm_t. exiting...\nTest FAILED\n");
            do_finalize_and_cleanup(&cleanup);
            return EXIT_FAILURE;
        }
        ncclUniqueId id = {};
        if (rank == 0) {
            ncclResult_t nccl_error = ncclGetUniqueId(&id);
            if (nccl_error != ncclSuccess) {
                printf("Example FAILED: NCCL error = %d, ncclGetUniqueId failed\n",
                       nccl_error);
                free(nccl_comm);
                do_finalize_and_cleanup(&cleanup);
                return EXIT_FAILURE;
            }
        }
        mpi_error = MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (mpi_error != 0) {
            if (rank == 0)
                printf("Example FAILED: MPI error = %d, MPI_Bcast(ncclUniqueId) failed\n",
                       mpi_error);
            free(nccl_comm);
            do_finalize_and_cleanup(&cleanup);
            return EXIT_FAILURE;
        }
        ncclResult_t nccl_error = ncclCommInitRank(nccl_comm, size, id, rank);
        if (nccl_error != ncclSuccess) {
            if (rank == 0)
                printf("Example FAILED: NCCL error = %d, ncclCommInitRank failed\n",
                       nccl_error);
            free(nccl_comm);
            do_finalize_and_cleanup(&cleanup);
            return EXIT_FAILURE;
        }
        cleanup.nccl_comm = nccl_comm;
    }
#endif

    void *comm = NULL;
#ifdef USE_OPENMPI
    if (strcmp(comm_backend_name, "openmpi") == 0)
        comm = (void *)mpi_comm;
#endif
#ifdef USE_NCCL
    if (strcmp(comm_backend_name, "nccl") == 0)
        comm = (void *)nccl_comm;
#endif

    test_context_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.rank          = rank;
    ctx.size          = size;
    ctx.mpi_comm      = mpi_comm;
    ctx.host_mpi_comm = MPI_COMM_WORLD;
    ctx.comm          = comm;
    snprintf(ctx.comm_backend_name, sizeof(ctx.comm_backend_name), "%s",
             comm_backend_name);
    ctx.data_h          = data_h;
    ctx.number_elements = number_elements;
    ctx.seed            = 1;
    ctx.stream          = stream;
#ifdef USE_NCCL
    ctx.nccl_comm = nccl_comm;
#endif

    ctx.commIfaceLib = (void *)dlopen(comm_layer_libname, RTLD_NOW);
    if (ctx.commIfaceLib == NULL) {
        if (rank == 0) {
            char *err = dlerror();
            printf("Error: failed to open the communication layer library %s: %s\n",
                   comm_layer_libname, err ? err : "unknown error");
            fflush(stdout);
        }
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    dlerror();
    ctx.commIface = (cudssDistributedInterface_t *)dlsym(ctx.commIfaceLib,
                                                         "cudssDistributedInterface");
    if (ctx.commIface == NULL) {
        if (rank == 0) {
            char *err = dlerror();
            printf("Error: failed to find symbol cudssDistributedInterface in %s: %s\n",
                   comm_layer_libname, err ? err : "unknown error");
            fflush(stdout);
        }
        dlclose(ctx.commIfaceLib);
        ctx.commIfaceLib = NULL;
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    }
    int dist_major = 0, dist_minor = 0, dist_patch = 0;
    if (ctx.commIface->cudssDistributedGetProperty == NULL) {
        if (rank == 0) {
            printf("rank %d: Error: cudssDistributedGetProperty is NULL\n", rank);
            fflush(stdout);
        }
        dlclose(ctx.commIfaceLib);
        ctx.commIfaceLib = NULL;
        do_finalize_and_cleanup(&cleanup);
        return EXIT_FAILURE;
    } else {
        int prop_err = ctx.commIface->cudssDistributedGetProperty(MAJOR_VERSION, &dist_major);
        if (prop_err != 0) {
            if (rank == 0) {
                printf("rank %d: Error: cudssDistributedGetProperty(MAJOR_VERSION) returned %d\n",
                    rank, prop_err);
                fflush(stdout);
            }
            dlclose(ctx.commIfaceLib);
            ctx.commIfaceLib = NULL;
            do_finalize_and_cleanup(&cleanup);
            return EXIT_FAILURE;
        } else
        if ((prop_err = ctx.commIface->cudssDistributedGetProperty(MINOR_VERSION, &dist_minor)) != 0) {
            if (rank == 0) {
                printf("rank %d: Error: cudssDistributedGetProperty(MINOR_VERSION) returned %d\n",
                    rank, prop_err);
                fflush(stdout);
            }
            dlclose(ctx.commIfaceLib);
            ctx.commIfaceLib = NULL;
            do_finalize_and_cleanup(&cleanup);
            return EXIT_FAILURE;
        } else
        if ((prop_err = ctx.commIface->cudssDistributedGetProperty(PATCH_LEVEL, &dist_patch)) != 0) {
            if (rank == 0) {
                printf("rank %d: Error: cudssDistributedGetProperty(PATCH_LEVEL) returned %d\n", rank,
                    prop_err);
                fflush(stdout);
            }
            dlclose(ctx.commIfaceLib);
            ctx.commIfaceLib = NULL;
            do_finalize_and_cleanup(&cleanup);
            return EXIT_FAILURE;
        } else {
            const int dist_version_ok =
                (dist_major > CUDSS_DISTRIBUTED_INTERFACE_MIN_MAJOR_VERSION) ||
                (dist_major == CUDSS_DISTRIBUTED_INTERFACE_MIN_MAJOR_VERSION &&
                 dist_minor > CUDSS_DISTRIBUTED_INTERFACE_MIN_MINOR_VERSION) ||
                (dist_major == CUDSS_DISTRIBUTED_INTERFACE_MIN_MAJOR_VERSION &&
                 dist_minor == CUDSS_DISTRIBUTED_INTERFACE_MIN_MINOR_VERSION &&
                 dist_patch >= CUDSS_DISTRIBUTED_INTERFACE_MIN_PATCH_VERSION);
            if (!dist_version_ok) {
                if (rank == 0) {
                    printf("rank %d: Error: communication layer version %d.%d.%d is below minimum required "
                        "%d.%d.%d\n",
                        rank, dist_major, dist_minor, dist_patch,
                        CUDSS_DISTRIBUTED_INTERFACE_MIN_MAJOR_VERSION,
                        CUDSS_DISTRIBUTED_INTERFACE_MIN_MINOR_VERSION,
                        CUDSS_DISTRIBUTED_INTERFACE_MIN_PATCH_VERSION);
                    fflush(stdout);
                }
                dlclose(ctx.commIfaceLib);
                ctx.commIfaceLib = NULL;
                do_finalize_and_cleanup(&cleanup);
                return EXIT_FAILURE;
            }
        }
    }

    cleanup.commIfaceLib = ctx.commIfaceLib;

    int any_failed = 0;
    if (test_init_env_comm(&ctx) == 0)
        any_failed = 1;
    if (test_bcast_raw(&ctx) == 0)
        any_failed = 1;
    if (test_bcast_device(&ctx) == 0)
        any_failed = 1;
    if (test_comm_rank_size(&ctx) == 0)
        any_failed = 1;
    if (test_bcast_host(&ctx) == 0)
        any_failed = 1;
    if (test_reduce_allreduce(&ctx) == 0)
        any_failed = 1;
    if (test_send_recv(&ctx) == 0)
        any_failed = 1;
    if (test_scatterv(&ctx) == 0)
        any_failed = 1;
    if (test_comm_split_free(&ctx) == 0)
        any_failed = 1;

    do_finalize_and_cleanup(&cleanup);
    if (!any_failed) {
        if (rank == 0)
            printf("Example PASSED\n");
        return EXIT_SUCCESS;
    } else {
        if (rank == 0)
            printf("Example FAILED\n");
        return EXIT_FAILURE;
    }
}
