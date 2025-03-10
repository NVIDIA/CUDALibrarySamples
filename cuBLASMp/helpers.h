/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include <cal.h>
#include <mpi.h>
#include <stdbool.h>
#include <string.h>

#define MPI_CHECK(call)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        int status = call;                                                                                             \
        if (status != MPI_SUCCESS)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "MPI error at %s:%d : %d\n", __FILE__, __LINE__, status);                                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define NVSHMEM_CHECK(call)                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        int status = call;                                                                                             \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            fprintf(stderr, "NVSHMEM error at %s:%d : %d\n", __FILE__, __LINE__, status);                              \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CAL_CHECK(call)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        calError_t status = call;                                                                                      \
        if (status != CAL_OK)                                                                                          \
        {                                                                                                              \
            fprintf(stderr, "CAL error at %s:%d : %d\n", __FILE__, __LINE__, status);                                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            fprintf(stderr, "CUDA error at %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(status));             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUBLASMP_CHECK(call)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasMpStatus_t status = call;                                                                                \
        if (status != CUBLASMP_STATUS_SUCCESS)                                                                         \
        {                                                                                                              \
            fprintf(stderr, "cuBLASMp error at %s:%d : %d\n", __FILE__, __LINE__, status);                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

struct Options
{
    // problem properties
    int m;
    int n;
    int k;
    int mbA;
    int nbA;
    int mbB;
    int nbB;
    int mbC;
    int nbC;
    int ia;
    int ja;
    int ib;
    int jb;
    int ic;
    int jc;

    // grid
    int p;
    int q;
    char grid_layout;

    // others
    bool verbose;

    void printHelp()
    {
        printf("Available options:\n"
               "    -m\n"
               "    -n\n"
               "    -k\n"
               "    -mbA\n"
               "    -nbA\n"
               "    -mbB\n"
               "    -nbB\n"
               "    -mbC\n"
               "    -nbC\n"
               "    -ia\n"
               "    -ja\n"
               "    -ib\n"
               "    -jb\n"
               "    -ic\n"
               "    -jc\n"
               "    -p\n"
               "    -q\n"
               "    -grid_layout\n"
               "    -verbose\n");
    }

    void print()
    {
        printf(
            "Parameters: "
            "m=%d n=%d k=%d "
            "mbA=%d nbA=%d mbB=%d nbB=%d mbC=%d nbC=%d "
            "ia=%d ja=%d ib=%d jb=%d ic=%d jc=%d p=%d q=%d grid_layout=%c verbose=%d\n",
            m,
            n,
            k,
            mbA,
            nbA,
            mbB,
            nbB,
            mbC,
            nbC,
            ia,
            ja,
            ib,
            jb,
            ic,
            jc,
            p,
            q,
            grid_layout,
            verbose);
    }

    void parse(int argc, char** argv)
    {
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i], "-m") == 0)
            {
                m = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-n") == 0)
            {
                n = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-k") == 0)
            {
                k = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-mbA") == 0)
            {
                mbA = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-nbA") == 0)
            {
                nbA = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-mbB") == 0)
            {
                mbB = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-nbB") == 0)
            {
                nbB = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-mbC") == 0)
            {
                mbC = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-nbC") == 0)
            {
                nbC = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-ia") == 0)
            {
                ia = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-ja") == 0)
            {
                ja = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-ib") == 0)
            {
                ib = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-jb") == 0)
            {
                jb = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-ic") == 0)
            {
                ic = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-jc") == 0)
            {
                jc = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-p") == 0)
            {
                p = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-q") == 0)
            {
                q = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-grid_layout") == 0)
            {
                grid_layout = *argv[++i];
            }
            else if (strcmp(argv[i], "-verbose") == 0)
            {
                verbose = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-help") == 0)
            {
                printHelp();
                exit(0);
            }
            else
            {
                printf("unknown option: %s\n", argv[i]);
                printHelp();
                exit(1);
            }
        }
    }

    void validate()
    {
        if (ia && mbA && (ia - 1) % mbA != 0)
        {
            fprintf(stderr, "Error: ia must be a multiple of mbA\n");
            exit(1);
        }

        if (ja && nbA && (ja - 1) % nbA != 0)
        {
            fprintf(stderr, "Error: ja must be a multiple of nbA\n");
            exit(1);
        }

        if (ib && mbB && (ib - 1) % mbB != 0)
        {
            fprintf(stderr, "Error: ib must be a multiple of mbB\n");
            exit(1);
        }

        if (jb && nbB && (jb - 1) % nbB != 0)
        {
            fprintf(stderr, "Error: jb must be a multiple of nbB\n");
            exit(1);
        }

        if (ic && mbC && (ic - 1) % mbC != 0)
        {
            fprintf(stderr, "Error: ic must be a multiple of mbC\n");
            exit(1);
        }

        if (jc && nbC && (jc - 1) % nbC != 0)
        {
            fprintf(stderr, "Error: jc must be a multiple of nbC\n");
            exit(1);
        }
    }
};

static inline int getLocalDevice()
{
    int localRank;
    MPI_Comm localComm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    return localRank % deviceCount;
}

static calError_t allgather(void* src_buf, void* recv_buf, size_t size, void* data, void** request)
{
    MPI_Request req;
    int err = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size, MPI_BYTE, (MPI_Comm)(data), &req);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    *request = (void*)(req);
    return CAL_OK;
}

static calError_t request_test(void* request)
{
    MPI_Request req = (MPI_Request)(request);
    int completed;
    int err = MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

static calError_t request_free(void* request)
{
    return CAL_OK;
}