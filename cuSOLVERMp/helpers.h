/*
 * Copyright 2023 NVIDIA Corporation.  All rights reserved.
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

#pragma once

#include <stdbool.h>
#include <string.h>
#include <mpi.h>
#include <cal.h>

typedef struct _Options
{
    // problem properties
    int m;
    int n;
    int nrhs;
    int mbA;
    int nbA;
    int mbB;
    int nbB;
    int mbQ;
    int nbQ;
    int mbZ;
    int nbZ;
    int ia;
    int ja;
    int ib;
    int jb;
    int iq;
    int jq;
    int iz;
    int jz;

    // grid
    int  p;
    int  q;
    char grid_layout;

    // others
    bool verbose;
} Options;

void printHelp(const Options* opts)
{
    printf("Available options:\n"
           "    -m\n"
           "    -n\n"
           "    -nrhs\n"
           "    -mbA\n"
           "    -nbA\n"
           "    -mbB\n"
           "    -nbB\n"
           "    -mbQ\n"
           "    -nbQ\n"
           "    -mbZ\n"
           "    -nbZ\n"
           "    -ia\n"
           "    -ja\n"
           "    -ib\n"
           "    -jb\n"
           "    -iq\n"
           "    -jq\n"
           "    -iz\n"
           "    -jz\n"
           "    -p\n"
           "    -q\n"
           "    -grid_layout\n"
           "    -verbose\n");
}

void print(const Options* opts)
{
    printf("Parameters: "
           "m=%d n=%d nrhs=%d "
           "mbA=%d nbA=%d mbB=%d nbB=%d mbQ=%d nbQ=%d mbZ=%d nbZ=%d"
           "ia=%d ja=%d ib=%d jb=%d iq=%d jq=%d iz=%d jz=%d p=%d q=%d grid_layout=%c verbose=%d\n",
           opts->m,
           opts->n,
           opts->nrhs,
           opts->mbA,
           opts->nbA,
           opts->mbB,
           opts->nbB,
           opts->mbQ,
           opts->nbQ,
           opts->mbZ,
           opts->nbZ,
           opts->ia,
           opts->ja,
           opts->ib,
           opts->jb,
           opts->iq,
           opts->jq,
           opts->iz,
           opts->jz,
           opts->p,
           opts->q,
           opts->grid_layout,
           opts->verbose);
}

void parse(Options* opts, int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-m") == 0)
        {
            opts->m = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-n") == 0)
        {
            opts->n = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-nrhs") == 0)
        {
            opts->nrhs = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-mbA") == 0)
        {
            opts->mbA = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-nbA") == 0)
        {
            opts->nbA = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-mbB") == 0)
        {
            opts->mbB = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-nbB") == 0)
        {
            opts->nbB = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-mbQ") == 0)
        {
            opts->mbQ = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-nbQ") == 0)
        {
            opts->nbQ = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-mbZ") == 0)
        {
            opts->mbZ = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-nbZ") == 0)
        {
            opts->nbZ = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-ia") == 0)
        {
            opts->ia = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-ja") == 0)
        {
            opts->ja = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-ib") == 0)
        {
            opts->ib = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-jb") == 0)
        {
            opts->jb = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-iq") == 0)
        {
            opts->iq = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-jq") == 0)
        {
            opts->jq = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-iz") == 0)
        {
            opts->iz = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-jz") == 0)
        {
            opts->jz = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-p") == 0)
        {
            opts->p = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-q") == 0)
        {
            opts->q = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-grid_layout") == 0)
        {
            const char* grid_layout = argv[++i];
            opts->grid_layout       = (grid_layout[0] == 'r' || grid_layout[0] == 'R' ? 'R' : 'C');
        }
        else if (strcmp(argv[i], "-verbose") == 0)
        {
            opts->verbose = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "-help") == 0)
        {
            printHelp(opts);
            exit(0);
        }
        else
        {
            printf("unknown option: %s\n", argv[i]);
            printHelp(opts);
            exit(1);
        }
    }
}

void validate(const Options* opts)
{
    if (opts->ia && opts->mbA && (opts->ia - 1) % opts->mbA != 0)
    {
        fprintf(stderr, "Error: IA must be a multiple of mbA\n");
        exit(1);
    }

    if (opts->ja && opts->nbA && (opts->ja - 1) % opts->nbA != 0)
    {
        fprintf(stderr, "Error: JA must be a multiple of nbA\n");
        exit(1);
    }

    if (opts->ib && opts->mbB && (opts->ib - 1) % opts->mbB != 0)
    {
        fprintf(stderr, "Error: IB must be a multiple of mbB\n");
        exit(1);
    }

    if (opts->jb && opts->nbB && (opts->jb - 1) % opts->nbB != 0)
    {
        fprintf(stderr, "Error: JB must be a multiple of nbB\n");
        exit(1);
    }

    if (opts->iq && opts->mbQ && (opts->iq - 1) % opts->mbQ != 0)
    {
        fprintf(stderr, "Error: IQ must be a multiple of mbQ\n");
        exit(1);
    }

    if (opts->jq && opts->nbQ && (opts->jq - 1) % opts->nbQ != 0)
    {
        fprintf(stderr, "Error: JQ must be a multiple of nbQ\n");
        exit(1);
    }

    if (opts->iz && opts->mbZ && (opts->iz - 1) % opts->mbZ != 0)
    {
        fprintf(stderr, "Error: IZ must be a multiple of mbZ\n");
        exit(1);
    }

    if (opts->jz && opts->nbZ && (opts->jz - 1) % opts->nbZ != 0)
    {
        fprintf(stderr, "Error: JZ must be a multiple of nbZ\n");
        exit(1);
    }
}

static inline int getLocalRank()
{
    int      localRank;
    MPI_Comm localComm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    return localRank;
}

static calError_t allgather(void* src_buf, void* recv_buf, size_t size, void* data, void** request)
{
    MPI_Request req;
    int         err = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size, MPI_BYTE, (MPI_Comm)(data), &req);
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
    int         completed;
    int         err = MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
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
