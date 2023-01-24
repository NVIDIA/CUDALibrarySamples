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

#include <mpi.h>
#include <cal.h>

struct Options
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
    int ia;
    int ja;
    int ib;
    int jb;
    int iq;
    int jq;

    // grid
    int p;
    int q;

    // others
    bool verbose;

    void printHelp() const
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
            "    -ia\n"
            "    -ja\n"
            "    -ib\n"
            "    -jb\n"
            "    -iq\n"
            "    -jq\n"
            "    -p\n"
            "    -q\n"
        );
    }

    void print() const
    {
        printf("Parameters: m=%d n=%d nrhs=%d mbA=%d nbA=%d mbB=%d nbB=%d mbQ=%d nbQ=%d ia=%d ja=%d ib=%d jb=%d iq=%d jq=%d p=%d q=%d\n",
            m,
            n,
            nrhs,
            mbA,
            nbA,
            mbB,
            nbB,
            mbQ,
            nbQ,
            ia,
            ja,
            ib,
            jb,
            iq,
            jq,
            p,
            q);
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
            else if (strcmp(argv[i], "-nrhs") == 0)
            {
                nrhs = atoi(argv[++i]);
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
            else if (strcmp(argv[i], "-mbQ") == 0)
            {
                mbQ = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-nbQ") == 0)
            {
                nbQ = atoi(argv[++i]);
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
            else if (strcmp(argv[i], "-iq") == 0)
            {
                iq = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-jq") == 0)
            {
                jq = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-p") == 0)
            {
                p = atoi(argv[++i]);
            }
            else if (strcmp(argv[i], "-q") == 0)
            {
                q = atoi(argv[++i]);
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
            fprintf(stderr, "Error: IA must be a multiple of mbA\n");
            exit(1);
        }

        if (ja && nbA && (ja - 1) % nbA != 0)
        {
            fprintf(stderr, "Error: JA must be a multiple of nbA\n");
            exit(1);
        }

        if (ib && mbB && (ib - 1) % mbB != 0)
        {
            fprintf(stderr, "Error: IB must be a multiple of mbB\n");
            exit(1);
        }

        if (jb && nbB && (jb - 1) % nbB != 0)
        {
            fprintf(stderr, "Error: JB must be a multiple of nbB\n");
            exit(1);
        }

        if (iq && mbQ && (iq - 1) % mbQ != 0)
        {
            fprintf(stderr, "Error: IQ must be a multiple of mbQ\n");
            exit(1);
        }

        if (jq && nbQ && (jq - 1) % nbQ != 0)
        {
            fprintf(stderr, "Error: JQ must be a multiple of nbQ\n");
            exit(1);
        }
    }
};

static inline int getLocalRank()
{
    int localRank;
    MPI_Comm localComm;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);

    return localRank;
}

static calError_t allgather(void *src_buf, void *recv_buf, size_t size, void *data,  void **request)
{
    MPI_Request req;
    int err = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size, MPI_BYTE, reinterpret_cast<MPI_Comm>(data), &req);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    *request = reinterpret_cast<void*>(req);
    return CAL_OK;
}

static calError_t request_test(void *request)
{
    MPI_Request req = reinterpret_cast<MPI_Request>(request);
    int         completed;
    int err = MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

static calError_t request_free(void *request)
{
    return CAL_OK;
}
