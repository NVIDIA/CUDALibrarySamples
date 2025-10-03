/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#pragma once

#include <stdbool.h>
#include <string.h>
#include <mpi.h>

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
           "mbA=%d nbA=%d mbB=%d nbB=%d mbQ=%d nbQ=%d mbZ=%d nbZ=%d "
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