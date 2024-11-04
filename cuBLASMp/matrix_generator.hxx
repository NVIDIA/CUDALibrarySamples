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

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

template <typename T, typename Generator>
void generate_distributed_matrix(
    int64_t m,
    int64_t n,
    T* a,
    int64_t mb,
    int64_t nb,
    int64_t ia,
    int64_t ja,
    int64_t lld,
    int64_t nprow,
    int64_t npcol,
    int64_t myprow,
    int64_t mypcol,
    Generator generator)
{
    if ((ia - 1) % mb != 0 || (ja - 1) % nb != 0)
    {
        throw std::runtime_error("offsets are not supported yet");
    }

    const int64_t numRowTiles = (m + mb - 1) / mb;
    const int64_t numColTiles = (n + nb - 1) / nb;
    mb = std::min(m, mb);
    nb = std::min(n, nb);

    for (int64_t i = 0; i < numRowTiles; i++)
    {
        for (int64_t j = 0; j < numColTiles; j++)
        {
            const int64_t tileI = (ia - 1) / mb + i;
            const int64_t tileJ = (ja - 1) / nb + j;

            const int64_t tileRowRank = tileI % nprow;
            const int64_t tileColRank = tileJ % npcol;

            if (tileRowRank == myprow && tileColRank == mypcol)
            {
                const int64_t locI = tileI / nprow;
                const int64_t locJ = tileJ / npcol;
                T* ptr = a + locI * mb + locJ * nb * lld;

                const int64_t tile_m = std::min({ m, mb, m - tileI * mb });
                const int64_t tile_n = std::min({ n, nb, n - tileJ * nb });

                for (int64_t k = 0; k < tile_m; k++)
                {
                    for (int64_t l = 0; l < tile_n; l++)
                    {
                        generator(ptr[k + l * lld], k, l, i == j);
                    }
                }
            }
        }
    }
}

template <typename T>
void generate_random_matrix(
    int64_t m,
    int64_t n,
    T* a,
    int64_t mb,
    int64_t nb,
    int64_t ia,
    int64_t ja,
    int64_t lld,
    int64_t nprow,
    int64_t npcol,
    int64_t myprow,
    int64_t mypcol)
{
    int rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    srand(rank);

    auto generator = [&](T& val, int64_t i, int64_t j, bool diag) { val = T(double(rand()) / RAND_MAX); };

    generate_distributed_matrix(m, n, a, mb, nb, ia, ja, lld, nprow, npcol, myprow, mypcol, generator);
}

template <typename T>
void generate_diag_matrix(
    int64_t m,
    int64_t n,
    T* a,
    int64_t mb,
    int64_t nb,
    int64_t ia,
    int64_t ja,
    int64_t lld,
    int64_t nprow,
    int64_t npcol,
    int64_t myprow,
    int64_t mypcol)
{
    generate_random_matrix(m, n, a, mb, nb, ia, ja, lld, nprow, npcol, myprow, mypcol);

    auto generator = [&](T& val, int64_t i, int64_t j, bool diag) {
        if (diag && i == j)
        {
            val += double(std::max(m, n));
        }
    };

    generate_distributed_matrix(m, n, a, mb, nb, ia, ja, lld, nprow, npcol, myprow, mypcol, generator);
}
