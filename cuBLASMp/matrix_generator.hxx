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

template <typename T>
static void generate_values(
    int seed,
    T* buffer,
    int64_t size,
    bool device_allocation,
    double min = 0.0,
    double max = 1.0)
{
    T* ptr = buffer;

    if (device_allocation)
    {
        ptr = reinterpret_cast<T*>(malloc(size * sizeof(T)));
    }

    std::srand(seed);
    std::for_each(ptr, ptr + size, [&](T& x) {
        const double v = double(std::rand()) / RAND_MAX;
        x = T(min + v * (max - min));
    });

    if (device_allocation)
    {
        CUDA_CHECK(cudaMemcpy(buffer, ptr, size * sizeof(T), cudaMemcpyHostToDevice));
        free(ptr);
    }
}

static void generate_values(
    int seed,
    void* buffer,
    int64_t size,
    cudaDataType_t type,
    bool device_allocation = false,
    double min = 0.0,
    double max = 1.0)
{
    switch (type)
    {
        case CUDA_R_4F_E2M1:
            generate_values(seed, reinterpret_cast<__nv_fp4_e2m1*>(buffer), size, device_allocation, min, max);
            break;
        case CUDA_R_8F_E4M3:
            generate_values(seed, reinterpret_cast<__nv_fp8_e4m3*>(buffer), size, device_allocation, min, max);
            break;
        case CUDA_R_8F_E5M2:
            generate_values(seed, reinterpret_cast<__nv_fp8_e5m2*>(buffer), size, device_allocation, min, max);
            break;
        case CUDA_R_8F_UE8M0:
            generate_values(seed, reinterpret_cast<__nv_fp8_e8m0*>(buffer), size, device_allocation, min, max);
            break;
        case CUDA_R_16F:
            generate_values(seed, reinterpret_cast<__half*>(buffer), size, device_allocation, min, max);
            break;
        case CUDA_R_16BF:
            generate_values(seed, reinterpret_cast<__nv_bfloat16*>(buffer), size, device_allocation, min, max);
            break;
        case CUDA_R_32F:
            generate_values(seed, reinterpret_cast<float*>(buffer), size, device_allocation, min, max);
            break;
        case CUDA_R_64F:
            generate_values(seed, reinterpret_cast<double*>(buffer), size, device_allocation, min, max);
            break;
        default: throw std::runtime_error("unsupported datatype");
    }
}
