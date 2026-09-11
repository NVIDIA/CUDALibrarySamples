/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * This example redistributes a distributed 2D tensor while preserving its
 * logical values. The source uses mode order [x,y] with contiguous slab
 * partitions on a 2D process grid. The destination uses mode order [y,x] with
 * block-cyclic partitions on the transposed process grid. Consequently, the
 * operation demonstrates mode permutation, rank-to-rank data movement, and a
 * change of distributed storage layout in one cutensorMpRedistribute call.
 */

#include "cutensorMp.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <vector>

namespace
{

bool mpiInitialized = false;

[[noreturn]] void fail(char const* library, char const* message, char const* file, int line)
{
    std::fprintf(stderr, "%s error at %s:%d: %s\n", library, file, line, message);
    if (mpiInitialized)
    {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        cudaError_t const status_ = (call);                                \
        if (status_ != cudaSuccess)                                        \
        {                                                                  \
            fail("CUDA", cudaGetErrorString(status_), __FILE__, __LINE__); \
        }                                                                  \
    } while (0)

#define MPI_CHECK(call)                                    \
    do                                                     \
    {                                                      \
        int const status_ = (call);                        \
        if (status_ != MPI_SUCCESS)                        \
        {                                                  \
            char message_[MPI_MAX_ERROR_STRING];           \
            int length_ = 0;                               \
            MPI_Error_string(status_, message_, &length_); \
            message_[length_] = '\0';                      \
            fail("MPI", message_, __FILE__, __LINE__);     \
        }                                                  \
    } while (0)

#define NCCL_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        ncclResult_t const status_ = (call);                               \
        if (status_ != ncclSuccess)                                        \
        {                                                                  \
            fail("NCCL", ncclGetErrorString(status_), __FILE__, __LINE__); \
        }                                                                  \
    } while (0)

#define CUTENSORMP_CHECK(call)                                                       \
    do                                                                               \
    {                                                                                \
        cutensorStatus_t const status_ = (call);                                     \
        if (status_ != CUTENSOR_STATUS_SUCCESS)                                      \
        {                                                                            \
            fail("cuTENSORMp", cutensorGetErrorString(status_), __FILE__, __LINE__); \
        }                                                                            \
    } while (0)

int64_t slabLocalExtent(int64_t globalExtent, int nranks, int rank)
{
    int64_t const base = globalExtent / nranks;
    int64_t const remainder = globalExtent % nranks;
    return base + (rank < remainder ? 1 : 0);
}

int64_t slabGlobalOffset(int64_t globalExtent, int nranks, int rank)
{
    int64_t const base = globalExtent / nranks;
    int64_t const remainder = globalExtent % nranks;
    return rank * base + std::min<int64_t>(rank, remainder);
}

int64_t cyclicLocalExtent(int64_t globalExtent, int64_t blockSize, int nranks, int rank)
{
    int64_t localExtent = 0;
    int64_t const numBlocks = (globalExtent + blockSize - 1) / blockSize;
    for (int64_t block = rank; block < numBlocks; block += nranks)
    {
        int64_t const blockBegin = block * blockSize;
        localExtent += std::min(blockSize, globalExtent - blockBegin);
    }
    return localExtent;
}

int64_t cyclicGlobalIndex(int64_t localIndex, int64_t blockSize, int nranks, int rank)
{
    int64_t const localBlock = localIndex / blockSize;
    int64_t const blockOffset = localIndex % blockSize;
    return (localBlock * nranks + rank) * blockSize + blockOffset;
}

float valueForGlobalCoordinate(int64_t x, int64_t y, int64_t extentX)
{
    return static_cast<float>(3 * (x + extentX * y) + 1);
}

void makeProcessGrid(int nranks, int& gridX, int& gridY)
{
    gridX = 1;
    for (int factor = 1; factor <= nranks / factor; ++factor)
    {
        if (nranks % factor == 0)
        {
            gridX = factor;
        }
    }
    gridY = nranks / gridX;
}

int localDevice()
{
    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm));

    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&localComm));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        fail("CUDA", "no CUDA devices are visible", __FILE__, __LINE__);
    }
    return localRank % deviceCount;
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_CHECK(MPI_Init(&argc, &argv));
    mpiInitialized = true;

    int rank = 0;
    int nranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    if (nranks < 2)
    {
        if (rank == 0)
        {
            std::fprintf(stderr, "This example requires at least two MPI ranks.\n");
        }
        MPI_CHECK(MPI_Finalize());
        mpiInitialized = false;
        return EXIT_FAILURE;
    }

    int const device = localDevice();
    CUDA_CHECK(cudaSetDevice(device));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    ncclUniqueId ncclId;
    if (rank == 0)
    {
        NCCL_CHECK(ncclGetUniqueId(&ncclId));
    }
    MPI_CHECK(MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm = nullptr;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, ncclId, rank));

    cutensorMpHandle_t handle = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreate(&handle, comm, device, stream));

    int gridX = 0;
    int gridY = 0;
    makeProcessGrid(nranks, gridX, gridY);

    int64_t const globalExtentX = static_cast<int64_t>(gridX) * 4 + 1;
    int64_t const globalExtentY = static_cast<int64_t>(gridY) * 4 + 1;
    int64_t const blockSize = 2;
    int64_t const srcExtent[] = {globalExtentX, globalExtentY};
    int64_t const dstExtent[] = {globalExtentY, globalExtentX};
    int64_t const srcNranksPerMode[] = {gridX, gridY};
    int64_t const dstNranksPerMode[] = {gridY, gridX};
    int64_t const dstBlockSize[] = {blockSize, blockSize};
    // Mode labels associate logical dimensions across descriptors. Different
    // orders therefore express the permutation xy -> yx.
    int32_t const srcModes[] = {'x', 'y'};
    int32_t const dstModes[] = {'y', 'x'};

    cutensorMpTensorDescriptor_t srcDesc = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreateTensorDescriptor(handle, &srcDesc, 2, srcExtent, nullptr, nullptr, nullptr,
                                                      srcNranksPerMode, nranks, nullptr, CUDA_R_32F));

    cutensorMpTensorDescriptor_t dstDesc = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreateTensorDescriptor(handle, &dstDesc, 2, dstExtent, nullptr, dstBlockSize, nullptr,
                                                      dstNranksPerMode, nranks, nullptr, CUDA_R_32F));

    cutensorMpOperationDescriptor_t operation = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreateRedistribution(handle, &operation, dstDesc, dstModes, srcDesc, srcModes));

    constexpr uint64_t deviceWorkspaceBudget = 64ULL * 1024ULL * 1024ULL;
    cutensorMpPlanPreference_t preference = nullptr;
    CUTENSORMP_CHECK(
        cutensorMpCreatePlanPreference(handle, &preference, CUTENSORMP_ALGO_DEFAULT, deviceWorkspaceBudget, 0));

    cutensorMpPlan_t plan = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreatePlan(handle, &plan, operation, preference));

    uint64_t deviceWorkspaceSize = 0;
    uint64_t hostWorkspaceSize = 0;
    CUTENSORMP_CHECK(cutensorMpPlanGetAttribute(handle, plan, CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE,
                                                &deviceWorkspaceSize, sizeof(deviceWorkspaceSize)));
    CUTENSORMP_CHECK(cutensorMpPlanGetAttribute(handle, plan, CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST,
                                                &hostWorkspaceSize, sizeof(hostWorkspaceSize)));

    int const srcCoordX = rank % gridX;
    int const srcCoordY = rank / gridX;
    int const dstCoordY = rank % gridY;
    int const dstCoordX = rank / gridY;

    int64_t const srcLocalExtentX = slabLocalExtent(globalExtentX, gridX, srcCoordX);
    int64_t const srcLocalExtentY = slabLocalExtent(globalExtentY, gridY, srcCoordY);
    int64_t const srcGlobalOffsetX = slabGlobalOffset(globalExtentX, gridX, srcCoordX);
    int64_t const srcGlobalOffsetY = slabGlobalOffset(globalExtentY, gridY, srcCoordY);
    int64_t const dstLocalExtentX = cyclicLocalExtent(globalExtentX, blockSize, gridX, dstCoordX);
    int64_t const dstLocalExtentY = cyclicLocalExtent(globalExtentY, blockSize, gridY, dstCoordY);

    std::vector<float> hostSrc(static_cast<size_t>(srcLocalExtentX * srcLocalExtentY));
    std::vector<float> hostDst(static_cast<size_t>(dstLocalExtentY * dstLocalExtentX));
    for (int64_t localY = 0; localY < srcLocalExtentY; ++localY)
    {
        for (int64_t localX = 0; localX < srcLocalExtentX; ++localX)
        {
            int64_t const globalX = srcGlobalOffsetX + localX;
            int64_t const globalY = srcGlobalOffsetY + localY;
            int64_t const localIndex = localX + srcLocalExtentX * localY;
            hostSrc[static_cast<size_t>(localIndex)] = valueForGlobalCoordinate(globalX, globalY, globalExtentX);
        }
    }

    float* src = nullptr;
    float* dst = nullptr;
    void* deviceWorkspace = nullptr;
    void* hostWorkspace = nullptr;
    CUDA_CHECK(cudaMalloc(&src, hostSrc.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, hostDst.size() * sizeof(float)));
    if (deviceWorkspaceSize != 0)
    {
        CUDA_CHECK(cudaMalloc(&deviceWorkspace, deviceWorkspaceSize));
    }
    if (hostWorkspaceSize != 0)
    {
        CUDA_CHECK(cudaMallocHost(&hostWorkspace, hostWorkspaceSize));
    }

    CUDA_CHECK(cudaMemcpyAsync(src, hostSrc.data(), hostSrc.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0, hostDst.size() * sizeof(float), stream));
    CUTENSORMP_CHECK(cutensorMpRedistribute(handle, plan, dst, src, deviceWorkspace, hostWorkspace));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(hostDst.data(), dst, hostDst.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int localErrors = 0;
    for (int64_t localX = 0; localX < dstLocalExtentX; ++localX)
    {
        int64_t const globalX = cyclicGlobalIndex(localX, blockSize, gridX, dstCoordX);
        for (int64_t localY = 0; localY < dstLocalExtentY; ++localY)
        {
            int64_t const globalY = cyclicGlobalIndex(localY, blockSize, gridY, dstCoordY);
            int64_t const localIndex = localY + dstLocalExtentY * localX;
            float const expected = valueForGlobalCoordinate(globalX, globalY, globalExtentX);
            if (globalX >= globalExtentX || globalY >= globalExtentY ||
                hostDst[static_cast<size_t>(localIndex)] != expected)
            {
                ++localErrors;
                if (localErrors <= 4)
                {
                    std::fprintf(stderr,
                                 "Rank %d mismatch at destination local coordinate (y=%lld, x=%lld): "
                                 "got %g, expected %g\n",
                                 rank, static_cast<long long>(localY), static_cast<long long>(localX),
                                 hostDst[static_cast<size_t>(localIndex)], expected);
                }
            }
        }
    }

    int globalErrors = 0;
    MPI_CHECK(MPI_Allreduce(&localErrors, &globalErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    if (rank == 0)
    {
        std::printf(
            "Redistributed a %lld-by-%lld tensor on a %d-by-%d process grid from modes [x,y] to [y,x], "
            "while changing both modes from contiguous slabs to block-cyclic blocks of size %lld.\n",
            static_cast<long long>(globalExtentX), static_cast<long long>(globalExtentY), gridX, gridY,
            static_cast<long long>(blockSize));
        std::printf("Required workspace per rank: device=%llu bytes, host=%llu bytes.\n",
                    static_cast<unsigned long long>(deviceWorkspaceSize),
                    static_cast<unsigned long long>(hostWorkspaceSize));
        std::printf("cuTENSORMp redistribution example: %s\n", globalErrors == 0 ? "PASSED" : "FAILED");
    }

    CUTENSORMP_CHECK(cutensorMpDestroyPlan(plan));
    CUTENSORMP_CHECK(cutensorMpDestroyOperationDescriptor(operation));
    CUTENSORMP_CHECK(cutensorMpDestroyPlanPreference(preference));
    CUTENSORMP_CHECK(cutensorMpDestroyTensorDescriptor(dstDesc));
    CUTENSORMP_CHECK(cutensorMpDestroyTensorDescriptor(srcDesc));
    CUTENSORMP_CHECK(cutensorMpClearOperationCache(handle));

    if (hostWorkspace != nullptr)
    {
        CUDA_CHECK(cudaFreeHost(hostWorkspace));
    }
    if (deviceWorkspace != nullptr)
    {
        CUDA_CHECK(cudaFree(deviceWorkspace));
    }
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFree(src));

    CUTENSORMP_CHECK(cutensorMpDestroy(handle));
    NCCL_CHECK(ncclCommDestroy(comm));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_CHECK(MPI_Finalize());
    mpiInitialized = false;
    return globalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
