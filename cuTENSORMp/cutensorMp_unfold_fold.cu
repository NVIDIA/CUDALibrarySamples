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
 * This example demonstrates distributed tensor matricization with cuTENSORMp
 * unfold/fold.
 *
 * Unfolding reshapes an entire distributed N-dimensional tensor into a
 * distributed 2-D matrix: the tensor modes are split into two ordered groups,
 * one group forms the matrix rows and the other the matrix columns. Folding is
 * the exact inverse, expanding the two matrix axes back into the original mode
 * groups. Both operations preserve every logical value; they only move data
 * between the tensor layout and the matrix layout.
 *
 * Why is this useful? Most dense linear-algebra libraries -- cuSOLVER,
 * cuSOLVERMp, and ScaLAPACK among them -- do not operate on general
 * N-dimensional tensors. They only understand 2-D matrices. So whenever a
 * tensor computation needs a matrix factorization or solve (LU, Cholesky, QR,
 * SVD, eigensolve, ...), the natural pattern is:
 *
 *     unfold  (tensor -> matrix)
 *     solve   (hand the matrix to cuSOLVERMp / ScaLAPACK)
 *     fold    (matrix -> tensor)
 *
 * The matrix produced by unfold is an ordinary 2-D block-cyclic matrix with its
 * own descriptor, block sizes, and rank distribution. Crucially, the matrix
 * block boundaries may cut through the stride-1 (fastest-varying) tensor mode of
 * a group: a row/column block need not align with a full cycle of that mode.
 * This is exactly the case that a same-rank cutensorMpRedistribute
 * reinterpretation cannot naturally express, and it lets you pick the block size
 * the matrix library prefers (e.g. a cuSOLVERMp tile) instead of one dictated by
 * the tensor layout.
 *
 * The example builds a 4-D tensor T(i,j,k,l), unfolds it into a matrix whose
 * rows flatten (i,j) and columns flatten (k,l), leaves a placeholder where a
 * cuSOLVERMp solve would run, folds the matrix back into a tensor, and verifies
 * that the round trip reproduces the original distributed tensor bit for bit.
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

// Contiguous-slab (default block) partition of one mode across `nranks`: rank
// `rank` owns `base` or `base + 1` consecutive indices, matching the default
// block distribution used when cutensorMpCreateTensorDescriptor receives a null
// block array.
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

// Deterministic value for a global tensor coordinate. The flattened global
// index is well below 2^24, so it is represented exactly in FP32 and the
// tensor -> matrix -> tensor round trip must reproduce it bit for bit.
float valueForGlobalCoordinate(int64_t i, int64_t j, int64_t k, int64_t l, int64_t extentI, int64_t extentJ,
                               int64_t extentK)
{
    int64_t const globalIndex = i + extentI * (j + extentJ * (k + extentK * l));
    return static_cast<float>(globalIndex);
}

// Factor `nranks` into the most square 2-D process grid (gridX * gridY).
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
    // Example use case for unfold/fold:
    // (a) initialization (MPI, NCCL, CUDA, cuTENSORMp handle)
    // (b) problem definition (tensor descriptor, matrix descriptor, group mode indices, group size)
    // (c) unfold (unfolding operation descriptor, plan, workspace, execute)
    // (d) perform computation on the matricized tensor (e.g. solve, factorization, etc.)
    // (e) fold (folding operation descriptor, plan, workspace, execute)
    // (f) OPTIONAL: verification
    // (g) cleanup

    // ------------------------------------------------------------------
    // (a) Initialization: MPI, one GPU per rank, NCCL, and the cuTENSORMp
    //     handle bound to that communicator and stream.
    // ------------------------------------------------------------------
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

    // ------------------------------------------------------------------
    // (b) Problem definition. The 4-D tensor T(i,j,k,l) is matricized so that
    // the matrix rows flatten the (i,j) group and the columns flatten the (k,l)
    // group:
    //     row = i + I*j        M = I * J = 630
    //     col = k + K*l        N = K * L = 240
    // The matrix block sizes (96, 64) are multiples of 32 (the FP32 unfold/fold
    // tile granularity) but deliberately NOT multiples of the fastest-varying
    // mode extents I=70 / K=48, so the matrix blocks cut through those modes --
    // the case that plain redistribution cannot express.
    // ------------------------------------------------------------------
    int64_t const extentI = 70;
    int64_t const extentJ = 9;
    int64_t const extentK = 48;
    int64_t const extentL = 5;

    int64_t const tensorExtent[] = {extentI, extentJ, extentK, extentL};
    // Distribute the tensor as contiguous slabs along mode i only; the other
    // modes are held locally in full. This keeps the local <-> global mapping a
    // simple slab computation for seeding and checking the data.
    int64_t const tensorNranksPerMode[] = {nranks, 1, 1, 1};

    int64_t const matrixM = extentI * extentJ;
    int64_t const matrixN = extentK * extentL;
    int64_t const matrixExtent[] = {matrixM, matrixN};
    int64_t const matrixBlock[] = {96, 64};

    // The matrix uses an ordinary 2-D block-cyclic distribution on a process
    // grid, exactly like the layout a matrix library would expect.
    int gridX = 0;
    int gridY = 0;
    makeProcessGrid(nranks, gridX, gridY);
    int64_t const matrixNranksPerMode[] = {gridX, gridY};

    // Row group = modes (i,j); column group = modes (k,l). Indices are
    // positional in tensor-descriptor order, listed group by group.
    int32_t const groupModeIndices[] = {0, 1, 2, 3};
    int64_t const groupSize[] = {2, 2};

    cutensorMpTensorDescriptor_t descTensor = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreateTensorDescriptor(handle, &descTensor, 4, tensorExtent, nullptr, nullptr, nullptr,
                                                      tensorNranksPerMode, nranks, nullptr, CUDA_R_32F));

    cutensorMpTensorDescriptor_t descMatrix = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreateTensorDescriptor(handle, &descMatrix, 2, matrixExtent, nullptr, matrixBlock,
                                                      nullptr, matrixNranksPerMode, nranks, nullptr, CUDA_R_32F));

    // Local tensor shard: slab along mode i, full extents on j, k, l. The local
    // storage is dense column-major (mode i fastest), matching the default
    // descriptor layout.
    int64_t const localExtentI = slabLocalExtent(extentI, nranks, rank);
    int64_t const globalOffsetI = slabGlobalOffset(extentI, nranks, rank);
    int64_t const tensorLocalElements = localExtentI * extentJ * extentK * extentL;

    std::vector<float> hostTensorIn(static_cast<size_t>(tensorLocalElements));
    for (int64_t l = 0; l < extentL; ++l)
    {
        for (int64_t k = 0; k < extentK; ++k)
        {
            for (int64_t j = 0; j < extentJ; ++j)
            {
                for (int64_t li = 0; li < localExtentI; ++li)
                {
                    int64_t const globalI = globalOffsetI + li;
                    int64_t const localIndex = li + localExtentI * (j + extentJ * (k + extentK * l));
                    hostTensorIn[static_cast<size_t>(localIndex)] =
                        valueForGlobalCoordinate(globalI, j, k, l, extentI, extentJ, extentK);
                }
            }
        }
    }

    // The destination matrix shard size follows from the 2-D block-cyclic
    // distribution; query it from the descriptor instead of recomputing it.
    uint64_t matrixStorageBytes = 0;
    CUTENSORMP_CHECK(cutensorMpTensorDescriptorGetAttribute(
        handle, descMatrix, CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_STORAGE_SIZE, &matrixStorageBytes,
        sizeof(matrixStorageBytes)));

    float* tensorIn = nullptr;
    float* matrix = nullptr;
    float* tensorOut = nullptr;
    CUDA_CHECK(cudaMalloc(&tensorIn, std::max<size_t>(hostTensorIn.size() * sizeof(float), sizeof(float))));
    CUDA_CHECK(cudaMalloc(&matrix, std::max<uint64_t>(matrixStorageBytes, sizeof(float))));
    CUDA_CHECK(cudaMalloc(&tensorOut, std::max<size_t>(hostTensorIn.size() * sizeof(float), sizeof(float))));

    CUDA_CHECK(cudaMemcpyAsync(tensorIn, hostTensorIn.data(), hostTensorIn.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(matrix, 0, matrixStorageBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(tensorOut, 0, hostTensorIn.size() * sizeof(float), stream));

    // ------------------------------------------------------------------
    // (c) Unfold: tensor -> matrix. Build the unfolding operation descriptor,
    //     plan it, size the workspace, and execute.
    // ------------------------------------------------------------------
    cutensorMpOperationDescriptor_t unfoldOp = nullptr;
    CUTENSORMP_CHECK(
        cutensorMpCreateUnfolding(handle, &unfoldOp, descTensor, 2, groupModeIndices, groupSize, descMatrix));

    constexpr uint64_t deviceWorkspaceBudget = 256ULL * 1024ULL * 1024ULL;
    cutensorMpPlanPreference_t preference = nullptr;
    CUTENSORMP_CHECK(
        cutensorMpCreatePlanPreference(handle, &preference, CUTENSORMP_ALGO_DEFAULT, deviceWorkspaceBudget, 0));

    cutensorMpPlan_t unfoldPlan = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreatePlan(handle, &unfoldPlan, unfoldOp, preference));

    uint64_t unfoldDeviceWorkspaceSize = 0;
    uint64_t unfoldHostWorkspaceSize = 0;
    CUTENSORMP_CHECK(cutensorMpPlanGetAttribute(handle, unfoldPlan, CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE,
                                                &unfoldDeviceWorkspaceSize, sizeof(unfoldDeviceWorkspaceSize)));
    CUTENSORMP_CHECK(cutensorMpPlanGetAttribute(handle, unfoldPlan, CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST,
                                                &unfoldHostWorkspaceSize, sizeof(unfoldHostWorkspaceSize)));

    void* unfoldDeviceWorkspace = nullptr;
    void* unfoldHostWorkspace = nullptr;
    if (unfoldDeviceWorkspaceSize != 0)
    {
        CUDA_CHECK(cudaMalloc(&unfoldDeviceWorkspace, unfoldDeviceWorkspaceSize));
    }
    if (unfoldHostWorkspaceSize != 0)
    {
        CUDA_CHECK(cudaMallocHost(&unfoldHostWorkspace, unfoldHostWorkspaceSize));
    }

    CUTENSORMP_CHECK(cutensorMpUnfold(handle, unfoldPlan, tensorIn, matrix, unfoldDeviceWorkspace, unfoldHostWorkspace));

    // ------------------------------------------------------------------
    // (d) Solve on the matricized tensor.
    //
    //     At this point `matrix` is an ordinary distributed 2-D block-cyclic
    //     matrix (extents 630 x 240, blocks 96 x 64). This is precisely the
    //     layout that cuSOLVERMp or ScaLAPACK consume, so a real workflow would
    //     hand `matrix` (together with `descMatrix`'s block-cyclic layout) to a
    //     matrix factorization or solve here, for example:
    //
    //         cusolverMpPotrf(...)  // Cholesky factorization
    //         cusolverMpGetrf(...)  // LU factorization
    //         cusolverMpSyevd(...)  // symmetric eigensolver
    //
    //     This example intentionally performs no matrix computation so that the
    //     unfold -> fold round trip is a pure identity and can be verified
    //     exactly. The matrix is left unchanged and folded straight back.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // (e) Fold: matrix -> tensor. Note the parameter order differs from
    //     unfolding -- the matrix (source) and tensor (destination) descriptors
    //     come before the grouping arrays -- and the plan carries its own
    //     workspace requirements.
    // ------------------------------------------------------------------
    cutensorMpOperationDescriptor_t foldOp = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreateFolding(handle, &foldOp, descMatrix, descTensor, 2, groupModeIndices, groupSize));

    cutensorMpPlan_t foldPlan = nullptr;
    CUTENSORMP_CHECK(cutensorMpCreatePlan(handle, &foldPlan, foldOp, preference));

    uint64_t foldDeviceWorkspaceSize = 0;
    uint64_t foldHostWorkspaceSize = 0;
    CUTENSORMP_CHECK(cutensorMpPlanGetAttribute(handle, foldPlan, CUTENSORMP_PLAN_REQUIRED_WORKSPACE_DEVICE,
                                                &foldDeviceWorkspaceSize, sizeof(foldDeviceWorkspaceSize)));
    CUTENSORMP_CHECK(cutensorMpPlanGetAttribute(handle, foldPlan, CUTENSORMP_PLAN_REQUIRED_WORKSPACE_HOST,
                                                &foldHostWorkspaceSize, sizeof(foldHostWorkspaceSize)));

    void* foldDeviceWorkspace = nullptr;
    void* foldHostWorkspace = nullptr;
    if (foldDeviceWorkspaceSize != 0)
    {
        CUDA_CHECK(cudaMalloc(&foldDeviceWorkspace, foldDeviceWorkspaceSize));
    }
    if (foldHostWorkspaceSize != 0)
    {
        CUDA_CHECK(cudaMallocHost(&foldHostWorkspace, foldHostWorkspaceSize));
    }

    CUTENSORMP_CHECK(cutensorMpFold(handle, foldPlan, matrix, tensorOut, foldDeviceWorkspace, foldHostWorkspace));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ------------------------------------------------------------------
    // (f) Verification (optional). This section is illustrative: it shows the
    //     kinds of checks a caller can perform. In a real unfold -> solve ->
    //     fold pipeline the folded tensor would NOT equal the input (the solve
    //     changes the data), so the round-trip check (f2) below would be
    //     replaced by application-specific validation; the layout introspection
    //     (f1) applies regardless.
    // ------------------------------------------------------------------

    // (f1) Introspect the realized rank-local layout of the matricized tensor.
    //      The library derives these attributes from the global extents, block
    //      sizes, and process grid we requested; querying them is how a caller
    //      inspects what each rank actually owns. Note this reads descriptor
    //      metadata fixed at descriptor-creation time -- it does not by itself
    //      confirm the unfold moved data correctly (that is what (f2) does).
    uint32_t matrixNumModes = 0;
    int64_t matrixLocalExtent[2] = {0, 0};
    int64_t matrixLocalBlock[2] = {0, 0};
    uint64_t matrixLocalStorageBytes = 0;
    CUTENSORMP_CHECK(cutensorMpTensorDescriptorGetAttribute(
        handle, descMatrix, CUTENSORMP_TENSOR_DESCRIPTOR_NUM_MODES, &matrixNumModes, sizeof(matrixNumModes)));
    CUTENSORMP_CHECK(cutensorMpTensorDescriptorGetAttribute(
        handle, descMatrix, CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_EXTENT, matrixLocalExtent, sizeof(matrixLocalExtent)));
    CUTENSORMP_CHECK(cutensorMpTensorDescriptorGetAttribute(
        handle, descMatrix, CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_BLOCK_SIZE, matrixLocalBlock, sizeof(matrixLocalBlock)));
    CUTENSORMP_CHECK(cutensorMpTensorDescriptorGetAttribute(handle, descMatrix,
                                                            CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_STORAGE_SIZE,
                                                            &matrixLocalStorageBytes, sizeof(matrixLocalStorageBytes)));
    if (rank == 0)
    {
        std::printf("Rank 0 owns a %lld-by-%lld block of the %u-mode matrix (local block size %lld-by-%lld, "
                    "local storage %llu bytes).\n",
                    static_cast<long long>(matrixLocalExtent[0]), static_cast<long long>(matrixLocalExtent[1]),
                    matrixNumModes, static_cast<long long>(matrixLocalBlock[0]),
                    static_cast<long long>(matrixLocalBlock[1]),
                    static_cast<unsigned long long>(matrixLocalStorageBytes));
    }

    // (f2) Round-trip data check. Valid here ONLY because no solve was performed
    //      between unfold and fold, so the fold must reproduce this rank's local
    //      tensor shard bit for bit.
    std::vector<float> hostTensorOut(static_cast<size_t>(tensorLocalElements));
    CUDA_CHECK(cudaMemcpy(hostTensorOut.data(), tensorOut, hostTensorOut.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    int localErrors = 0;
    for (int64_t index = 0; index < tensorLocalElements; ++index)
    {
        if (hostTensorOut[static_cast<size_t>(index)] != hostTensorIn[static_cast<size_t>(index)])
        {
            ++localErrors;
            if (localErrors <= 4)
            {
                std::fprintf(stderr, "Rank %d round-trip mismatch at local index %lld: got %g, expected %g\n", rank,
                             static_cast<long long>(index), hostTensorOut[static_cast<size_t>(index)],
                             hostTensorIn[static_cast<size_t>(index)]);
            }
        }
    }

    int globalErrors = 0;
    MPI_CHECK(MPI_Allreduce(&localErrors, &globalErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    if (rank == 0)
    {
        std::printf(
            "Unfolded a %lld-by-%lld-by-%lld-by-%lld tensor into a %lld-by-%lld matrix "
            "(row group (i,j), column group (k,l)) with %lld-by-%lld block-cyclic tiles on a %d-by-%d grid, "
            "then folded it back.\n",
            static_cast<long long>(extentI), static_cast<long long>(extentJ), static_cast<long long>(extentK),
            static_cast<long long>(extentL), static_cast<long long>(matrixM), static_cast<long long>(matrixN),
            static_cast<long long>(matrixBlock[0]), static_cast<long long>(matrixBlock[1]), gridX, gridY);
        std::printf("Required workspace per rank: unfold device=%llu host=%llu, fold device=%llu host=%llu (bytes).\n",
                    static_cast<unsigned long long>(unfoldDeviceWorkspaceSize),
                    static_cast<unsigned long long>(unfoldHostWorkspaceSize),
                    static_cast<unsigned long long>(foldDeviceWorkspaceSize),
                    static_cast<unsigned long long>(foldHostWorkspaceSize));
        std::printf("cuTENSORMp unfold/fold example: %s\n", globalErrors == 0 ? "PASSED" : "FAILED");
    }

    // ------------------------------------------------------------------
    // (g) Cleanup: destroy plans, operation descriptors, the plan preference,
    //     tensor descriptors, workspaces, buffers, the handle, and the NCCL
    //     communicator and CUDA stream.
    // ------------------------------------------------------------------
    CUTENSORMP_CHECK(cutensorMpDestroyPlan(foldPlan));
    CUTENSORMP_CHECK(cutensorMpDestroyPlan(unfoldPlan));
    CUTENSORMP_CHECK(cutensorMpDestroyOperationDescriptor(foldOp));
    CUTENSORMP_CHECK(cutensorMpDestroyOperationDescriptor(unfoldOp));
    CUTENSORMP_CHECK(cutensorMpDestroyPlanPreference(preference));
    CUTENSORMP_CHECK(cutensorMpDestroyTensorDescriptor(descMatrix));
    CUTENSORMP_CHECK(cutensorMpDestroyTensorDescriptor(descTensor));
    CUTENSORMP_CHECK(cutensorMpClearOperationCache(handle));

    if (foldHostWorkspace != nullptr)
    {
        CUDA_CHECK(cudaFreeHost(foldHostWorkspace));
    }
    if (foldDeviceWorkspace != nullptr)
    {
        CUDA_CHECK(cudaFree(foldDeviceWorkspace));
    }
    if (unfoldHostWorkspace != nullptr)
    {
        CUDA_CHECK(cudaFreeHost(unfoldHostWorkspace));
    }
    if (unfoldDeviceWorkspace != nullptr)
    {
        CUDA_CHECK(cudaFree(unfoldDeviceWorkspace));
    }
    CUDA_CHECK(cudaFree(tensorOut));
    CUDA_CHECK(cudaFree(matrix));
    CUDA_CHECK(cudaFree(tensorIn));

    CUTENSORMP_CHECK(cutensorMpDestroy(handle));
    NCCL_CHECK(ncclCommDestroy(comm));
    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_CHECK(MPI_Finalize());
    mpiInitialized = false;
    return globalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
