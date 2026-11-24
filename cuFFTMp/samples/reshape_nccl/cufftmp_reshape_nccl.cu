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

#include <vector>
#include <cstdio>
#include <cufftMp.h>
#include <iostream>
#include <mpi.h>
#include <nccl.h>

#include "../common/error_checks.hpp"
#include "../common/nccl_check.hpp"

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(size != 2) {
        printf("This example has to be run with 2 MPI ranks\n");
        return 1;
    }

    int ndevices;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank % ndevices));
    printf("Hello from rank %d/%d using GPU %d\n", rank, size, rank % ndevices);

    // Rank 0 creates the NCCL unique ID and MPI sends it to all ranks.
    // ncclCommInitRank uses the current device, so call cudaSetDevice first.
    ncclUniqueId id;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, size, id, rank));

    // Assume we have a 4x5 "global" world with the following data.
    //     ------------> y
    //   | [0  1  2  3  4  ]
    //   | [5  6  7  8  9  ]
    //   | [10 11 12 13 14 ]
    //   | [15 16 17 18 19 ]
    // x v
    //
    // Initially, data is distributed as follows
    // [0  1  2  | 3  4  ]
    // [5  6  7  | 8  9  ]
    // [10 11 12 | 13 14 ]
    // [15 16 17 | 18 19 ]
    //  ^rank 0^  ^rank 1^
    // where every rank owns part of it, stored in row major format
    //
    // and we wish to redistribute it as
    // rank 0 [0  1  2  3  4  ]
    //        [5  6  7  8  9  ]
    //        -----------------
    // rank 1 [10 11 12 13 14 ]
    //        [15 16 17 18 19 ]
    // where every rank should own part of it, stored in row major format
    //
    // To do so, we describe the data using boxes
    // where every box is described as low-high, where
    // low and high are the lower and upper corners, respectively
    //
    // - The "world" box is of size 4x5
    // - The input boxes are  [ (0,0)-(4,3), (0,3)--(4,5) ]
    // - The output boxes are [ (0,0)-(2,5), (2,0)--(4,5) ]
    //
    // Since our data is 2D, in the following, the first dimension is always 0-1
    Box3D in_box_0 = {
        {0, 0, 0}, {1, 4, 3}, {12, 3, 1}
    };
    Box3D in_box_1 = {
        {0, 0, 3}, {1, 4, 5}, {8, 2, 1}
    };
    Box3D out_box_0 = {
        {0, 0, 0}, {1, 2, 5}, {10, 5, 1}
    };
    Box3D out_box_1 = {
        {0, 2, 0}, {1, 4, 5}, {10, 5, 1}
    };

    std::vector<int> src_host, dst_host_expected;
    if(rank == 0) {
        src_host = {0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17};
        dst_host_expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    } else {
        src_host = {3, 4, 8, 9, 13, 14, 18, 19};
        dst_host_expected = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    }

    cufftReshapeHandle handle;
    CUFFT_CHECK(cufftMpCreateReshape(&handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    Box3D &in_box  = (rank == 0 ? in_box_0  : in_box_1);
    Box3D &out_box = (rank == 0 ? out_box_0 : out_box_1);
    CUFFT_CHECK(cufftMpMakeReshape(handle, sizeof(int), 3,
        in_box.lower,   in_box.upper, in_box.strides,
        out_box.lower,  out_box.upper, out_box.strides,
        &nccl_comm, CUFFT_COMM_NCCL));

    // With NCCL, the reshape packs the data before it sends it, so it needs a workspace
    size_t workspace;
    CUFFT_CHECK(cufftMpGetReshapeSize(handle, &workspace));
    printf("Rank %d needs a workspace of %zu B\n", rank, workspace);

    // With NCCL, any device memory works: no symmetric heap, and sizes can differ between ranks
    void *src = nullptr, *dst = nullptr, *scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&src, src_host.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dst, dst_host_expected.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&scratch, workspace));

    // Move CPU data to GPU, and set the destination data to 0.
    CUDA_CHECK(cudaMemcpyAsync(src, src_host.data(), src_host.size() * sizeof(int), cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemsetAsync(dst, 0, dst_host_expected.size() * sizeof(int), stream));

    // NCCL receives the data in stream order, so no barrier is necessary before the reshape
    // Apply reshape
    CUFFT_CHECK(cufftMpExecReshapeAsync(handle, dst, src, scratch, stream));

    // Move GPU data to CPU
    std::vector<int> dst_host(dst_host_expected.size());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(dst_host.data(), dst, dst_host_expected.size() * sizeof(int), cudaMemcpyDefault));
    CUDA_CHECK(cudaFree(src));
    CUDA_CHECK(cudaFree(dst));
    CUDA_CHECK(cudaFree(scratch));
    CUFFT_CHECK(cufftMpDestroyReshape(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Display data
    std::cout << "Input data on rank " << rank << ":";
    for(auto &v: src_host) std::cout << " " << v;
    std::cout << "\n";

    std::cout << "Expected output data on rank " << rank << ":";
    for(auto &v: dst_host_expected) std::cout << " " << v;
    std::cout << "\n";

    std::cout << "Output data on rank " << rank << ":";
    for(auto &v: dst_host) std::cout << " " << v;
    std::cout << "\n";

    int errors = 0;
    for(int i = 0; i < (int)dst_host_expected.size(); i++) {
        if(dst_host[i] != dst_host_expected[i]) errors++;
    }

    // cuFFTMp does not destroy the communicator: the application owns it
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    MPI_Finalize();

    if(errors == 0) {
        std::cout << "PASSED\n";
        return 0;
    } else {
        std::cout << "FAILED\n";
        return 1;
    }
}
