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

#include <numeric>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cufftMp.h>
#include <mpi.h>
#include <nccl.h>

#include "../common/error_checks.hpp"
#include "../common/nccl_check.hpp"
#include "../common/generate_random.hpp"
#include "../common/scaling.cuh"
#include "../iterators/box_iterator.hpp"

/**
 * This sample illustrates how to share scratch memory among plans
 * with the NCCL communication backend (CUFFT_COMM_NCCL).
 *
 * This can be done like this
 * 1. Create (but don't make) plans
 * 2. Call cufftSetAutoAllocation(plan, false) on all plans, on all ranks
 * 3. Call cufftMpMakePlan3d(plan, ..., &nccl_comm, CUFFT_COMM_NCCL, &scratch_size) on all plans
 *    and retrieve the required scratch size per plan
 * 4. Compute the maximum scratch size across plans. With NCCL, each rank can use its own value.
 * 5. Allocate memory using cudaMalloc
 * 6. Call cufftSetWorkArea(plan, buffer) on all plans
 * 7. Call cufftExec, cufftXtMemcpy, etc
 * 8. Destroy the plans and free the memory
 *
 * With NCCL, cufftSetAutoAllocation(plan, false) must come before the plan is made.
 * Such a plan can need more scratch than a plan with its own work area, and it can be slower.
 * Thus two plans with their own work areas can use less memory than two plans that share one
 * work area. Compare the workSize values on your system.
 */

/**
 * Input and output data are in X-Slabs
 */
void run_r2c_c2r(size_t nx, size_t ny, size_t nz, float* cpu_data, const int rank, const int size, ncclComm_t nccl_comm) {

    // Initialize plans and stream
    cufftHandle plan_r2c = 0;
    cufftHandle plan_c2r = 0;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));

    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));

    // Set the stream
    CUFFT_CHECK(cufftSetStream(plan_r2c, stream));
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));

    // We don't want to duplicate scratch, so we ask cuFFT to not allocate any
    CUFFT_CHECK(cufftSetAutoAllocation(plan_r2c, false));
    CUFFT_CHECK(cufftSetAutoAllocation(plan_c2r, false));

    // Make the plan and retrieve the required scratch size per plan
    size_t scratch_sizes[2];
    CUFFT_CHECK(cufftMpMakePlan3d(plan_r2c, nx, ny, nz, CUFFT_R2C, &nccl_comm, CUFFT_COMM_NCCL, &scratch_sizes[0]));
    CUFFT_CHECK(cufftMpMakePlan3d(plan_c2r, nx, ny, nz, CUFFT_C2R, &nccl_comm, CUFFT_COMM_NCCL, &scratch_sizes[1]));

    // Compute how much scratch to allocate. With NCCL, no maximum across ranks is necessary.
    size_t scratch_size = std::max(scratch_sizes[0], scratch_sizes[1]);

    // With NCCL, any device memory works
    void* scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&scratch, scratch_size));
    printf("Allocated %zu B of user scratch at %p on rank %d/%d\n", scratch_size, scratch, rank, size);

    // Pass the scratch to cuFFT
    CUFFT_CHECK(cufftSetWorkArea(plan_c2r, scratch));
    CUFFT_CHECK(cufftSetWorkArea(plan_r2c, scratch));

    // Allocate GPU memory, copy CPU data to GPU
    // Data is initially distributed according to CUFFT_XT_FORMAT_INPLACE
    cudaLibXtDesc *desc;
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &desc, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMemcpy(plan_r2c, (void*)desc, (void*)cpu_data, CUFFT_COPY_HOST_TO_DEVICE));

    // Run R2C
    CUFFT_CHECK(cufftXtExecDescriptor(plan_r2c, desc, desc, CUFFT_FORWARD));

    // At this point, data is distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    // This applies an element-wise scaling function to the GPU data located in desc->descriptor->data[0]
    auto [begin_d, end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_R2C,
                                         rank, size, nx, ny, nz, (cufftComplex*)desc->descriptor->data[0]);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 128;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());

    // Run C2R
    CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE));

    // Copy back to CPU and free
    // Data is again distributed according to CUFFT_XT_FORMAT_INPLACE
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUFFT_CHECK(cufftXtMemcpy(plan_c2r, (void*)cpu_data, (void*)desc, CUFFT_COPY_DEVICE_TO_HOST));
    CUFFT_CHECK(cufftXtFree(desc));

    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));

    // The plans do not use the scratch after cufftDestroy
    CUDA_CHECK(cudaFree(scratch));

    CUDA_CHECK(cudaStreamDestroy(stream));
};

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, size, id, rank));

    // Logical transform size
    size_t nx = size;      // any value >= size is OK
    size_t ny = size;      // any value >= size is OK
    size_t nz = 2 * size;  // needs to be even and >= size

    // We start with Slabs distributed along X (X-Slabs)
    // Ranks 0 ... (nx % size - 1) own 1 more element in the X dimension
    // All ranks own all elements in the Y and Z dimensions
    // The Z dimension has to be padded to accommodate the (nz / 2 + 1)
    // complex numbers assuming an in-place data layout.
    int ranks_with_onemore = nx % size;
    size_t my_nx = (nx / size) + (rank < ranks_with_onemore ? 1 : 0);
    size_t padded_nz = 2 * (nz / 2 + 1);

    // Local, distributed, data
    std::vector<float> data(my_nx * ny * padded_nz, 1.0);
    generate_random(data, rank);
    std::vector<float> ref = data;

    run_r2c_c2r(nx, ny, nz, data.data(), rank, size, nccl_comm);

    // Compute the error before exiting. This requires an MPI_Allreduce to collect the error on all ranks
    double error = compute_error(ref, data, buildBox3D(CUFFT_XT_FORMAT_INPLACE, CUFFT_R2C, rank, size, nx, ny, nz), MPI_COMM_WORLD);

    // Assess error, print on rank 0
    int code = assess_error(error, rank);

    // cuFFTMp does not destroy the communicator: the application owns it
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    MPI_Finalize();

    return code;
}
