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
#include <complex>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <cufft.h>
#include <cufftMp.h>
#include <mpi.h>
#include <nccl.h>

#include "../common/error_checks.hpp"
#include "../common/nccl_check.hpp"
#include "../common/scaling.cuh"
#include "../common/generate_random.hpp"
#include "../iterators/box_iterator.hpp"

/**
 * This sample illustrates a basic use of cuFFTMp using the built-in, optimized, data distributions.
 * It uses the NCCL communication backend (CUFFT_COMM_NCCL) instead of NVSHMEM:
 * - the application creates the NCCL communicator and passes it to cufftMpMakePlan3d,
 * - the data buffer is plain device memory from cudaMalloc,
 * - the application destroys the communicator after the last cuFFTMp call.
 *
 * It assumes the CPU data is initially distributed according to CUFFT_XT_FORMAT_INPLACE, a.k.a. X-Slabs.
 * Given a global array of size X * Y * Z, every MPI rank owns approximately (X / ngpus) * Y * Z entries.
 * More precisely,
 * - The first (X % ngpus) MPI ranks each own (X / ngpus + 1) planes of size Y * Z,
 * - The remaining MPI ranks each own (X / ngpus) planes of size Y * Z
 *
 * The CPU data is then copied to the GPU and a forward transform is applied.
 *
 * After that transform, GPU data is distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED, a.k.a. Y-Slabs.
 * Given a global array of size X * Y * Z, every MPI rank owns approximately X * (Y / ngpus) * Z entries.
 * More precisely,
 * - The first (Y % ngpus) MPI ranks each own (Y / ngpus + 1) planes of size X * Z,
 * - The remaining MPI ranks each own (Y / ngpus) planes of size X * Z
 *
 * A scaling kernel is applied, on the distributed GPU data (distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED)
 * This kernel prints some elements to illustrate the CUFFT_XT_FORMAT_INPLACE_SHUFFLED data distribution and
 * normalizes the entries by (nx * ny * nz)
 *
 * Finally, a backward transform is applied.
 * After this, data is again distributed according to CUFFT_XT_FORMAT_INPLACE, same as the input data.
 *
 * Data is finally copied back to CPU and compared to the input data. They should be almost identical.
 */

void run_c2c_fwd_inv(size_t nx, size_t ny, size_t nz, std::vector<std::complex<float>>& cpu_data, int rank, int size, ncclComm_t nccl_comm) {

    // Allocate GPU memory, copy CPU data to GPU
    // Data is initially distributed according to CUFFT_XT_FORMAT_INPLACE
    // With NCCL, any device memory works: no symmetric heap, and sizes can differ between ranks
    // An in-place buffer must hold the larger of the local input (X-slab) and the local output (Y-slab).
    // The two sizes are equal here because nx == ny.
    cuComplex* gpu_data = nullptr;
    CUDA_CHECK(cudaMalloc(&gpu_data, cpu_data.size() * sizeof(cuComplex)));
    CUDA_CHECK(cudaMemcpy(gpu_data, cpu_data.data(), cpu_data.size() * sizeof(cuComplex), cudaMemcpyDefault));

    cufftHandle plan = 0;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));

    CUFFT_CHECK(cufftCreate(&plan));

    CUFFT_CHECK(cufftSetStream(plan, stream));

    CUFFT_CHECK(cufftXtSetSubformatDefault(plan, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));

    size_t workspace;
    CUFFT_CHECK(cufftMpMakePlan3d(plan, nx, ny, nz, CUFFT_C2C, &nccl_comm, CUFFT_COMM_NCCL, &workspace));

    // Run C2C Fwd
    CUFFT_CHECK(cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD));

    // Data is now distributed as Y-Slabs
    // We run a kernel on the distributed data, using the BoxIterator's for convenience
    auto[begin_d, end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_C2C,
                                        rank, size, nx, ny, nz, gpu_data);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 128;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());

    // Run C2C Bwd
    CUFFT_CHECK(cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_INVERSE));

    // Copy back and free
    // Data is distributed as X-Slabs again
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(cpu_data.data(), gpu_data, cpu_data.size() * sizeof(cuComplex), cudaMemcpyDefault));

    CUDA_CHECK(cudaFree(gpu_data));
    CUFFT_CHECK(cufftDestroy(plan));

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

    // Rank 0 creates the NCCL unique ID and MPI sends it to all ranks.
    // ncclCommInitRank uses the current device, so call cudaSetDevice first.
    ncclUniqueId id;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, size, id, rank));

    size_t nx = (argc >= 2 ? atoi(argv[1]) : 8*size);  // any value >= size is OK
    size_t ny = (argc >= 2 ? atoi(argv[1]) : 8*size);  // any value >= size is OK
    size_t nz = (argc >= 2 ? atoi(argv[1]) : 8*size);  // any value >= size is OK

    // We start with X-Slabs
    // Ranks 0 ... (nx % size - 1) have 1 more element in the X dimension
    // and every rank owns all elements in the Y and Z dimensions.
    int ranks_cutoff = nx % size;
    size_t my_nx = (nx / size) + (rank < ranks_cutoff ? 1 : 0);
    size_t my_ny =  ny;
    size_t my_nz =  nz;

    printf("Hello from rank %d/%d using GPU %d transform of size %zu x %zu x %zu, local size %zu x %zu x %zu\n", rank, size, rank % ndevices, nx, ny, nz, my_nx, my_ny, my_nz);

    // Generate local, distributed, data
    std::vector<std::complex<float>> data(my_nx * my_ny * my_nz);
    generate_random(data, rank);
    std::vector<std::complex<float>> ref = data;

    // Run Forward and Inverse FFT
    run_c2c_fwd_inv(nx, ny, nz, data, rank, size, nccl_comm);

    // Compute the error before exiting. This requires an MPI_Allreduce to collect the error on all ranks
    double error = compute_error(ref, data, buildBox3D(CUFFT_XT_FORMAT_INPLACE, CUFFT_C2C, rank, size, nx, ny, nz), MPI_COMM_WORLD);

    // Assess error, print on rank 0
    int code = assess_error(error, rank);

    // cuFFTMp does not destroy the communicator: the application owns it
    NCCL_CHECK(ncclCommDestroy(nccl_comm));
    MPI_Finalize();

    return code;
}
