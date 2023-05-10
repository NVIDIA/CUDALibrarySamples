#include <numeric>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cufftMp.h>
#include <mpi.h>
#include <nvshmem.h>

#include "../common/error_checks.hpp"
#include "../common/generate_random.hpp"
#include "../common/scaling.cuh"
#include "../iterators/box_iterator.hpp"

/**
 * This samples illustrates a basic use of cuFFTMp using the built-in, optimized, data distributions
 * in the case of an R2C - C2R transform
 * 
 * It performs 
 * - forward transform
 * - printing and scaling of the entries
 * - inverse transform
 */

void run_r2c_c2r(size_t nx, size_t ny, size_t nz, std::vector<float>& cpu_data, const int rank, const int size, MPI_Comm comm) {

    // Allocate GPU memory, copy CPU data to GPU
    // Data is initially distributed according to CUFFT_XT_FORMAT_INPLACE
    cuComplex* gpu_data = (cuComplex*)nvshmem_malloc(cpu_data.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpy(gpu_data, cpu_data.data(), cpu_data.size() * sizeof(float), cudaMemcpyDefault));

    // Initialize plans and stream
    cufftHandle plan_r2c = 0;
    cufftHandle plan_c2r = 0;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));

    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));

    // Attach the MPI communicator to the plans
    CUFFT_CHECK(cufftMpAttachComm(plan_r2c, CUFFT_COMM_MPI, &comm));
    CUFFT_CHECK(cufftMpAttachComm(plan_c2r, CUFFT_COMM_MPI, &comm));

    // Set the stream
    CUFFT_CHECK(cufftSetStream(plan_r2c, stream));
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));

    // Set default subformats
    CUFFT_CHECK(cufftXtSetSubformatDefault(plan_r2c, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));
    CUFFT_CHECK(cufftXtSetSubformatDefault(plan_c2r, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));

    // Make the plan
    size_t workspace;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, nx, ny, nz, CUFFT_R2C, &workspace));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, nx, ny, nz, CUFFT_C2R, &workspace));

    // Run R2C
    // cufftXtSetSubformatDefault(plan_r2c, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED) + CUFFT_FORWARD
    // means gpu_data is distributed according to CUFFT_XT_FORMAT_INPLACE
    // Note: R2C transforms are implicitly forward
    CUFFT_CHECK(cufftExecR2C(plan_r2c, (cufftReal*)gpu_data, (cufftComplex*)gpu_data));

    // At this point, data is distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    auto [begin_d, end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_R2C, 
                                         rank, size, nx, ny, nz, gpu_data);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 128;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz);
    
    // Run C2R
    // cufftXtSetSubformatDefault(plan_c2r, CUFFT_XT_FORMAT_INPLACE, CUFFT_XT_FORMAT_INPLACE_SHUFFLED) + CUFFT_INVERSE
    // means gpu_data is distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED
    // Note: C2R transforms are implicitly inverse
    CUFFT_CHECK(cufftExecC2R(plan_c2r, (cufftComplex*)gpu_data, (cufftReal*)gpu_data));

    // Copy back to CPU and free
    // Data is again distributed according to CUFFT_XT_FORMAT_INPLACE
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(cpu_data.data(), gpu_data, cpu_data.size() * sizeof(float), cudaMemcpyDefault));

    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));

    CUDA_CHECK(cudaStreamDestroy(stream));
    nvshmem_free(gpu_data);
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

    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = (void*)&comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Logical transform size
    size_t nx = size;      // any value >= size is OK
    size_t ny = size;      // any value >= size is OK
    size_t nz = 2 * size;  // need to be even and >= size

    // We start with Slabs distributed along X (X-Slabs)
    // Ranks 0 ... (nx % size - 1) own 1 more element in the X dimension
    // All ranks own all element in the Y and Z dimension
    // The Z dimension has to be padded to accomodate the (nz / 2 + 1) 
    // complex numbers assuming an in-place data layout.
    int ranks_with_onemore = nx % size;
    size_t my_nx = (nx / size) + (rank < ranks_with_onemore ? 1 : 0);
    size_t padded_nz = 2 * (nz / 2 + 1);

    // Local, distributed, data
    std::vector<float> data(my_nx * ny * padded_nz, 1.0);
    generate_random(data, rank);
    std::vector<float> ref = data;

    // R2C + scaling + C2R
    run_r2c_c2r(nx, ny, nz, data, rank, size, MPI_COMM_WORLD);

    // Compute error
    double error = compute_error(ref, data, buildBox3D(CUFFT_XT_FORMAT_INPLACE, CUFFT_R2C, rank, size, nx, ny, nz));

    nvshmem_init();
    MPI_Finalize();

    return assess_error(error);
}
