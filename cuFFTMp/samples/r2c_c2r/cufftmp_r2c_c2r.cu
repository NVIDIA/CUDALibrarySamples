#include <numeric>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cufftMp.h>
#include <mpi.h>

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

void run_r2c_c2r(size_t nx, size_t ny, size_t nz, float* cpu_data, const int rank, const int size, MPI_Comm comm) {

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

    // Make the plan
    size_t workspace;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, nx, ny, nz, CUFFT_R2C, &workspace));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, nx, ny, nz, CUFFT_C2R, &workspace));

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
    
    // Run C2R
    CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE));

    // Copy back to CPU and free
    // Data is again distributed according to CUFFT_XT_FORMAT_INPLACE
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUFFT_CHECK(cufftXtMemcpy(plan_c2r, (void*)cpu_data, (void*)desc, CUFFT_COPY_DEVICE_TO_HOST));
    CUFFT_CHECK(cufftXtFree(desc));

    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));

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
    run_r2c_c2r(nx, ny, nz, data.data(), rank, size, MPI_COMM_WORLD);

    // Compute error
    double error = compute_error(ref, data, buildBox3D(CUFFT_XT_FORMAT_INPLACE, CUFFT_R2C, rank, size, nx, ny, nz));

    MPI_Finalize();

    return assess_error(error);
}
