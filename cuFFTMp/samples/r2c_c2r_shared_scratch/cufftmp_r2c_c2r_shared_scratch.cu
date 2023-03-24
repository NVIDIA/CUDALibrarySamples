#include <numeric>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <nvshmem.h>
#include <cufftMp.h>
#include <mpi.h>

#include "../common/error_checks.hpp"
#include "../common/generate_random.hpp"
#include "../common/scaling.cuh"
#include "../iterators/box_iterator.hpp"

/**
 * This samples illustrates how to share scratch memory among plans to minimize memory usage.
 * 
 * This can be done like this
 * 1. Create (but don't make) plans
 * 2. Call cufftSetAutoAllocation(plan, false) on all plans
 * 3. Call cufftMakePlan3d(plan, ..., scratch_size) on all plans and retrieve the required scratch size per plan
 * 4. Compute the maximum scratch size accros plans _AND_ accross MPI ranks (see note below on nvshmem_malloc)
 * 5. Allocate memory using nvshmem_malloc
 * 6. Call cufftSetWorkArea(plan, buffer) on all plans
 * 7. Call cufftExec, cufftXtMemcpy, etc
 * 8. Free memory using nvshmem_free
 * 9. Destroy the plans
 */

/**
 * Computes the maximum of `local` accross MPI ranks
 */
size_t allreduce_max(size_t local, MPI_Comm comm) {
    long long int inout_ = local;
    MPI_Allreduce(MPI_IN_PLACE, &inout_, 1, MPI_LONG_LONG_INT, MPI_MAX, comm);
    return inout_;
}

/**
 * /!\ IMPORTANT /!\
 * 
 * nvshmem_malloc requires the same "size" argument on every MPI rank
 * Hence, if scratch_size is not identical on every rank, the max accross 
 * ranks should be used. 
 * See https://docs.nvidia.com/hpc-sdk/nvshmem/api/docs/gen/api/memory.html#c.nvshmem_malloc
 * 
 * Except for FFT kernels that don't require any scratch (like powers of 2), there
 * is no guarantees that cuFFT requires the same amount of scratch on all ranks.
 * Hence, the user should compute the max across MPI ranks (e.g. using MPI_Allreduce)
 * and pass this to nvshmem_malloc.
 */

/**
 * Input and out data are in X-Slabs
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

    // We don't want to duplicate scratch, so we ask cuFFT to not allocate any
    CUFFT_CHECK(cufftSetAutoAllocation(plan_r2c, false));
    CUFFT_CHECK(cufftSetAutoAllocation(plan_c2r, false));

    // Make the plan and retrieve the required scratch size per plan
    size_t scratch_sizes[2];
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, nx, ny, nz, CUFFT_R2C, &scratch_sizes[0]));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, nx, ny, nz, CUFFT_C2R, &scratch_sizes[1]));

    // Compute how much scratch to allocate
    size_t scratch_size = allreduce_max(std::max(scratch_sizes[0], scratch_sizes[1]), comm);

    // Allocate scratch size using NVSHMEM
    void* scratch = nvshmem_malloc(scratch_size);
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
    
    // Run C2R
    CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE));

    // Copy back to CPU and free
    // Data is again distributed according to CUFFT_XT_FORMAT_INPLACE
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUFFT_CHECK(cufftXtMemcpy(plan_c2r, (void*)cpu_data, (void*)desc, CUFFT_COPY_DEVICE_TO_HOST));
    CUFFT_CHECK(cufftXtFree(desc));

    // Free the sratch
    // We need to do this before cufftDestroy because we need to keep NVSHMEM initialized
    nvshmem_free(scratch);

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

    run_r2c_c2r(nx, ny, nz, data.data(), rank, size, MPI_COMM_WORLD);

    // Compute error
    double error = compute_error(ref, data, buildBox3D(CUFFT_XT_FORMAT_INPLACE, CUFFT_R2C, rank, size, nx, ny, nz));
    
    MPI_Finalize();

    return assess_error(error);
}
