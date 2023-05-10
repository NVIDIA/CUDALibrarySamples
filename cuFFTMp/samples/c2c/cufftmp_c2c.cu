#include <numeric>
#include <vector>
#include <complex>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <cufft.h>
#include <cufftMp.h>
#include <mpi.h>

#include "../common/error_checks.hpp"
#include "../common/scaling.cuh"
#include "../common/generate_random.hpp"
#include "../iterators/box_iterator.hpp"

/**
 * This samples illustrates a basic use of cuFFTMp using the built-in, optimized, data distributions.
 * 
 * It assumes the CPU data is initially distributed according to CUFFT_XT_FORMAT_INPLACE, a.k.a. X-Slabs.
 * Given a global array of size X * Y * Z, every MPI rank owns approximately (X / ngpus) * Y * Z entries.
 * More precisely, 
 * - The first (X % ngpus) MPI rank each own (X / ngpus + 1) planes of size Y * Z,
 * - The remaining MPI rank each own (X / ngpus) planes of size Y * Z
 * 
 * The CPU data is then copied on GPU and a forward transform is applied.
 * 
 * After that transform, GPU data is distributed according to CUFFT_XT_FORMAT_INPLACE_SHUFFLED, a.k.a. Y-Slabs.
 * Given a global array of size X * Y * Z, every MPI rank owns approximately X * (Y / ngpus) * Z entries.
 * More precisely, 
 * - The first (Y % ngpus) MPI rank each own (Y / ngpus + 1) planes of size X * Z,
 * - The remaining MPI rank each own (Y / ngpus) planes of size X * Z
 * 
 * A scaling kerel is applied, on the distributed GPU data (distributed according to CUFFT_XT_FORMAT_INPLACE)
 * This kernel prints some elements to illustrate the CUFFT_XT_FORMAT_INPLACE_SHUFFLED data distribution and
 * normalize entries by (nx * ny * nz)
 * 
 * Finally, a backward transform is applied.
 * After this, data is again distributed according to CUFFT_XT_FORMAT_INPLACE, same as the input data.
 * 
 * Data is finally copied back to CPU and compared to the input data. They should be almost identical.
 */

void run_c2c_fwd_inv(size_t nx, size_t ny, size_t nz, std::complex<float>* cpu_data, int rank, int size, MPI_Comm comm) {

    cufftHandle plan = 0;
    cudaStream_t stream = nullptr;

    CUDA_CHECK(cudaStreamCreate(&stream));

    CUFFT_CHECK(cufftCreate(&plan));

    CUFFT_CHECK(cufftMpAttachComm(plan, CUFFT_COMM_MPI, &comm));

    CUFFT_CHECK(cufftSetStream(plan, stream));

    size_t workspace;
    CUFFT_CHECK(cufftMakePlan3d(plan, nx, ny, nz, CUFFT_C2C, &workspace));

    // Allocate memory, copy CPU data to GPU
    // Data is distributed as X-Slabs
    cudaLibXtDesc *desc;
    CUFFT_CHECK(cufftXtMalloc(plan, &desc, CUFFT_XT_FORMAT_INPLACE));
    CUFFT_CHECK(cufftXtMemcpy(plan, (void*)desc, (void*)cpu_data, CUFFT_COPY_HOST_TO_DEVICE));

    // Run C2C Fwd
    CUFFT_CHECK(cufftXtExecDescriptor(plan, desc, desc, CUFFT_FORWARD));

    // Data is now distributed as Y-Slabs
    // We run a kernel on the distributed data, using the BoxIterator's for convenience
    auto[begin_d, end_d] = BoxIterators(CUFFT_XT_FORMAT_INPLACE_SHUFFLED, CUFFT_C2C, 
                                        rank, size, nx, ny, nz, (cufftComplex*)desc->descriptor->data[0]);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 128;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz);
    
    // Run C2C Bwd
    CUFFT_CHECK(cufftXtExecDescriptor(plan, desc, desc, CUFFT_INVERSE));

    // Copy back and free
    // Data is distributed as X-Slabs again
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUFFT_CHECK(cufftXtMemcpy(plan, (void*)cpu_data, (void*)desc, CUFFT_COPY_DEVICE_TO_HOST));
    CUFFT_CHECK(cufftXtFree(desc));

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

    size_t nx = (argc >= 2 ? atoi(argv[1]) : 8*size);  // any value >= size is OK
    size_t ny = (argc >= 2 ? atoi(argv[1]) : 8*size);  // any value >= size is OK
    size_t nz = (argc >= 2 ? atoi(argv[1]) : 8*size);  // any value >= size is OK

    // We start with X-Slabs
    // Ranks 0 ... (nx % size - 1) have 1 more element in the X dimension
    // and every rank own all elements in the Y and Z dimensions.
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
    run_c2c_fwd_inv(nx, ny, nz, data.data(), rank, size, MPI_COMM_WORLD);

    // Compute error
    double error = compute_error(ref, data, buildBox3D(CUFFT_XT_FORMAT_INPLACE, CUFFT_C2C, rank, size, nx, ny, nz));

    MPI_Finalize();

    return assess_error(error);
}
