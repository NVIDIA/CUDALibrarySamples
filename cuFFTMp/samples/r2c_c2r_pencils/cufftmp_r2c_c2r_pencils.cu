#include <numeric>
#include <iostream>
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
 * This samples illustrates a basic use of cuFFTMp using custom data distributions
 * in the case of an R2C - C2R transform
 * 
 * It performs 
 * - forward transform
 * - printing and scaling of the entries
 * - inverse transform
 */

void run_r2c_c2r_pencils(size_t nx, size_t ny, size_t nz, float* cpu_data, Box3D box_real, Box3D box_complex, const int rank, const int size, MPI_Comm comm) {

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

    // Describe the data distribution
    // R2C plans only support CUFFT_XT_FORMAT_DISTRIBUTED_INPUT and always perform a CUFFT_FORWARD transform
    // C2R plans only support CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT ans always perform a CUFFT_INVERSE transform
    // So, in both, the "input" box should be the real box and the "output" box should be the complex box
    CUFFT_CHECK(cufftXtSetDistribution(plan_r2c, 3, box_real.lower, box_real.upper, box_complex.lower, box_complex.upper, box_real.strides, box_complex.strides));
    CUFFT_CHECK(cufftXtSetDistribution(plan_c2r, 3, box_real.lower, box_real.upper, box_complex.lower, box_complex.upper, box_real.strides, box_complex.strides));

    // Set the stream
    CUFFT_CHECK(cufftSetStream(plan_r2c, stream));
    CUFFT_CHECK(cufftSetStream(plan_c2r, stream));

    // Make the plan
    size_t workspace;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, nx, ny, nz, CUFFT_R2C, &workspace));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, nx, ny, nz, CUFFT_C2R, &workspace));

    // Allocate GPU memory, copy CPU data to GPU
    // Data is initially distributed according to CUFFT_XT_FORMAT_DISTRIBUTED_INPUT, i.e., box_real
    cudaLibXtDesc *desc;
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &desc, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT));
    CUFFT_CHECK(cufftXtMemcpy(plan_r2c, desc, cpu_data, CUFFT_COPY_HOST_TO_DEVICE));

    // Run R2C
    CUFFT_CHECK(cufftXtExecDescriptor(plan_r2c, desc, desc, CUFFT_FORWARD));

    // At this point, data is distributed according to CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT, i.e., box_complex
    // This applies an element-wise scaling function to the GPU data located in desc->descriptor->data[0]
    auto [begin_d, end_d] = BoxIterators(box_complex, (cufftComplex*)desc->descriptor->data[0]);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 128;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz);
    
    // Run C2R
    CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE));

    // Copy back to CPU and free
    // Data is again distributed according to CUFFT_XT_FORMAT_DISTRIBUTED_INPUT, i.e., box_real
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUFFT_CHECK(cufftXtMemcpy(plan_c2r, cpu_data, desc, CUFFT_COPY_DEVICE_TO_HOST));
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

    // Create pencils along Z for the input
    int nranks1d = std::sqrt(size);
    if(nranks1d * nranks1d != size) {
        std::cout << "The number of MPI ranks should be a perfect square\n";
        return 1;
    }

    // Define custom data distribution
    int64 nx               = 5;
    int64 ny               = 6;
    int64 nz               = 7;
    int64 nz_real          = nz;
    int64 nz_complex       = (nz/2+1);
    int64 nz_real_padded   = 2*nz_complex;

    // Describe the data distribution using boxes
    auto make_box = [](int64 lower[3], int64 upper[3], int64 strides[3]) {
        Box3D box;
        for(int i = 0; i < 3; i++) {
            box.lower[i] = lower[i];
            box.upper[i] = upper[i];
            box.strides[i] = strides[i];
        }
        return box;
    };

    auto displacement = [](int64 length, int rank, int size) {
        int ranks_cutoff = length % size;
        return (rank < ranks_cutoff ? rank * (length / size + 1) : ranks_cutoff * (length / size + 1) + (rank - ranks_cutoff) * (length / size));
    };

    std::vector<Box3D> boxes_real;
    std::vector<Box3D> boxes_complex;
    for(int i = 0; i < nranks1d; i++) {
        for(int j = 0; j < nranks1d; j++) {
            {
                // Input data are real pencils in X & Y, along Z
                // Strides are packed and in-place (i.e., real is padded)
                int64 lower[3]   = {displacement(nx, i,   nranks1d), displacement(ny, j,   nranks1d), 0};
                int64 upper[3]   = {displacement(nx, i+1, nranks1d), displacement(ny, j+1, nranks1d), nz_real};
                int64 strides[3] = {(upper[1]-lower[1])*nz_real_padded, nz_real_padded, 1};
                boxes_real.push_back(make_box(lower, upper, strides));
            }
            {
                // Output data are complex pencils in X & Z, along Y (picked arbitrarily)
                // Strides are packed
                // For best performances, the local dimension in the input (Z, here) and output (Y, here) should be different
                // to ensure cuFFTMp will only perform two communication phases.
                // If Z was also local in the output, cuFFTMp would perform three communication phases, decreasing performances.
                int64 lower[3]   = {displacement(nx, i,   nranks1d), 0,  displacement(nz_complex, j,   nranks1d)};
                int64 upper[3]   = {displacement(nx, i+1, nranks1d), ny, displacement(nz_complex, j+1, nranks1d)};
                int64 strides[3] = {(upper[1]-lower[1])*(upper[2]-lower[2]), (upper[2]-lower[2]), 1};
                boxes_complex.push_back(make_box(lower, upper, strides));
            }
        }
    }

    // Generate CPU data
    Box3D box_real = boxes_real[rank];
    std::vector<float> input_cpu_data((box_real.upper[0] - box_real.lower[0]) * box_real.strides[0]);
    generate_random(input_cpu_data, rank);
    auto ref = input_cpu_data;
    
    // Print input data
    auto[in_begin_h, in_end_h] = BoxIterators(box_real, input_cpu_data.data());
    for (auto it = in_begin_h; it != in_end_h; ++it) { 
        std::cout << "Input data, global 3D index [" << it.x() << "," << it.y() << "," << it.z() << "], local index " << it.i() << ", rank " << rank << " is " << *it << "\n";
    }

    // Compute a forward + normalization + inverse FFT
    run_r2c_c2r_pencils(nx, ny, nz, input_cpu_data.data(), boxes_real[rank], boxes_complex[rank], rank, size, MPI_COMM_WORLD);

    // Compute error
    double error = compute_error(ref, input_cpu_data, box_real);

    MPI_Finalize();

    return assess_error(error);
}
