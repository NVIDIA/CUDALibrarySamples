#include <numeric>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <iostream>
#include <stdio.h>
#include <cufftMp.h>
#include <mpi.h>

#include "../iterators/box_iterator.hpp"
#include "../common/generate_random.hpp"
#include "../common/scaling.cuh"
#include "../common/error_checks.hpp"

/**
 * This sample illustrates the use of cufftXtSetDistribution
 * to support arbitrary user data distributions.
 * 
 * It performs
 * - Forward FFT
 * - Element wise kernel
 * - Inverse FFT
 * 
 * It also print the input and output data in their respective data distribution, illustrating
 * the user of the BoxIterator's.
 */

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    int ndevices;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank % ndevices));
    printf("Hello from rank %d/%d using GPU %d\n", rank, size, rank % ndevices);

    int nranks1d = std::sqrt(size);
    if(nranks1d * nranks1d != size) {
        std::cout << "The number of MPI ranks should be a perfect square\n";
        return 1;
    }

    // Define custom data distribution
    int64 N = 2;
    int64 nx = N * nranks1d;
    int64 ny = N * nranks1d;
    int64 nz = N * nranks1d;

    // Describing a data distribution is done using 3D "boxes"
    // 
    // A 3D box is defined as { {x_start, y_start, z_start}, {x_end, y_end, z_end}, {x_strides, y_strides, z_strides} }
    // where
    // - {x/y/z}_{start/end} are the lower and upper corner of the boxes relatived to the global 3D box (of size nx * ny * nz)
    // - {x/y/z}_strides are the local strides
    std::vector<Box3D> input_boxes;
    std::vector<Box3D> output_boxes;
    for(int i = 0; i < nranks1d; i++) {
        for(int j = 0; j < nranks1d; j++) {
            // Input data are pencils in X & Y, along Z
            input_boxes.push_back({{i*N, j*N, 0}, {(i+1)*N, (j+1)*N, nz}, {N*nz, nz, 1}});
            // Output data are pencils in X & Z, along Y
            output_boxes.push_back({{i*N, 0, j*N}, {(i+1)*N, ny, (j+1)*N}, {N*ny, N, 1}}); 
        }
    }

    // Generate CPU data
    Box3D in = input_boxes[rank];
    std::vector<std::complex<float>> input_cpu_data((in.upper[0] - in.lower[0]) * in.strides[0]);
    generate_random(input_cpu_data, rank);
    
    auto[in_begin_h, in_end_h] = BoxIterators(in, input_cpu_data.data());
    for (auto it = in_begin_h; it != in_end_h; ++it) { 
        std::cout << "input data, global 3D index [" << it.x() << "," << it.y() << "," << it.z() << "], local index " << it.i() << ", rank " << rank << " is (" << it->real() << "," << it->imag() << ")\n";
    }
    
    // Create a stream
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create and initialize a cuFFT plan
    cufftHandle  plan = 0;
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftMpAttachComm(plan, CUFFT_COMM_MPI, &mpi_comm));
    CUFFT_CHECK(cufftSetStream(plan, stream));
    CUFFT_CHECK(cufftXtSetDistribution(plan, 3, 
        input_boxes[rank].lower,   input_boxes[rank].upper, 
        output_boxes[rank].lower,  output_boxes[rank].upper, 
        input_boxes[rank].strides, output_boxes[rank].strides));

    // Make the cuFFT plan
    size_t scratch;
    CUFFT_CHECK(cufftMakePlan3d(plan, nx, ny, nz, CUFFT_C2C, &scratch));

    // Copy CPU data to GPU
    cudaLibXtDesc *desc;
    CUFFT_CHECK(cufftXtMalloc(plan, &desc, CUFFT_XT_FORMAT_DISTRIBUTED_INPUT));
    CUFFT_CHECK(cufftXtMemcpy(plan, (void*)desc, (void*)input_cpu_data.data(), CUFFT_COPY_HOST_TO_DEVICE));

    // Execute the cuFFT plan
    CUFFT_CHECK(cufftXtExecDescriptor(plan, desc, desc, CUFFT_FORWARD));

    // Manipulate GPU data using a user kernel in the same stream as cuFFT
    Box3D out = output_boxes[rank];
    auto[out_begin_d, out_end_d] = BoxIterators(out, (cufftComplex*)desc->descriptor->data[0]);
    const size_t num_elements = std::distance(out_begin_d, out_end_d);
    const size_t num_threads  = 128;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;
    scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(out_begin_d, out_end_d, rank, size, nx, ny, nz);

    // Copy GPU data back to CPU
    std::vector<std::complex<float>> output_cpu_data((out.upper[0] - out.lower[0]) * out.strides[0], {-1000,-1000});
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUFFT_CHECK(cufftXtMemcpy(plan, (void*)output_cpu_data.data(), (void*)desc, CUFFT_COPY_DEVICE_TO_HOST));

    // Print the CPU data
    auto[out_begin_h, out_end_h] = BoxIterators(out, output_cpu_data.data());
    for (auto it = out_begin_h; it != out_end_h; ++it) { 
        std::cout << "intermediate data, global 3D index [" << it.x() << "," << it.y() << "," << it.z() << "], local index " << it.i() << ", rank " << rank << " is (" << it->real() << "," << it->imag() << ")\n";
    }

    // Run inverse plan
    CUFFT_CHECK(cufftXtExecDescriptor(plan, desc, desc, CUFFT_INVERSE));

    // Copy to CPU
    output_cpu_data.resize((in.upper[0] - in.lower[0]) * in.strides[0]);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUFFT_CHECK(cufftXtMemcpy(plan, (void*)output_cpu_data.data(), (void*)desc, CUFFT_COPY_DEVICE_TO_HOST));

    // Cleanup
    CUFFT_CHECK(cufftXtFree(desc));
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Compute error
    double error = compute_error(input_cpu_data, output_cpu_data, in);

    MPI_Finalize();

    return assess_error(error);
}
