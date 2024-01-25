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
 * in the case of an R2C - C2R transform using slab decomposition.
 * This mimics the cuFFTMp use case in GROMACS 
 * (https://www.gromacs.org/, 
 *  https://manual.gromacs.org/nightly/install-guide/index.html#using-cufftmp, 
 *  https://github.com/gromacs/gromacs/blob/main/src/gromacs/fft/gpu_3dfft_cufftmp.cpp#L110)
 * 
 * It performs 
 * - forward transform
 * - scaling of the entries
 * - inverse transform
 * 
 * see README.md for more details.
 */

void run_r2c_c2r_slabs(size_t nx, size_t ny, size_t nz, const int cycles, const int warmup_runs, float* cpu_data, Box3D box_real, Box3D box_complex, const int rank, const int size, MPI_Comm comm) {

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

    // Create three events for each cycle to store the timings for R2C, scaling and C2R respectively.
    std::vector<cudaEvent_t> events(3*cycles + 1, nullptr);
    for (auto &event: events) {
        CUDA_CHECK(cudaEventCreate(&event));
    }

    // Prepare the data structure needed for the second kernel (scaling kernel).
    auto [begin_d, end_d] = BoxIterators(box_complex, (cufftComplex*)desc->descriptor->data[0]);
    const size_t num_elements = std::distance(begin_d, end_d);
    const size_t num_threads  = 128;
    const size_t num_blocks   = (num_elements + num_threads - 1) / num_threads;

    // Do some warm-up runs for profiling purposes
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    for (int warmup_run = 0; warmup_run < warmup_runs; ++warmup_run) {
        CUFFT_CHECK(cufftXtExecDescriptor(plan_r2c, desc, desc, CUFFT_FORWARD));
        scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz, false);
        CUDA_CHECK(cudaGetLastError());
        CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE));
    }
    
    // For timing purposes, sync all devices before looping
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    double CPU_start = 0, CPU_end = 0; 
    CPU_start = MPI_Wtime(); 
    for (int cycle = 0; cycle < cycles; ++cycle) {

        // First step: R2C transform
        CUDA_CHECK(cudaEventRecord(events[cycle*3], stream));
        CUFFT_CHECK(cufftXtExecDescriptor(plan_r2c, desc, desc, CUFFT_FORWARD));

        // Second step: Scaling kernel (see common/scaling.cuh)
        // At this point, data is distributed according to CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT, i.e., box_complex
        // This applies an element-wise scaling function to the GPU data located in desc->descriptor->data[0]
        // An additional flag is passed to the scaling function to mute prints for performance reasons.
        CUDA_CHECK(cudaEventRecord(events[cycle*3+1], stream));
        scaling_kernel<<<num_blocks, num_threads, 0, stream>>>(begin_d, end_d, rank, size, nx, ny, nz, false);
        CUDA_CHECK(cudaGetLastError());
        
        // Third step: C2R trasnform
        CUDA_CHECK(cudaEventRecord(events[cycle*3+2], stream));
        CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE));
    }
    // Record the last event.
    CUDA_CHECK(cudaEventRecord(events[cycles*3], stream));
    
    // Copy back to CPU 
    // Data is again distributed according to CUFFT_XT_FORMAT_DISTRIBUTED_INPUT, i.e., box_real
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    CPU_end = MPI_Wtime();
    CUFFT_CHECK(cufftXtMemcpy(plan_c2r, cpu_data, desc, CUFFT_COPY_DEVICE_TO_HOST));

    // Compute timings and report
    float time_r2c = 0, time_scaling = 0, time_c2r = 0;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        float time_tmp = 0;
        CUDA_CHECK(cudaEventElapsedTime(&time_tmp, events[cycle*3], events[cycle*3+1]));
        time_r2c += time_tmp;
        CUDA_CHECK(cudaEventElapsedTime(&time_tmp, events[cycle*3+1], events[cycle*3+2]));
        time_scaling += time_tmp;
        CUDA_CHECK(cudaEventElapsedTime(&time_tmp, events[cycle*3+2], events[cycle*3+3]));
        time_c2r += time_tmp;
    }

    if (rank == 0) {
        std::cout << "For a total of " << cycles << " cycles" << std::endl;
        std::cout << "Average time for 1 cycle (GPU timer): " << (time_r2c + time_scaling + time_c2r) / cycles << " ms" << std::endl;
        std::cout << "Average time for 1 cycle (CPU timer): " << (CPU_end - CPU_start) / cycles * 1e3 << " ms" << std::endl;
        std::cout << "Average time for each individual step (GPU timer):" << std::endl;
        std::cout << "C2R: " << time_r2c / cycles << " ms" << std::endl;
        std::cout << "Scaling: " << time_scaling / cycles << " ms" << std::endl;
        std::cout << "R2C: " << time_c2r / cycles << " ms" << std::endl;
    }

    // Free/Destroy 
    CUFFT_CHECK(cufftXtFree(desc));

    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));

    CUDA_CHECK(cudaStreamDestroy(stream));

    for (auto &event: events) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
};

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ndevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndevices));
    CUDA_CHECK(cudaSetDevice(rank % ndevices));
    printf("Hello from rank %d/%d using GPU %d\n", rank, size, rank % ndevices);

    // compute multiple cycles for timing purposes.
    int cycles = 10; 
    // use 1 cycle as warm up runs
    int warmup_runs = 1;

    // Define FFT sizes (160 x 160 x 160 in 3D).
    int64 nx               = 160;
    int64 ny               = 160;
    int64 nz               = 160;
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

    // Create boxes for custom data distribution.
    Box3D box_real, box_complex;
    {
        // Input data are X-slabs. 
        // Strides are packed and in-place (i.e., real is padded)
        int64 lower[3]   = {nx / size * (rank),   0,  0};
        int64 upper[3]   = {nx / size * (rank+1), ny, nz_real};
        int64 strides[3] = {(upper[1]-lower[1])*nz_real_padded, nz_real_padded, 1};
        box_real = make_box(lower, upper, strides);
    }
    {
        // Output data are Y-slabs.
        // Strides are packed
        int64 lower[3]   = {0,  ny / size * (rank),   0};
        int64 upper[3]   = {nx, ny / size * (rank+1), nz_complex};
        int64 strides[3] = {(upper[1]-lower[1])*(upper[2]-lower[2]), (upper[2]-lower[2]), 1};
        box_complex = make_box(lower, upper, strides);
    }

    // Generate CPU data
    std::vector<float> input_cpu_data((box_real.upper[0] - box_real.lower[0]) * box_real.strides[0]);
    generate_random(input_cpu_data, rank);

    // Store CPU data as reference. the input_cpu_data is used as a buffer later to hold output data.
    auto ref = input_cpu_data;

    // Compute a forward + normalization + inverse FFT
    run_r2c_c2r_slabs(nx, ny, nz, cycles, warmup_runs, input_cpu_data.data(), box_real, box_complex, rank, size, MPI_COMM_WORLD);

    // Compute error before exiting. require an MPI_allreduce to collect error on all ranks
    double error = compute_error(ref, input_cpu_data, box_real, MPI_COMM_WORLD);

    // Tolerance is relaxed as the error grows at O(cycles + warmup_cycles).
    int code = assess_error(error, rank, (cycles + warmup_runs)*1e-6);

    MPI_Finalize();

    return code;
}
