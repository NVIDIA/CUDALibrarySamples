#include <vector>
#include <cstdio>
#include <nvshmem.h>
#include <cufftMp.h>
#include <iostream>
#include <cassert>
#include <mpi.h>

#include "../common/error_checks.hpp"

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

    // Assume we have a 4x4 "global" world with the following data
    //     ------------> y
    //   | [0  1  2  3 ]
    //   | [4  5  6  7 ]
    //   | [8  9  10 11]
    //   | [12 13 14 15]
    // x v
    // 
    // Initially, data is distributed as follow
    // [0  1   | 2  3  ]
    // [4  5   | 6  7  ]
    // [8  9   | 10 11 ]
    // [12 13  | 14 15 ]
    // ^rank 0^ ^rank 1^
    // where every rank owns part of it, stored in row major format
    // 
    // and we wish to redistribute it as
    // rank 0 [0  1   2  3  ]
    //        [4  5   6  7  ]
    //        ---------------
    // rank 1 [8  9   10 11 ]
    //        [12 13  14 15 ]
    // where every rank should own part of it, stored in row major format
    //
    // To do so, we describe the data using boxes
    // where every box is described as low-high, where
    // low and high are the lower and upper corners, respectively
    //
    // - The "world" box is of size 4x4
    // - The input boxes are  [ (0,0)-(4,2), (0,2)--(4,4) ]
    // - The output boxes are [ (0,0)-(2,4), (2,0)--(4,4) ]
    //
    // Since our data is 2D, in the following, the first dimension is always 0-1
    Box3D in_box_0 = {
        {0, 0, 0}, {1, 4, 2}, {8, 2, 1}
    };
    Box3D in_box_1 = {
        {0, 0, 2}, {1, 4, 4}, {8, 2, 1}
    };
    Box3D out_box_0 = {
        {0, 0, 0}, {1, 2, 4}, {8, 4, 1}
    };
    Box3D out_box_1 = {
        {0, 2, 0}, {1, 4, 4}, {8, 4, 1}
    };

    std::vector<int> src_host, dst_host_expected;
    if(rank == 0) {
        src_host = {0, 1, 4, 5, 8,  9,  12, 13};
        dst_host_expected = {0, 1, 2, 3, 4, 5, 6, 7};
    } else {
        src_host = {2, 3, 6, 7, 10, 11, 14, 15};
        dst_host_expected = {8, 9, 10, 11, 12, 13, 14, 15};
    }

    cufftReshapeHandle handle;
    CUFFT_CHECK(cufftMpCreateReshape(&handle));
    CUFFT_CHECK(cufftMpAttachReshapeComm(handle, CUFFT_COMM_MPI, &comm));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    Box3D &in_box  = (rank == 0 ? in_box_0  : in_box_1);
    Box3D &out_box = (rank == 0 ? out_box_0 : out_box_1);
    CUFFT_CHECK(cufftMpMakeReshape(handle, sizeof(int), 3,
        in_box.lower,   in_box.upper,
        out_box.lower,  out_box.upper,
        in_box.strides, out_box.strides));

    size_t workspace;
    CUFFT_CHECK(cufftMpGetReshapeSize(handle, &workspace));
    assert(workspace == 0);
    
    // Move CPU data to GPU
    void *src = nvshmem_malloc(8 * sizeof(int));
    void *dst = nvshmem_malloc(8 * sizeof(int));
    CUDA_CHECK(cudaMemcpy(src, src_host.data(), 8 * sizeof(int), cudaMemcpyDefault));

    // Apply reshape
    CUFFT_CHECK(cufftMpExecReshapeAsync(handle, dst, src, nullptr, stream));

    // Move GPU data to CPU
    std::vector<int> dst_host(8);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(dst_host.data(), dst, 8 * sizeof(int), cudaMemcpyDefault));
    nvshmem_free(src);
    nvshmem_free(dst);
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
    for(int i = 0; i < dst_host_expected.size(); i++) {
        if(dst_host[i] != dst_host_expected[i]) errors++;
    }

    MPI_Finalize();

    if(errors == 0) {
        std::cout << "PASSED\n";
        return 0;
    } else {
        std::cout << "FAILED\n";
        return 1;
    }
}
