#pragma once

#include <cuda_runtime.h>
#include <cublasdx.hpp>

#include <iostream>
#include <sstream>

template<class Tensor>
void print_slices(const Tensor& T, std::string msg) {
    std::cout << msg << std::endl;

    for (int s = 0; s < cute::size<0>(T); ++s) {
        printf("------------------------------ s = %d\n", s);
        for (int m = 0; m < cute::size<1>(T); ++m) {
            for (int n = 0; n < cute::size<2>(T); ++n) {
                printf("%d ", T(cute::make_coord(s, m, n)));
            }
            printf("\n");
        }
    }
    printf("\n");
}

void print_device_properties() {
    cudaDeviceProp prop;

    int device_count = 0;
    CUDA_CHECK_AND_EXIT(cudaGetDeviceCount(&device_count));

    std::stringstream ss;
    ss << "Number of CUDA devices: " << device_count << std::endl << std::endl;

    for (auto device_id = 0; device_id < device_count; device_id++) {
        CUDA_CHECK_AND_EXIT(cudaGetDeviceProperties(&prop, device_id));

        ss << "Device " << device_id << ": " << prop.name << std::endl;
        ss << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        ss << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB" << std::endl;
        ss << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        ss << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        ss << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        ss << "  Warp size: " << prop.warpSize << std::endl;
        ss << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        ss << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        ss << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        ss << std::endl;
    }

    std::cout << ss.str();
}
