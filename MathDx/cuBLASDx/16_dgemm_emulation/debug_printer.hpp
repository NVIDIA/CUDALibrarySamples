/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    int            sm_clock, mem_clock;

    int device_count = 0;
    CUDA_CHECK_AND_EXIT(cudaGetDeviceCount(&device_count));

    std::stringstream ss;
    ss << "Number of CUDA devices: " << device_count << std::endl << std::endl;

    for (auto device_id = 0; device_id < device_count; device_id++) {
        CUDA_CHECK_AND_EXIT(cudaGetDeviceProperties(&prop, device_id));
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&sm_clock, cudaDevAttrClockRate, device_id));
        CUDA_CHECK_AND_EXIT(cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, device_id));

        ss << "Device " << device_id << ": " << prop.name << std::endl;
        ss << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        ss << "  Total global memory: " << (prop.totalGlobalMem >> 20) << " MB" << std::endl;
        ss << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        ss << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        ss << "  Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        ss << "  Warp size: " << prop.warpSize << std::endl;

        ss << "  Clock Rate: " << sm_clock / 1000.f << " MHz" << std::endl;
        ss << "  Memory Clock Rate: " << mem_clock / 1000.f << " MHz" << std::endl;

        ss << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        ss << std::endl;
    }

    std::cout << ss.str();
}
