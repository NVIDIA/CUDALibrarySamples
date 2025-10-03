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

#include <iostream>
#include <vector>

#include <curanddx.hpp>
#include "../common.hpp"

#ifdef CURANDDX_EXAMPLE_NVPL_RAND_AVAILABLE
#    include <nvpl_rand.h>
#    define NVPL_RAND_CHECK(x)                                  \
        do {                                                    \
            if ((x) != NVPL_RAND_STATUS_SUCCESS) {              \
                printf("Error at %s:%d\n", __FILE__, __LINE__); \
                _Exit(EXIT_FAILURE);                            \
            }                                                   \
        } while (0)
#endif

// This example demonstrates how to use the default PCG generator and cuRANDDx Thread-level operator to generate a sequence of random 32-bit 
// numbers and compare with the results generated using NVPL RAND, if available, with STRICT ordering

template<class RNG, typename data_type>
__global__ void generate_kernel(data_type*                      d_out,
                                const unsigned long long        seed,
                                const typename RNG::offset_type offset,
                                const size_t                    size) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= size)
        return;

    curanddx::uniform_bits<data_type> dist;

    // compare with NVPL RAND PCG strict ordering
    RNG rng(seed, 0, offset + tid);

    d_out[tid] = dist.generate(rng);
}

template<unsigned int Arch>
int simple_pcg_thread_api() {
    using RNG = decltype(curanddx::Generator<curanddx::pcg>() + curanddx::SM<Arch>() + curanddx::Thread());

    using DataType = typename RNG::bitgenerator_result_type;

    // Allocate output memory
    DataType*    d_out;
    const size_t size = 5000;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_out, size * sizeof(DataType)));

    unsigned long long seed   = 1234ULL;
    unsigned long long offset = 1ULL;

    // Invokes kernel
    const unsigned int block_dim = 256;
    const unsigned int grid_size = (size + block_dim - 1) / block_dim;
    
    generate_kernel<RNG, DataType><<<grid_size, block_dim, 0>>>(d_out, seed, offset, size);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<DataType> h_out(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, size * sizeof(DataType), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));

#ifdef CURANDDX_EXAMPLE_NVPL_RAND_AVAILABLE
    // nvpl RAND
    std::vector<DataType> h_ref(size);
    nvplRandGenerator_t   gen;
    const auto            generator_type = NVPL_RAND_RNG_PSEUDO_PCG;

    nvplRandMTCreateGeneratorDefault(&gen, generator_type);
    NVPL_RAND_CHECK(nvplRandSetPseudoRandomGeneratorSeed(gen, seed));
    NVPL_RAND_CHECK(nvplRandSetGeneratorOffset(gen, offset));
    NVPL_RAND_CHECK(nvplRandMTSetGeneratorOrdering(gen, NVPL_RAND_ORDERING_STRICT));

    // Generate
    NVPL_RAND_CHECK(nvplRandGenerate(gen, h_ref.data(), size));
    NVPL_RAND_CHECK(nvplRandDestroyGenerator(gen));

    // Compare Results
    if (h_out == h_ref) {
        std::cout << "SUCCESS: Same sequence is generated with cuRANDDx and nvpl RAND API using STRICT ordering.\n";
    } else {
        int count {0};
        for (auto i = 0U; i < size; i++) {
            if (h_out[i] != h_ref[i] && count < 10) {
                printf("array_curanddx[%u] = %u, array_nvplrand[%u] = %u \n", i, h_out[i], i, h_ref[i]);
                count++;
            }
        }
        std::cout << "FAILED: Different sequence is generated with cuRANDDx and NVPL RAND Host API using LEGACY "
                     "ordering.\n";
        return 1;
    }

    // compute hash to be used if nvpl rand is not available
    unsigned int xor_nvplrand = 0x0;
    for (auto i = 0U; i < size; i++) {
        xor_nvplrand ^= h_ref[i];
    }
    std::cout << "NVPL RAND reference xor output is " << std::hex << xor_nvplrand << std::endl;
    return 0;

#else
    // Compare hash
    unsigned int xor_curand = 0x0;
    for (auto i = 0U; i < size; i++) {
        xor_curand ^= h_out[i];
    }
    if (xor_curand == 0xaa706742) {
        std::cout << "Compared to the hash value: Same sequence is generated with NVPL RAND and cuRANDDx generator "
                     "using STRICT ordering.\n";
        std::cout << "SUCCESS \n";
        return 0;
    } else {
        std::cout
            << "FAILED: different sequence is generated with NVPL RAND and cuRANDDx generator using STRICT ordering.\n";
        return -1;
    }
#endif
}

template<unsigned int Arch>
struct simple_pcg_thread_api_functor {
    int operator()() { return simple_pcg_thread_api<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<simple_pcg_thread_api_functor>();
}