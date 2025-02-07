// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <iostream>
#include <vector>

#include <curanddx.hpp>
#include "common.hpp"

// This example demonstrates how to launch two kernels, one for initializing the states, one for generating normally distribute 
// random numbers with Box-Muller method, using XORWOW generator. 
// 
// cuRANDDx functions used in the example kernels:
// (1) state initialization using init() function, or the constructor of the RNG object
// (2) generate2() function as Box-Muller method is selected for normal distribution
//
// The generated RNs are the same as the results using cuRAND host API with CURAND_ORDERING_PSEUDO_LEGACY ordering
// The ordering requires that the result at offset n is from position  
// (n mod 4096) * 2^67 + (i/4096) in the original XORWOW sequence
// https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types 

constexpr unsigned int subsequences = 4096;

template<class RNG>
__global__ void init_kernel(RNG* states, const unsigned long long seed, const typename RNG::offset_type offset) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    RNG rng;
    rng.init(seed, ((offset + tid) % subsequences), ((offset + tid) / subsequences)); // seed, subsequence, offset

    // Alternatively one can use the constructor directly
    //RNG rng(seed, ((offset + tid) % subsequences), ((offset + tid) / subsequences)); 
    
    states[tid] = rng;
}

template<class RNG, typename DataType>
__global__ void generate_kernel(RNG* states, float2* d_out, const size_t size, const DataType mean, const DataType stddev) {
    int       tid     = blockDim.x * blockIdx.x + threadIdx.x;
    const int threads = blockDim.x * gridDim.x;

    curanddx::normal<DataType, curanddx::box_muller> dist(mean, stddev);

    RNG rng = states[tid];

    for (auto idx = tid; idx < size / 2; idx += threads) {
        d_out[idx] = dist.generate2(rng);
    }

    // Each thread updates "states" in global memory so "states" can be used for later kernels
    states[tid] = rng;
}


template<unsigned int Arch>
int xorwow_init_and_generate_thread_api() {
    using RNG      = decltype(curanddx::Generator<curanddx::xorwow>() + curanddx::SM<Arch>() + curanddx::Thread());
    using DataType = float;

    // Allocate output memory
    DataType*    d_out;
    const size_t size = 50000;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_out, size * sizeof(DataType)));

    const unsigned long long seed   = 1234ULL;
    const typename RNG::offset_type offset = 2ULL;

    const unsigned int block_dim = 256;
    const unsigned int grid_size = 16;

    const DataType mean = 0; 
    const DataType stddev = 2;

    // Allocate an array of states
    RNG* states;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&states, block_dim * grid_size * sizeof(RNG)));

    // Invoke the init kernel first to set up the states
    init_kernel<RNG><<<grid_size, block_dim>>>(states, seed, offset);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    // Invoke the generate kernel to generate RNs and update the states
    generate_kernel<RNG, DataType><<<grid_size, block_dim>>>(states, (float2*)d_out, size, mean, stddev);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<DataType> h_out(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, size * sizeof(DataType), cudaMemcpyDeviceToHost));

    // cuRAND host API
    curandGenerator_t gen_curand;
    DataType*         d_ref;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_ref, size * sizeof(DataType)));

    CURAND_CHECK_AND_EXIT(curandCreateGenerator(&gen_curand, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CHECK_AND_EXIT(curandSetPseudoRandomGeneratorSeed(gen_curand, seed));
    CURAND_CHECK_AND_EXIT(curandSetGeneratorOffset(gen_curand, offset));
    CURAND_CHECK_AND_EXIT(curandSetGeneratorOrdering(gen_curand, CURAND_ORDERING_PSEUDO_LEGACY));

    CURAND_CHECK_AND_EXIT(curandGenerateNormal(gen_curand, d_ref, size, mean, stddev));

    std::vector<DataType> h_ref(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_ref.data(), d_ref, size * sizeof(DataType), cudaMemcpyDeviceToHost));

    CURAND_CHECK_AND_EXIT(curandDestroyGenerator(gen_curand));
    CUDA_CHECK_AND_EXIT(cudaFree(states));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ref));

    // Compare Results between cuRAND host API and cuRANDDx
    if (h_out == h_ref) {
        std::cout << "SUCCESS: Same sequence is generated with cuRANDDx and cuRAND Host API using LEGACY ordering.\n";
        return 0;
    } else {
        unsigned int count {0};
        for (auto i = 0U; i < size; i++) {
            if (h_out[i] != h_ref[i] && count < 10) {
                std::cout << "array_curanddx[" << i << "] = " << h_out[i] << " array_curand[" << i << "] = " << h_ref[i]
                          << std::endl;
                count++;
            }
        }
        std::cout << "FAILED: Different sequence is generated with cuRANDDx and cuRAND Host API using LEGACY "
                     "ordering.\n";
        return 1;
    }
}

template<unsigned int Arch>
struct xorwow_init_and_generate_thread_api_functor {
    int operator()() { return xorwow_init_and_generate_thread_api<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<xorwow_init_and_generate_thread_api_functor>();
}
