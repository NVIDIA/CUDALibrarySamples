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
#include "../common.hpp"

// This example demonstrates how to use cuRANDDx thread-level operator to generate a sequence of random numbers
// using Philox generator with user-defined round number (default 10). 

// The generated log-normally distributed values are the same as the results using cuRAND host API 
// with CURAND_ORDERING_PSEUDO_LEGACY ordering. 
// The ordering is that using subsequences different sequences each thread generates 4 32-bit random values, 
// and each four values from one sequence are followed by four values from next sequence.
// https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types Philox_4x32_10 pseudorandom generator

constexpr unsigned int subsequences = 65536;

template<class RNG, typename DataType>
__global__ void generate_kernel(double4*                        d_out,
                                const unsigned long long        seed,
                                const typename RNG::offset_type offset,
                                const size_t                    size,
                                const DataType                  mean,
                                const DataType                  stddev) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size / 4)
        return;

    curanddx::log_normal<DataType, curanddx::box_muller> dist(mean, stddev);

    RNG rng(seed, ((offset + i) % subsequences), ((offset + i) / subsequences)); // seed, subsequence, offset

    d_out[i] = dist.generate4(rng);
}

template<unsigned int Arch>
int philox_thread_api() {
    using RNG      = decltype(curanddx::Generator<curanddx::philox4_32>() + curanddx::PhiloxRounds<10>() +
                         curanddx::SM<Arch>() + curanddx::Thread());
    using DataType = double;

    // Allocate output memory
    DataType*    d_out;
    const size_t size = 5000;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_out, size * sizeof(DataType)));

    const unsigned long long        seed   = 1234ULL;
    const typename RNG::offset_type offset = 1ULL;

    const DataType mean   = 1;
    const DataType stddev = 0.5;

    // Invokes kernel
    const unsigned int block_dim = 256;
    const unsigned int grid_size = (size / 4 + block_dim - 1) / block_dim;
    generate_kernel<RNG, DataType><<<grid_size, block_dim, 0>>>((double4*)d_out, seed, offset, size, mean, stddev);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<DataType> h_out(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, size * sizeof(DataType), cudaMemcpyDeviceToHost));

    // cuRAND host API
    curandGenerator_t gen_curand;
    DataType*         d_ref;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_ref, size * sizeof(DataType)));

    CURAND_CHECK_AND_EXIT(curandCreateGenerator(&gen_curand, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CHECK_AND_EXIT(curandSetPseudoRandomGeneratorSeed(gen_curand, seed));
    CURAND_CHECK_AND_EXIT(curandSetGeneratorOffset(gen_curand, offset * 4));
    CURAND_CHECK_AND_EXIT(curandSetGeneratorOrdering(gen_curand, CURAND_ORDERING_PSEUDO_LEGACY));

    CURAND_CHECK_AND_EXIT(curandGenerateLogNormalDouble(gen_curand, d_ref, size, mean, stddev));

    std::vector<DataType> h_ref(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_ref.data(), d_ref, size * sizeof(DataType), cudaMemcpyDeviceToHost));

    CURAND_CHECK_AND_EXIT(curandDestroyGenerator(gen_curand));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ref));

    // Compare Results
    if (h_out == h_ref) {
        std::cout << "SUCCESS: Same sequence is generated with cuRANDDx and cuRAND Host API using LEGACY ordering.\n";
        return 0;
    } else {
        int count {0};
        for (auto i = 0U; i < size; i++) {
            if (h_out[i] != h_ref[i] && count++ < 10) {
                std::cout << "array_curanddx[" << i << "] = " << h_out[i] << " array_curand[" << i << "] = " << h_ref[i]
                          << std::endl;
                ;
            }
        }
        std::cout << "FAILED: Different sequence is generated with cuRANDDx and cuRAND Host API using LEGACY "
                     "ordering.\n";
        return 1;
    }
}

template<unsigned int Arch>
struct philox_thread_api_functor {
    int operator()() { return philox_thread_api<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<philox_thread_api_functor>();
}
