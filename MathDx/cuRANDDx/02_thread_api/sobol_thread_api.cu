// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <iostream>
#include <vector>
#include <cassert>

#include <curanddx.hpp>
#include "../common.hpp"

// This example demonstrates how to use quasi-random generator scrambled sobol64 and cuRANDDx thread-level operator
// to generate a sequence of random numbers and compare with the results generated using cuRAND host API

template<class RNG, typename DataType>
__global__ void generate_kernel(DataType*                                 d_out,
                                typename RNG::direction_vector_type*      direction_vectors,
                                const typename RNG::scrambled_const_type* scrambled_consts,
                                const typename RNG::offset_type           offset,
                                const unsigned int                        num_dims,
                                const size_t                              size,
                                DataType                                  input1,
                                DataType                                  input2) {
    int       tid     = blockDim.x * blockIdx.x + threadIdx.x;
    const int threads = blockDim.x * gridDim.x;

    curanddx::uniform<float> my_uniform(input1, input2);

    // When generating n results in m dimensions, the output will consist of n//m results from dimension 0, followed by
    // another n/m results from dimension 1, etc.
    const int size_per_dim = size / num_dims;

    for (auto idx = tid; idx < size; idx += threads) {
        // for each dimension, tid0 generates the first number, tid1 the second number etc.
        const int idex_per_dim = idx % size_per_dim;
        const int dim          = idx / size_per_dim;
        RNG       rng(dim, direction_vectors, offset + idex_per_dim, scrambled_consts);


        d_out[idx] = my_uniform.generate(rng);
    }
}

template<unsigned int Arch>
int sobol_thread_api() {
    using RNG =
        decltype(curanddx::Generator<curanddx::scrambled_sobol64>() + curanddx::SM<Arch>() + curanddx::Thread());

    using DataType = float;

    // Allocate output memory
    DataType*    d_out;
    const size_t size = 5000;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_out, size * sizeof(DataType)));

    // size has to be evenly divisible by dimensions
    const unsigned int sobol_dims = 200;
    assert(size % sobol_dims == 0);

    const typename RNG::offset_type offset = 10ULL;
    const DataType                  min_v  = -1;
    const DataType                  max_v  = 1;

    // Step 1: Get pointers to the host direction vectors using cuRAND API
    using direction_vector_type = typename RNG::direction_vector_type;
    direction_vector_type* h_direction_vectors;
    CURAND_CHECK_AND_EXIT(
        curandGetDirectionVectors64(&h_direction_vectors, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6));

    // Step 2: Allocate memory and copy one direction vector per dimension to the device
    direction_vector_type* direction_vectors_gmem_ptr;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&(direction_vectors_gmem_ptr), sobol_dims * sizeof(direction_vector_type)));

    CUDA_CHECK_AND_EXIT(cudaMemcpy(direction_vectors_gmem_ptr,
                                   h_direction_vectors,
                                   sobol_dims * sizeof(direction_vector_type),
                                   cudaMemcpyHostToDevice));

    // Step 3: Get pointers to the host scrambled constants
    using scrambled_const_type = typename RNG::scrambled_const_type;
    scrambled_const_type* h_scrambled_consts;
    scrambled_const_type* scrambled_consts_gmem_ptr;
    CURAND_CHECK_AND_EXIT(curandGetScrambleConstants64(&h_scrambled_consts));

    // Step 4: Allocate memory and copy one scrambled const per dimension to the device
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&(scrambled_consts_gmem_ptr), sobol_dims * sizeof(scrambled_const_type)));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(scrambled_consts_gmem_ptr,
                                   h_scrambled_consts,
                                   sobol_dims * sizeof(scrambled_const_type),
                                   cudaMemcpyHostToDevice));

    // Invokes a kernel which uses cuRANDDx functions
    const unsigned int block_dim = 256;
    const unsigned int grid_size = 16;
    generate_kernel<RNG, DataType><<<grid_size, block_dim, 0>>>(
        d_out, direction_vectors_gmem_ptr, scrambled_consts_gmem_ptr, offset, sobol_dims, size, min_v, max_v);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<DataType> h_out(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out.data(), d_out, size * sizeof(DataType), cudaMemcpyDeviceToHost));

    // cuRAND host API
    curandGenerator_t gen_curand;
    DataType*         d_ref;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_ref, size * sizeof(DataType)));

    CURAND_CHECK_AND_EXIT(curandCreateGenerator(&gen_curand, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64));
    CURAND_CHECK_AND_EXIT(curandSetGeneratorOffset(gen_curand, offset));
    CURAND_CHECK_AND_EXIT(curandSetQuasiRandomGeneratorDimensions(gen_curand, sobol_dims));
    CURAND_CHECK_AND_EXIT(curandSetGeneratorOrdering(gen_curand, CURAND_ORDERING_QUASI_DEFAULT));

    CURAND_CHECK_AND_EXIT(curandGenerateUniform(gen_curand, d_ref, size));

    std::vector<DataType> h_ref(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_ref.data(), d_ref, size * sizeof(DataType), cudaMemcpyDeviceToHost));

    // scale the reference data to range(min_v, max_v)
    for( DataType &v : h_ref )  v = min_v + v * (max_v - min_v);

    CURAND_CHECK_AND_EXIT(curandDestroyGenerator(gen_curand));
    CUDA_CHECK_AND_EXIT(cudaFree(scrambled_consts_gmem_ptr));
    CUDA_CHECK_AND_EXIT(cudaFree(direction_vectors_gmem_ptr));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ref));

    // Compare Results
    if (h_out == h_ref) {
        std::cout
            << "SUCCESS: Same sequence is generated with cuRANDDx and cuRAND Host API using QUASI_DEFAULT ordering.\n";
        return 0;
    } else {
        for (auto i = 0U; i < 10; i++) {
            std::cout << "array_curanddx[" << i << "] = " << h_out[i] << " array_curand[" << i << "] = " << h_ref[i]
                      << std::endl;
        }
        std::cout << "FAILED: Different sequence is generated with cuRANDDx and cuRAND Host API using LEGACY "
                     "ordering.\n";
        return 1;
    }
}

template<unsigned int Arch>
struct sobol_thread_api_functor {
    int operator()() { return sobol_thread_api<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<sobol_thread_api_functor>();
}
