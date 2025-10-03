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

// cuRAND uses a single 32-bit random value to generate one double-precision uniform/normal/lognormal distributed value
// A more precise way is using two 32-bit values to generate one double-precision value
#ifndef CURANDDX_MRG_DOUBLE_DISTRIBUTION_CURAND_COMPATIBLE
#    define CURANDDX_MRG_DOUBLE_DISTRIBUTION_CURAND_COMPATIBLE
#endif

#include <curanddx.hpp>
#include "../common.hpp"

// This example demonstrates how to use cuRANDDx thread-level operator to generate:
// (1) a sequence of uniformly-distributed FP64 numbers,
// (2) a sequence of normally-distributed FP32 numbers.
// This can be done either in a single kernel or multiple kernels depending on the application workflow.
// 
// The generated RNs are the same as the results using cuRAND host API with CURAND_ORDERING_PSEUDO_LEGACY ordering. 
// The ordering requires that the result at offset n is from position 
// (n mod 4096) * 2^76 + (n/4096) in the original MRG32k3a sequence
// https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types MRG32k3a pseudorandom generator

constexpr unsigned int subsequences = 4096;

template<class RNG>
__global__ void generate_kernel(double*                         d_out,
                                float2*                         d_out_normal,
                                const unsigned long long        seed,
                                const typename RNG::offset_type offset,
                                const size_t                    size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    curanddx::uniform<double>                     my_uniform; // default min = 0, max = 1
    curanddx::normal<float, curanddx::box_muller> my_normal;  // default mean = 0, stddev = 1

    RNG rng(seed, (offset + tid) % subsequences, (offset + tid) / subsequences); // seed, subsequence, offset
    if (tid < size) {
        d_out[tid] = my_uniform.generate(rng);
    }

    // Offset is advanced for rng after each generate() call, only need to call skip_subsequence()
    rng.skip_subsequence(size % subsequences);
    if (tid < size / 2) {
        d_out_normal[tid] = my_normal.generate2(rng);
    }
}


template<unsigned int Arch>
int mrg_two_executions_thread_api() {
    using RNG = decltype(curanddx::Generator<curanddx::mrg32k3a>() + curanddx::SM<Arch>() + curanddx::Thread());

    // Allocate output memory
    double*            d_out_uniform_double;
    float*             d_out_normal;
    const unsigned int size = 5000;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_out_uniform_double, size * sizeof(double)));
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_out_normal, size * sizeof(float)));

    const unsigned long long        seed   = 1234ULL;
    const typename RNG::offset_type offset = 0ULL;

    const unsigned int block_dim = 256;
    const unsigned int grid_size = (size + block_dim - 1) / block_dim;

    // Launch a kernel to generate two distributions
    generate_kernel<RNG><<<grid_size, block_dim, 0>>>(d_out_uniform_double, (float2*)d_out_normal, seed, offset, size);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy to host
    std::vector<double> h_out_uniform_double(size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(h_out_uniform_double.data(), d_out_uniform_double, size * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<float> h_out_normal(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_out_normal.data(), d_out_normal, size * sizeof(float), cudaMemcpyDeviceToHost));

    // cuRAND host API
    curandGenerator_t gen_curand;
    double*           d_ref_uniform_double;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_ref_uniform_double, size * sizeof(double)));
    float* d_ref_normal;
    CUDA_CHECK_AND_EXIT(cudaMalloc((void**)&d_ref_normal, size * sizeof(float)));

    CURAND_CHECK_AND_EXIT(curandCreateGenerator(&gen_curand, CURAND_RNG_PSEUDO_MRG32K3A));
    CURAND_CHECK_AND_EXIT(curandSetPseudoRandomGeneratorSeed(gen_curand, seed));
    CURAND_CHECK_AND_EXIT(curandSetGeneratorOffset(gen_curand, 0));
    CURAND_CHECK_AND_EXIT(curandSetGeneratorOrdering(gen_curand, CURAND_ORDERING_PSEUDO_LEGACY));

    CURAND_CHECK_AND_EXIT(curandGenerateUniformDouble(gen_curand, d_ref_uniform_double, size));
    CURAND_CHECK_AND_EXIT(curandGenerateNormal(gen_curand, d_ref_normal, size, 0, 1));

    std::vector<double> h_ref_uniform_double(size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(h_ref_uniform_double.data(), d_ref_uniform_double, size * sizeof(double), cudaMemcpyDeviceToHost));
    std::vector<float> h_ref_normal(size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(h_ref_normal.data(), d_ref_normal, size * sizeof(float), cudaMemcpyDeviceToHost));

    CURAND_CHECK_AND_EXIT(curandDestroyGenerator(gen_curand));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out_uniform_double));
    CUDA_CHECK_AND_EXIT(cudaFree(d_out_normal));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ref_uniform_double));
    CUDA_CHECK_AND_EXIT(cudaFree(d_ref_normal));

    // Compare Results
    if (h_out_uniform_double == h_ref_uniform_double && h_out_normal == h_ref_normal) {
        std::cout
            << "SUCCESS: \nSame sequence is generated with cuRANDDx and cuRAND Host API for three distributions.\n";
        return 0;
    } else {
        if (h_out_uniform_double != h_ref_uniform_double) {
            int count {0};
            for (auto i = 0U; i < size; i++) {
                if (h_out_uniform_double[i] != h_ref_uniform_double[i] && count++ < 10) {
                    std::cout << "uniform: array_curanddx[" << i << "] = " << h_out_uniform_double[i]
                              << ", array_curand = " << h_ref_uniform_double[i] << " \n";
                }
            }
        }
        if (h_out_normal != h_ref_normal) {
            int count = 0;
            for (auto i = 0U; i < size; i++) {
                if (h_out_normal[i] != h_ref_normal[i] && count++ < 10) {
                    std::cout << "normal: array_curanddx[" << i << "] = " << h_out_normal[i]
                              << ", array_curand = " << h_ref_normal[i] << " \n";
                }
            }
        }
        std::cout << "FAILED: \nDifferent sequence is generated with cuRANDDx and cuRAND Host API using LEGACY "
                     "ordering.\n";
        return 1;
    }
}

template<unsigned int Arch>
struct mrg_two_executions_thread_api_functor {
    int operator()() { return mrg_two_executions_thread_api<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<mrg_two_executions_thread_api_functor>();
}