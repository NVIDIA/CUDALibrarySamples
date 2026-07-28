/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

template<class GEMM>
__global__ void gemm_kernel_shared(const typename GEMM::c_value_type  alpha,
                                   const typename GEMM::a_value_type* a,
                                   const typename GEMM::b_value_type* b,
                                   const typename GEMM::c_value_type  beta,
                                   typename GEMM::c_value_type*       c) {
    CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(GEMM);
    extern __shared__ __align__(16) cublasdx::byte smem[];

    // Make global memory tensor
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

    // Make shared memory tensor
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a_shared_tensor          = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor          = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
    auto c_shared_tensor          = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

    // Load data from global memory tensor to shared memory tensor
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<GEMM, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    // Execute GEMM
    GEMM().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
    __syncthreads();

    // Store data from shared memory tensor to global memory tensor
    cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
}

template<class GEMM>
__global__ void gemm_kernel_registers_accumulation(const typename GEMM::a_value_type* a,
                                                   const typename GEMM::b_value_type* b,
                                                   typename GEMM::c_value_type*       c) {
    CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(GEMM);
    extern __shared__ __align__(16) cublasdx::byte smem[];

    // Make global memory tensor
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

    // Make shared memory tensor
    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
    auto a_shared_tensor  = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor  = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());

    // Load data from global memory tensor to shared memory tensor
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    // Get default accumulator
    auto accumulator = GEMM::get_accumulator();

    // Execute GEMM with accumulation
    GEMM().execute(a_shared_tensor, b_shared_tensor, accumulator);

    // Partition Global C for GEMM and load appropriate elements into accumulator
    auto c_frag = accumulator.make_partition_and_copy(c_global_tensor);
    auto results = accumulator.get_results();
    cublasdx::axpby(1.0, results, 1.0, c_frag);

    // Partition Global C for GEMM and store appropriate elements to global memory
    accumulator.partition_and_copy(c_frag, c_global_tensor);
}

template<class GEMM>
__global__ void gemm_kernel_registers(const typename GEMM::a_value_type* a,
                                      const typename GEMM::b_value_type* b,
                                      typename GEMM::c_value_type*       c) {
    CUBLASDX_SKIP_IF_NOT_APPLICABLE_SM(GEMM);
    extern __shared__ __align__(16) cublasdx::byte smem[];

    // Make global memory tensor
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

    // Make shared memory tensor
    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
    auto a_shared_tensor  = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor  = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());

    // Load data from global memory tensor to shared memory tensor
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    // Execute GEMM and get accumulator in return
    auto accumulator = GEMM().execute(a_shared_tensor, b_shared_tensor);

    // Partition Global C for GEMM and store appropriate elements to global memory
    accumulator.partition_and_store(c_global_tensor);
}

template<unsigned int Arch>
int introduction_example() {
    using GEMM =
        decltype(cublasdx::Size<32, 32, 32>() + cublasdx::Precision<double>() + cublasdx::Type<cublasdx::type::real>() +
                 cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>() +
                 cublasdx::Function<cublasdx::function::MM>() + cublasdx::SM<Arch>() + cublasdx::Block() +
                 cublasdx::BlockDim<256>());

    using value_type = typename example::uniform_value_type_t<GEMM>;

    constexpr auto global_a_size = example::global_memory_size_of<GEMM>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<GEMM>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<GEMM>::c_size;

    std::vector<value_type> host_a(global_a_size);
    std::vector<value_type> host_b(global_b_size);
    std::vector<value_type> host_c(global_c_size);

    for (size_t i = 0; i < global_a_size; i++) {
        host_a[i] = static_cast<value_type>(static_cast<double>((i % 17) + 1) / 17.0);
    }
    for (size_t i = 0; i < global_b_size; i++) {
        host_b[i] = static_cast<value_type>(static_cast<double>((i % 13) + 1) / 13.0);
    }
    for (size_t i = 0; i < global_c_size; i++) {
        host_c[i] = static_cast<value_type>(static_cast<double>((i % 11) + 1) / 11.0);
    }

    // Allocate device memory for A, B, C matrices in one go
    value_type* abc;
    auto        size       = global_a_size + global_b_size + global_c_size;
    auto        size_bytes = size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMalloc(&abc, size_bytes));

    value_type* a = abc;
    value_type* b = abc + global_a_size;
    value_type* c = abc + global_a_size + global_b_size;

    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c_size * sizeof(value_type), cudaMemcpyHostToDevice));

    // Shared Memory API: C = alpha * A * B + beta * C
    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel_shared<GEMM><<<1, GEMM::block_dim, cublasdx::get_shared_storage_size<GEMM>()>>>(1.0, a, b, 1.0, c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    // Register Fragment Accumulation API: C = A * B + C
    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel_registers_accumulation<GEMM>
        <<<1, GEMM::block_dim, cublasdx::get_shared_storage_size_ab<GEMM>()>>>(a, b, c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    // Register Fragment API: C = A * B
    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel_registers<GEMM><<<1, GEMM::block_dim, cublasdx::get_shared_storage_size_ab<GEMM>()>>>(a, b, c);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    std::vector<value_type> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(host_output.data(), c, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    auto reference_host_output = example::reference_gemm<GEMM>(
        static_cast<value_type>(1.0), host_a, host_b, static_cast<value_type>(0.0), host_c);
    bool correct = example::check_error<GEMM>(host_output, reference_host_output);

    CUDA_CHECK_AND_EXIT(cudaFree(abc));
    if (correct) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

struct introduction_example_functor {
    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>, std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return introduction_example<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(introduction_example_functor {});
}
