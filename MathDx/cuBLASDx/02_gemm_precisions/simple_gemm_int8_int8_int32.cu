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

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

template<class BLAS,
         class AValueType = typename BLAS::a_value_type,
         class BValueType = typename BLAS::b_value_type,
         class CValueType = typename BLAS::c_value_type>
__launch_bounds__(BLAS::max_threads_per_block) //
    __global__                                 //
    void gemm_kernel(const AValueType* a,
                     const BValueType* b,
                     const CValueType* c,
                     const CValueType  alpha,
                     const CValueType  beta,
                     CValueType*       output) {
    extern __shared__ __align__(16) char smem[];

    auto a_global_tensor   = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor   = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor   = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());
    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());

    using a_engine =
        typename decltype(cublasdx::make_tensor(std::declval<decltype(a)>(), BLAS::get_layout_gmem_a()))::engine_type;
    static_assert(std::is_same_v<a_engine, typename decltype(a_global_tensor)::engine_type>, "");


    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<BLAS>(smem);
    auto a_shared_tensor  = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor  = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());


    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    auto accumulator     = BLAS().execute(a_shared_tensor, b_shared_tensor);
    auto result_fragment = accumulator.get_results();

    auto d_frag = accumulator.make_partition_and_copy(c_global_tensor);
    cublasdx::axpby(alpha, result_fragment, beta, d_frag);
    accumulator.partition_and_copy(d_frag, out_global_tensor);
}

// This is an example of int8 / int8 / int32 general matrix-matrix multiplication (GEMM) performed
// in a single CUDA block with use of Tensor Cores:
//
//              C = alpha * A * B + beta * C
//
// * A, B, and C are matrices containing:
//    A --> int8_t
//    B --> int8_t
//    C --> int32_t
// * alpha and beta are real int32_t values.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory. The results are verified against cuBLAS.
//
// In this example the number of threads participating in the GEMM operation is imposed by providing
// BlockDim operator in definition of the GEMM. If BlockDim operator is not used, cuBLASDx automatically
// selects number of threads. Block dimensions are provided via BLAS::block_dim trait.
template<unsigned int Arch>
int simple_gemm() {
    // Parameters m, n, k define the dimensions of matrices A, B, and C
    constexpr unsigned int m = 16;
    constexpr unsigned int n = 32;
    constexpr unsigned int k = 64;

    // Selected CUDA block size (1D)
    constexpr unsigned int block_size = 256;

    // GEMM definition using cuBLASDx operators:
    // 1. The size, the precision, and the type (real or complex) are set.
    // 2. The BLAS function is selected: MM (matrix multiplication).
    // 3. Block operator informs that GEMM should be performed on CUDA block level.
    // 4. BlockDim operator sets CUDA block dimensions that the kernel will be executed with.
    // 5. Targeted CUDA compute capability is selected with SM operator.
    using BLAS = decltype(cublasdx::Size<m, n, k>() + cublasdx::Precision<int8_t, int8_t, int32_t>() +
                          cublasdx::Type<cublasdx::type::real>() + cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Block() + cublasdx::BlockDim<block_size>() + cublasdx::SM<Arch>());

    using a_value_type = example::a_value_type_t<BLAS>;
    using b_value_type = example::b_value_type_t<BLAS>;
    using c_value_type = example::c_value_type_t<BLAS>;

    // Allocate managed memory for a, b, c, and output
    a_value_type* input_a;
    b_value_type* input_b;
    c_value_type* input_c;
    c_value_type* output_c;

    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&input_a, global_a_size * sizeof(a_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&input_b, global_b_size * sizeof(b_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&input_c, global_c_size * sizeof(c_value_type)));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output_c, global_c_size * sizeof(c_value_type)));

    c_value_type alpha = c_value_type(1.0);
    c_value_type beta  = c_value_type(2.0);

    // Fill the A, B, C matrices with random values
    auto host_a = example::get_random_data<a_value_type>(global_a_size);
    auto host_b = example::get_random_data<b_value_type>(global_b_size);
    auto host_c = example::get_random_data<c_value_type>(global_c_size);

    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(input_a, host_a.data(), global_a_size * sizeof(a_value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(input_b, host_b.data(), global_b_size * sizeof(b_value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(input_c, host_c.data(), global_c_size * sizeof(c_value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Increase max dynamic shared memory for the kernel if needed
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(
        gemm_kernel<BLAS>, cudaFuncAttributeMaxDynamicSharedMemorySize, cublasdx::get_shared_storage_size<BLAS>()));

    // Execute kernel
    gemm_kernel<BLAS><<<1, BLAS::block_dim, cublasdx::get_shared_storage_size<BLAS>()>>>(
        input_a, input_b, input_c, alpha, beta, output_c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<c_value_type> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output_c, global_c_size * sizeof(c_value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(input_a));
    CUDA_CHECK_AND_EXIT(cudaFree(input_b));
    CUDA_CHECK_AND_EXIT(cudaFree(input_c));
    CUDA_CHECK_AND_EXIT(cudaFree(output_c));

    // Calculate reference
    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a, host_b, beta, host_c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    // Check against reference
    if (example::check_error<BLAS>(host_output, reference_host_output)) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

struct simple_gemm_functor {
    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>, std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return simple_gemm<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(simple_gemm_functor {});
}
