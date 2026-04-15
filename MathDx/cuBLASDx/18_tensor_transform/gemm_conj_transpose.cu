/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This example demonstrates a complex GEMM where matrix A is used in its
// conjugate-transpose form:
//
//   C = alpha * A^H * B + beta * C
//
// where A is an (M x M) complex single-precision matrix, B is (M x N), and
// C is (M x N).  All matrices use column-major layout.
//
// The conjugate-transpose is achieved in two steps:
//   1. cublasdx::transpose_view(smem_a) swaps the first two tensor modes,
//      presenting A^T to the executor without any data movement.
//   2. cublasdx::conjugate{} is passed as an ALoadOp, applying element-wise
//      conjugation during the GEMM's internal copy from shared memory to
//      registers.
//
// Together these give: conj(A^T) = A^H.
//
// Note: M must equal K so that A is square.  transpose_view swaps the first
// two modes of the tensor, so using it on a non-square matrix would change the
// logical dimensions and break the GEMM size contract.
//
// Correctness is verified against a cuBLAS reference that receives the
// host-side conjugate-transposed A matrix.

#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cublasdx.hpp>

#include "../common/common.hpp"
#include "../reference/reference.hpp"

// -------------------------------------------------------------------------
// gemm_conj_transpose_kernel
//
// Each block performs a single GEMM: C = alpha * A^H * B + beta * C.
// transpose_view is applied to A in shared memory and conjugate{} is used
// as a load-time transform so the GEMM operates on A^H.
// -------------------------------------------------------------------------
template<class BLAS, class ValueType = typename example::uniform_value_type_t<BLAS>>
__launch_bounds__(BLAS::max_threads_per_block) __global__ void gemm_conj_transpose_kernel(const ValueType* a,
                                                                                          const ValueType* b,
                                                                                          const ValueType* c,
                                                                                          const ValueType  alpha,
                                                                                          const ValueType  beta,
                                                                                          ValueType*       output) {
    extern __shared__ __align__(16) cublasdx::byte smem[];

    auto a_global_tensor = cublasdx::make_tensor(a, BLAS::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, BLAS::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, BLAS::get_layout_gmem_c());

    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<BLAS>(smem);
    auto a_shared_tensor          = cublasdx::make_tensor(smem_a, BLAS::get_layout_smem_a());
    auto b_shared_tensor          = cublasdx::make_tensor(smem_b, BLAS::get_layout_smem_b());
    auto c_shared_tensor          = cublasdx::make_tensor(smem_c, BLAS::get_layout_smem_c());

    using alignment = cublasdx::alignment_of<BLAS>;
    cublasdx::copy<BLAS, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<BLAS, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<BLAS, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    // Compute C = alpha * A^H * B + beta * C:
    //   transpose_view(smem_a) presents A^T (layout change only, no data copy),
    //   cublasdx::conjugate{} as ALoadOp applies element-wise conjugation.
    BLAS().execute(alpha,
                   cublasdx::conj_transpose_view(a_shared_tensor),
                   b_shared_tensor,
                   beta,
                   cublasdx::transpose_view(c_shared_tensor));
    __syncthreads();

    auto out_global_tensor = cublasdx::make_tensor(output, BLAS::get_layout_gmem_c());
    cublasdx::copy<BLAS, alignment::c>(c_shared_tensor, out_global_tensor);
}

// -------------------------------------------------------------------------
// gemm_conj_transpose  - main example function, instantiated per SM arch
// -------------------------------------------------------------------------
template<unsigned int Arch>
int gemm_conj_transpose() {
    // M == K so that A is square and transpose_view preserves tensor shape.
    constexpr unsigned int square_size = 32;

    // All sizes are the same to support transpose_view and conj_transpose_view
    constexpr unsigned int m = square_size;
    constexpr unsigned int n = square_size;
    constexpr unsigned int k = square_size;

    // GEMM definition: C = alpha * A * B + beta * C
    // At execute time we pass transpose_view(A) with a conjugate load op,
    // effectively computing C = alpha * A^H * B + beta * C.
    using BLAS = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<float>() +
                          cublasdx::Type<cublasdx::type::complex>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Block() +
                          cublasdx::BlockDim<256>() +
                          cublasdx::SM<Arch>());

    using value_type = typename example::uniform_value_type_t<BLAS>;

    // Allocate managed memory for A, B, C, and output
    constexpr auto global_a_size = example::global_memory_size_of<BLAS>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<BLAS>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<BLAS>::c_size;

    value_type* inputs;
    value_type* output;
    auto        inputs_size       = global_a_size + global_b_size + global_c_size;
    auto        inputs_size_bytes = inputs_size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&inputs, inputs_size_bytes));
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&output, global_c_size * sizeof(value_type)));

    value_type* a     = inputs;
    value_type* b     = a + global_a_size;
    value_type* c     = b + global_b_size;
    value_type  alpha = value_type(1.0, 0.5);
    value_type  beta  = value_type(2.0, 0.0);

    // Fill A, B, C with random complex values
    auto host_a = example::get_random_data<value_type>(global_a_size);
    auto host_b = example::get_random_data<value_type>(global_b_size);
    auto host_c = example::get_random_data<value_type>(global_c_size);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(a, host_a.data(), global_a_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(b, host_b.data(), global_b_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(c, host_c.data(), global_c_size * sizeof(value_type), cudaMemcpyHostToDevice));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Set shared memory and launch kernel
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(gemm_conj_transpose_kernel<BLAS>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             cublasdx::get_shared_storage_size<BLAS>()));

    gemm_conj_transpose_kernel<BLAS>
        <<<1, BLAS::block_dim, cublasdx::get_shared_storage_size<BLAS>()>>>(a, b, c, alpha, beta, output);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<value_type> host_output(global_c_size);
    CUDA_CHECK_AND_EXIT(
        cudaMemcpy(host_output.data(), output, global_c_size * sizeof(value_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Free device memory
    CUDA_CHECK_AND_EXIT(cudaFree(inputs));
    CUDA_CHECK_AND_EXIT(cudaFree(output));

    // -----------------------------------------------------------------------
    // Reference: compute A^H on the host and pass it to reference_gemm.
    //
    // A is col_major (m x k): element (i,j) is at index [i + j*m].
    // A^H(i,j) = conj(A(j,i)) = conj(host_a[j + i*m]).
    // Since m == k, A^H is also (m x k) col_major.
    // -----------------------------------------------------------------------
    std::vector<value_type> host_a_ct(global_a_size);
    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < k; ++j) {
            auto val             = host_a[j + i * m];
            host_a_ct[i + j * m] = value_type(val.real(), -val.imag());
        }
    }
 
    // Transpose C
    std::vector<value_type> host_c_ct(global_c_size);
    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            host_c_ct[i + j * m] = host_c[j + i * m];
        }
    }

    auto reference_host_output = example::reference_gemm<BLAS>(alpha, host_a_ct, host_b, beta, host_c_ct);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());

    // Transpose results
    std::vector<value_type> reference_host_output_ct(global_c_size);
    for (unsigned int i = 0; i < m; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            reference_host_output_ct[i + j * m] = reference_host_output[j + i * m];
        }
    }

    // Check against reference
    if (example::check_error<BLAS>(host_output, reference_host_output_ct)) {
        std::cout << "Success" << std::endl;
        return 0;
    }
    std::cout << "Failure" << std::endl;
    return 1;
}

struct gemm_conj_transpose_functor {
    template<int Arch, cublasdx::sm_modifier Modifier>
    int operator()(std::integral_constant<int, Arch>, std::integral_constant<cublasdx::sm_modifier, Modifier>) {
        return gemm_conj_transpose<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(gemm_conj_transpose_functor {});
}
