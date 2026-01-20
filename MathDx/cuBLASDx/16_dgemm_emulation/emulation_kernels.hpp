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
#include <cassert>
#include <cstdio>

#include <cublasdx.hpp>
using namespace cublasdx;

#include <cub/block/block_reduce.cuh>

#include "../common/common.hpp"
#include "tensor_helpers.hpp"

// This header contains the CUDA kernels implementing the Ozaki scheme for
// emulating double precision GEMM using multiple lower precision operations.
//
// The Ozaki scheme consists of several stages:
// 1. Preprocessing: Find maximum values to determine scaling factors
// 2. Slicing: Decompose double precision values into int8_t slices
// 3. Matrix multiplication: Compute GEMM on slice combinations
// 4. Reconstruction: Combine results back to double precision
//
// Each kernel below implements one of these stages.

// ============================================================================
// OZAKI SCHEME KERNEL 1: PREPROCESSING - Maximum Value Reduction
// ============================================================================


enum class slice_matrix
{
    a,
    b
};

// This kernel finds the maximum absolute value in each row/column of the input matrix.
// This is needed to determine appropriate scaling factors for the slicing process.
//
// The maximum value is converted to an exponent shift using max_to_exponent_shift(),
// which determines how many bits we need to represent the largest value in each row/column.
//
// Template parameters:
//   BlockSize: Number of threads per block for reduction
//   InTensor: Input tensor type
//   OutTensor: Output tensor type
//
// Kernel arguments:
//   in_tensor: Input matrix (double precision)
//   out_tensor: Output shift values (one per row/column)
template<int BlockSize, slice_matrix SliceMatrix, class InTensor, class OutTensor>
__launch_bounds__(BlockSize, 2) __global__ void max_reduce_kernel(InTensor in_tensor, OutTensor out_tensor) {
    using datatype    = example::tensor_value_type_t<InTensor>;
    using BlockReduce = cub::BlockReduce<datatype, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const auto [tile_size_x, tile_size_y] = in_tensor.layout().shape();
    auto tid                              = threadIdx.x;
    auto bid                              = blockIdx.x;

    // Assume that tensor is reduced along the last dimension
    auto const row_index = example::conditional_return<SliceMatrix == slice_matrix::a>(bid, cublasdx::slice);
    auto const col_index = example::conditional_return<SliceMatrix == slice_matrix::a>(cublasdx::slice, bid);

    auto global_tile = in_tensor(row_index, col_index);

    // 1. Find local maximum absolute value for this thread
    double local_max = 0;

    auto const length = (SliceMatrix == slice_matrix::a) ? tile_size_y : tile_size_x;
    for (auto i = tid; i < length; i += BlockSize) {
        local_max = cuda::std::max<double>(local_max, cuda::std::abs(global_tile(i)));
    }

    // 2. Compute block-wide reduction to find maximum across all threads
    __syncthreads();
    const double block_max = BlockReduce(temp_storage).Reduce(local_max, [](const auto& a, const auto& b) {
        return cuda::std::max<double>(a, b);
    });

    // 3. Convert maximum value to exponent shift and store to global memory
    // This shift determines the scaling factor for slicing this row/column
    if (tid == 0) {
        out_tensor(bid) = max_to_exponent_shift(block_max);
    }
}

// ============================================================================
// OZAKI SCHEME KERNEL 2: SLICING - Double Precision to Int8 Decomposition
// ============================================================================

// This kernel decomposes each double precision value into multiple int8_t slices.
// For a double precision value x with scaling factor s, we create slices such that:
//   x ≈ Σ(i=0 to slices-1) slice_i * 2^(s - i*8)
// where each slice_i is an int8_t value.
//
// Template parameters:
//   BlockSize: Number of threads per block
//   Slices: Number of slices per double precision value
//   InTensor: Input tensor container type
//   ShiftTensor: Shift tensor container type
//   OutTensor: Output tensor container type
//
// Kernel arguments:
//   in_tensor: Input matrix (double precision)
//   shift_tensor: Scaling factors computed by max_reduce_kernel
//   out_tensor: Output slices [slices, rows, cols]
//   reduction_dim_size: Size of the reduction dimension (for indexing)
template<int BlockSize, int Slices, slice_matrix SliceMatrix, class InTensor, class ShiftTensor, class OutTensor>
__launch_bounds__(BlockSize, 2) __global__
    void slice_kernel(InTensor in_tensor, ShiftTensor shift_tensor, OutTensor out_tensor, int32_t reduction_dim_size) {
    using in_datatype  = example::tensor_value_type_t<InTensor>;
    using out_datatype = example::tensor_value_type_t<OutTensor>;

    const auto tid = threadIdx.x + blockIdx.x * BlockSize;

    // Calculate which matrix element this thread processes
    auto slow_idx = tid / reduction_dim_size;
    auto fast_idx = tid % reduction_dim_size;

    auto const row_idx = (SliceMatrix == slice_matrix::a) ? slow_idx : fast_idx;
    auto const col_idx = (SliceMatrix == slice_matrix::a) ? fast_idx : slow_idx;

    // Decompose the double precision value into multiple int8_t slices
    // using the appropriate scaling factor for this row/column
    const cuda::std::array slices =
        slices_from_fp64<out_datatype, Slices>(in_tensor(row_idx, col_idx), shift_tensor(slow_idx));

// Store all slices for this matrix element
#pragma unroll
    for (int elem = 0; elem < Slices; ++elem) {
        out_tensor(row_idx, col_idx, elem) = slices[elem];
    }
}

// ============================================================================
// OZAKI SCHEME KERNEL 3: MATRIX MULTIPLICATION WITH RECONSTRUCTION
// ============================================================================

// This is the main kernel that performs the Ozaki scheme matrix multiplication.
// It computes the product: C = A * B where A and B have been decomposed into slices.
//
// The algorithm iterates over slice combinations in a diagonal pattern:
//   - diag = 0: A_slice[0] * B_slice[0]
//   - diag = 1: A_slice[0] * B_slice[1] + A_slice[1] * B_slice[0]
//   - diag = 2: A_slice[0] * B_slice[2] + A_slice[1] * B_slice[1] + A_slice[2] * B_slice[0]
//   - etc.
//
// Each diagonal represents slice combinations that contribute to the same
// power of 2 in the final result reconstruction.
//
// Template parameters:
//   BLAS: cuBLASDx BLAS type defining tile size, precision, etc.
//   Alpha: Alpha scalar type
//   ATensor: A matrix slice tensor types
//   BTensor, BLayout: B matrix slice tensor types
//   Beta: Beta scalar type
//   CTensor: C matrix tensor types (double precision)
//   AShiftTensor: A scaling factor tensor types
//   BShiftTensor: B scaling factor tensor types
//   Slices: Number of slices in the decomposition
//
// Kernel arguments:
//   alpha: Scalar multiplier for A*B
//   gmem_a: A matrix slices [slices, m, k]
//   gmem_b: B matrix slices [slices, k, n]
//   beta: Scalar multiplier for existing C values
//   gmem_c_fp64: C matrix (double precision output)
//   gmem_shift_a: A scaling factors (one per row)
//   gmem_shift_b: B scaling factors (one per column)
template<class BLAS,
         class DevicePipeline,
         class Alpha,
         class Beta,
         class CTensor,
         class AShiftTensor,
         class BShiftTensor,
         int32_t Slices>
__launch_bounds__(DevicePipeline::max_threads_per_block, 1) __global__
    void fused_epilogue_kernel(__grid_constant__ DevicePipeline const device_pipeline,
                               Alpha                                  alpha,
                               Beta                                   beta,
                               CTensor                                gmem_c_fp64,
                               AShiftTensor const                     gmem_shift_a,
                               BShiftTensor const                     gmem_shift_b) {
#ifdef __CUDA_ARCH__
    extern __shared__ __align__(device_pipeline.buffer_alignment()) char smem[];
    if constexpr (cublasdx::sm_of_v<BLAS> == __CUDA_ARCH__) {
        // ================================
        // 1. SETUP AND TILE PREPARATION
        // ================================

        constexpr int tile_m = cublasdx::size_of_v_m<BLAS>;
        constexpr int tile_n = cublasdx::size_of_v_n<BLAS>;

        constexpr auto initial_diag = Slices - 1;
        constexpr auto initial_term = 0;

        auto const smem_shift_layout_a =
            example::make_layout_from_tuples(cuda::std::make_tuple(cuda::std::integral_constant<int, tile_m> {}),
                                             cuda::std::make_tuple(cuda::std::integral_constant<int, 1> {}));

        auto const smem_shift_layout_b =
            example::make_layout_from_tuples(cuda::std::make_tuple(cuda::std::integral_constant<int, tile_n> {}),
                                             cuda::std::make_tuple(cuda::std::integral_constant<int, 1> {}));

        auto [pipeline_smem, smem_shift_a, smem_shift_b] = cublasdx::shared_memory::
            slice<char, example::tensor_value_type_t<AShiftTensor>, example::tensor_value_type_t<BShiftTensor>>(
                smem,
                device_pipeline.buffer_alignment(),
                device_pipeline.buffer_size(),
                cublasdx::alignment_of_v_a<BLAS>,
                smem_shift_layout_a,
                cublasdx::alignment_of_v_b<BLAS>,
                smem_shift_layout_b);

        // Copy general purpose data
        cublasdx::copy<BLAS, 16>(gmem_shift_a(cublasdx::slice, blockIdx.x), smem_shift_a);
        cublasdx::copy<BLAS, 16>(gmem_shift_b(cublasdx::slice, blockIdx.y), smem_shift_b);
        cublasdx::copy_wait();

        // Get pipeline tile
        auto tile_pipeline = device_pipeline.get_tile(pipeline_smem,
                                                      cublasdx::make_coord(blockIdx.x, initial_term),
                                                      cublasdx::make_coord(blockIdx.y, initial_diag));

        auto accumulator = tile_pipeline.get_accumulator();

        // ================================
        // 2. FP64 C INPUT / OUTPUT TILE SETUP
        // ================================

        auto tile_c_fp64_gmem = cublasdx::get_tile(gmem_c_fp64, BLAS::c_shape, blockIdx.x, blockIdx.y);

        // ============================================
        // 3. OZAKI SCHEME DIAGONAL ITERATION
        // ============================================

        // Iterate over diagonals in reverse order (highest power of 2 first)
        // This ensures proper accumulation order for numerical stability
#    pragma unroll 1
        for (auto diag = initial_diag; diag >= 0; --diag) {

            // Initialize accumulator for this diagonal
            accumulator.clear();

            // ==========================================
            // 4. SLICE COMBINATION COMPUTATION
            // ==========================================

            // Compute all slice combinations that contribute to this diagonal
            // For diagonal d, we compute: A_slice[i] * B_slice[d-i] for i = 0 to d
#    pragma unroll 1
            for (auto term = initial_term; term <= diag; ++term) {
                // =========================================
                // 5. N-STAGE MEMORY PIPELINE FOR GEMM
                // =========================================

                tile_pipeline.execute(accumulator);

                const auto next_slice_row = (term == diag) ? 0 : term + 1;                         // A slice index
                const auto next_slice_col = (term == diag) ? (diag - 1) : (diag - next_slice_row); // B slice index
                device_pipeline.reset_tile(tile_pipeline,
                                           cublasdx::make_coord(blockIdx.x, next_slice_row),
                                           cublasdx::make_coord(blockIdx.y, next_slice_col));
            } /* end of slice combination loop */

            // ========================================
            // 6. RESULT RECONSTRUCTION AND EPILOGUE
            // ========================================
            // Convert accumulated int32_t results back to double precision
            // and apply appropriate scaling based on slice positions
	    if(accumulator.is_thread_active()) {
                auto gemm_results = accumulator.get_results();
    
                // Load existing C values
                auto d_fp64_frag = cublasdx::make_fragment_like<double>(gemm_results);
                auto c_fp64_frag = accumulator.make_partition_and_copy(tile_c_fp64_gmem);
    
                // Process each element in the register fragment
    #    pragma unroll
                for (int i = 0; i < cublasdx::size(d_fp64_frag); ++i) {
                    const auto [global_x, global_y] = accumulator.map_fragment_index(i);
                    const auto shift_a_elem         = smem_shift_a(global_x);
                    const auto shift_b_elem         = smem_shift_b(global_y);
    
                    // Convert int32_t slice result back to double precision
                    // with appropriate scaling for this diagonal and element
                    d_fp64_frag(i) = nth_slice_to_fp64<int32_t, int8_t>(diag, gemm_results(i), shift_a_elem + shift_b_elem);
                }
    
                // Apply alpha/beta scaling and accumulate into C
                // Use beta only for the first diagonal (highest order), then just add (beta=1.0)
                cublasdx::axpby(alpha, d_fp64_frag, (diag == Slices - 1) ? beta : 1.0, c_fp64_frag);
    
                // Store results back to global memory
                accumulator.partition_and_copy(c_fp64_frag, tile_c_fp64_gmem);
	    }
        }
    }
#endif
}

// ============================================================================
// OZAKI SCHEME ORCHESTRATION FUNCTION
// ============================================================================

// This function orchestrates the entire slice matrix multiplication process.
// It sets up the cuBLASDx kernel configuration and launches the fused epilogue kernel.
//
// The function handles:
//   1. cuBLASDx type creation and configuration
//   2. Grid and block dimension calculation
//   3. Shared memory size calculation
//   4. Kernel launch with timing
//   5. Result collection and error checking
//
// Template parameters:
//   Arch: GPU architecture (SM version)
//   Params: Emulation parameters (tile shape, slices, etc.)
//   GEMMShape: Shape of the GEMM operation
//   GEMMArr: Memory layout arrangement
//
// Returns:
//   Tuple of (execution_time_ms, results_vector)
template<int                   Arch,
         cublasdx::sm_modifier Modifier,
         class Params,
         class GEMMShape,
         class GEMMArr,
         class AShiftTensor,
         class BShiftTensor,
         class Alpha,
         class ASliceTensor,
         class BSliceTensor,
         class Beta,
         class CTensor>
auto slice_matmul_and_epilogue(GEMMShape          gemm_shape,
                               GEMMArr            gemm_arrangement,
                               AShiftTensor const d_tensor_shift_a,
                               BShiftTensor const d_tensor_shift_b,
                               Alpha              alpha,
                               ASliceTensor const d_tensor_slice_a,
                               BSliceTensor const d_tensor_slice_b,
                               Beta               beta,
                               CTensor const      d_tensor_c,
                               cudaStream_t       stream) {
    // ================================
    // Type definitions and validation
    // ================================

    using SliceAValueType = example::tensor_value_type_t<ASliceTensor>; // int8_t
    using SliceBValueType = example::tensor_value_type_t<BSliceTensor>; // int8_t
    using AccValueType    = example::tensor_value_type_t<AShiftTensor>; // int32_t
    using OutputType      = example::tensor_value_type_t<CTensor>;      // double

    static_assert(std::is_same_v<SliceAValueType, SliceBValueType>, "A and B slices must be of the same type");

    constexpr typename Params::tile_shape tile_shape = {};
    constexpr typename Params::cta_shape  cta_shape  = {};

    // Timing events for performance measurement
    cudaEvent_t start, stop;
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&start));
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&stop));

    // Extract problem dimensions
    const unsigned m = cuda::std::get<0>(gemm_shape);
    const unsigned n = cuda::std::get<1>(gemm_shape);
    const unsigned k = cuda::std::get<2>(gemm_shape);

    /* ===================================================================== */
    /*                        Prepare cuBLASDx kernel                        */
    /* ===================================================================== */

    // Extract memory layout arrangements
    constexpr auto global_arrangement_a = cuda::std::get<0>(gemm_arrangement);
    constexpr auto global_arrangement_b = cuda::std::get<1>(gemm_arrangement);
    constexpr auto global_arrangement_c = cuda::std::get<2>(gemm_arrangement);

    // ================================
    // cuBLASDx precision configuration
    // ================================

    // Compute precision (use Tensor Cores of this precision)
    using a_compute_precision = SliceAValueType; // int8_t for slices
    using b_compute_precision = SliceBValueType; // int8_t for slices
    using c_compute_precision = AccValueType;    // int32_t for accumulation

    // Number type, either real or complex
    constexpr auto type = cublasdx::type::real;

    // Create data type, based on precision and type (real / complex)
    using a_compute_value_type = example::get_value_type_t<a_compute_precision>;
    using b_compute_value_type = example::get_value_type_t<b_compute_precision>;
    using c_compute_value_type = example::get_value_type_t<c_compute_precision>;

    // ======================================
    // Configurable cuBLASDx tile properties
    // ======================================

    // Extract tile dimensions from parameters
    constexpr unsigned int tile_m = cuda::std::get<0>(tile_shape);
    constexpr unsigned int tile_n = cuda::std::get<1>(tile_shape);
    constexpr unsigned int tile_k = cuda::std::get<2>(tile_shape);

    // Number of threads to compute the tile
    constexpr unsigned int cta_threads_x = cuda::std::get<0>(cta_shape);
    constexpr unsigned int cta_threads_y = cuda::std::get<1>(cta_shape);
    constexpr unsigned int cta_threads_z = cuda::std::get<2>(cta_shape);

    // Arrangement of data in per-threadblock tiles
    constexpr auto tile_arr_a      = global_arrangement_a;
    constexpr auto tile_arr_b      = global_arrangement_b;
    constexpr auto tile_arr_c_fp64 = global_arrangement_c;

    // Maximal alignment for shared memory vectorization
    constexpr unsigned int maximal_alignment  = 16;
    constexpr unsigned int cublasdx_alignment = maximal_alignment;

    // ================================
    // Verify configuration correctness
    // ================================

    const bool divisible = (m % tile_m == 0 and n % tile_n == 0 and k % tile_k == 0);
    if (not divisible) {
        std::cerr << "M, N, K dimensions must be divisible by tile_m, tile_n, tile_k" << std::endl;
        assert(false);
    }

    // ================================
    // cuBLASDx type creation
    // ================================

    using BLAS =
        decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                 cublasdx::Precision<SliceAValueType, SliceBValueType, AccValueType>() + cublasdx::Type<type>() +
                 cublasdx::Function<cublasdx::function::MM>() +
                 cublasdx::Arrangement<tile_arr_a, tile_arr_b, tile_arr_c_fp64>() + cublasdx::Block() +
                 cublasdx::BlockDim<cta_threads_x, cta_threads_y, cta_threads_z>() + cublasdx::StaticBlockDim() +
                 cublasdx::Alignment<cublasdx_alignment, cublasdx_alignment, cublasdx_alignment>() +
                 cublasdx::WithPipeline() + cublasdx::EnableInputStreaming() + cublasdx::SM<Arch, Modifier>());

    // ================================
    // Pipeline configuration
    // ================================

    static constexpr int  manual_pipeline_depth   = 0;
    static constexpr bool override_pipeline_depth = (manual_pipeline_depth != 0);

    constexpr unsigned stage_shared_req = tile_m * tile_k * sizeof(SliceAValueType) +
                                          tile_k * tile_n * sizeof(SliceBValueType) +
                                          sizeof(cublasdx::pipeline_stage_scratch_t);

    constexpr unsigned available_shared_memory =
        commondx::device_info<Arch>::shared_memory() - (tile_m + tile_n) * sizeof(int32_t);
    constexpr unsigned maximal_pipeline_depth =
        cuda::std::min<unsigned>(16, (available_shared_memory - 32) / stage_shared_req);
    constexpr unsigned pipeline_depth = override_pipeline_depth ? manual_pipeline_depth : maximal_pipeline_depth;

    auto opt_device_pipeline = cublasdx::suggest_device_pipeline<pipeline_depth, BLAS, cublasdx::external_accumulation>(
        d_tensor_slice_a, d_tensor_slice_b);

    if (not opt_device_pipeline) {
        std::cout << "Incorrect pipeline configuration, please ensure global tensors are divisible by tile"
                  << std::endl;
        exit(1);
    }

    auto k_stages = k / tile_k;

    if (k_stages < pipeline_depth) {
        std::cerr << "PipelineDepth must be less or equal to GEMM k stages, please adjust manual_pipeline_depth"
                  << std::endl;
        exit(1);
    }

    auto device_pipeline    = opt_device_pipeline.value();
    using device_pipeline_t = cuda::std::remove_cvref_t<decltype(device_pipeline)>;

    // ================================
    // Shared memory size calculation
    // ================================

    auto shared_memory_size = cublasdx::make_shared_storage_calculator()
                                  .add(device_pipeline.buffer_alignment(), device_pipeline.buffer_size())
                                  .add(maximal_alignment, sizeof(int32_t), tile_m) // shift_a
                                  .add(maximal_alignment, sizeof(int32_t), tile_n) // shift_b
                                  .get();

    // ================================
    // Kernel preparation
    // ================================

    auto kernel = fused_epilogue_kernel<BLAS,
                                        device_pipeline_t,
                                        Alpha,        /* alpha */
                                        Beta,         /* beta */
                                        CTensor,      /* CLayout */
                                        AShiftTensor, /* AShiftLayout */
                                        BShiftTensor, /* BShiftLayout */
                                        Params::slices>;

    // Set dynamic shared memory size
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    // =======================================
    // Execute kernel for correctness results
    // =======================================
    //
    dim3 grid_dim = dim3 {(m / tile_m), (n / tile_n), 1};


    // First run to get correct results
    kernel<<<grid_dim, device_pipeline.get_block_dim(), shared_memory_size, stream>>>(
        device_pipeline, alpha, beta, d_tensor_c, d_tensor_shift_a, d_tensor_shift_b);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    // Copy results to host
    std::vector<OutputType> results(m * n);
    CUDA_CHECK_AND_EXIT(cudaMemcpy(
        results.data(), d_tensor_c.data().get(), results.size() * sizeof(OutputType), cudaMemcpyDeviceToHost));

    /* ===================================================================== */
    /*                           Performance measurement                      */
    /* ===================================================================== */

    // Warm-up runs
    for (auto warm_up = 0; warm_up < Params::kernel_warm_up_repeats; ++warm_up) {
        kernel<<<grid_dim, device_pipeline.get_block_dim(), shared_memory_size, stream>>>(
            device_pipeline, alpha, beta, d_tensor_c, d_tensor_shift_a, d_tensor_shift_b);
    }

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    CUDA_CHECK_AND_EXIT(cudaEventRecord(start, stream));

    // Performance measurement runs
    for (auto perf_run = 0; perf_run < Params::kernel_repeats; ++perf_run) {
        kernel<<<grid_dim, device_pipeline.get_block_dim(), shared_memory_size, stream>>>(
            device_pipeline, alpha, beta, d_tensor_c, d_tensor_shift_a, d_tensor_shift_b);
    }

    CUDA_CHECK_AND_EXIT(cudaEventRecord(stop, stream));
    CUDA_CHECK_AND_EXIT(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float time_cublasdx_ms = 0.0f;
    CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time_cublasdx_ms, start, stop));

    // ================================
    // Cleanup and return results
    // ================================

    CUDA_CHECK_AND_EXIT(cudaEventDestroy(start));
    CUDA_CHECK_AND_EXIT(cudaEventDestroy(stop));

    return std::make_tuple(time_cublasdx_ms, results);
}
