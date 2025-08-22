#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

#include <cublasdx.hpp>
using namespace cublasdx;

#include <cub/block/block_reduce.cuh>

#include "../common/common.hpp"

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

// This kernel finds the maximum absolute value in each row/column of the input matrix.
// This is needed to determine appropriate scaling factors for the slicing process.
//
// The maximum value is converted to an exponent shift using max_to_exponent_shift(),
// which determines how many bits we need to represent the largest value in each row/column.
//
// Template parameters:
//   BlockSize: Number of threads per block for reduction
//   InEngine: Input tensor engine type
//   InLayout: Input tensor layout type  
//   OutEngine: Output tensor engine type
//   OutLayout: Output tensor layout type
//
// Kernel arguments:
//   in_tensor: Input matrix (double precision)
//   out_tensor: Output shift values (one per row/column)
template<int BlockSize, typename InEngine, class InLayout, class OutEngine, class OutLayout>
__launch_bounds__(BlockSize, 2) __global__ void 
max_reduce_kernel(cute::Tensor<InEngine, InLayout> in_tensor, cute::Tensor<OutEngine, OutLayout> out_tensor) {
    using datatype = typename InEngine::value_type;
    using BlockReduce = cub::BlockReduce<datatype, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const auto tile_size = cute::size(out_tensor.layout());
    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    // Assume that tensor is reduced along the last dimension
    auto global_tile = in_tensor(bid, cute::_);

    // 1. Find local maximum absolute value for this thread
    double local_max = 0;

    for(auto i = tid; i < tile_size; i += BlockSize) {
        local_max = cute::max(local_max, cute::abs(global_tile(i)));
    }

    // 2. Compute block-wide reduction to find maximum across all threads
    __syncthreads();
    const double block_max = BlockReduce(temp_storage).Reduce(local_max, 
        [](const auto& a, const auto& b) { return cute::max(a, b);});
    
    // 3. Convert maximum value to exponent shift and store to global memory
    // This shift determines the scaling factor for slicing this row/column
    if(tid == 0) {
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
//   InEngine: Input tensor engine type (double precision)
//   InLayout: Input tensor layout type
//   ShiftEngine: Shift tensor engine type (int32_t)
//   ShiftLayout: Shift tensor layout type
//   OutEngine: Output tensor engine type (int8_t slices)
//   OutLayout: Output tensor layout type
//
// Kernel arguments:
//   in_tensor: Input matrix (double precision)
//   shift_tensor: Scaling factors computed by max_reduce_kernel
//   out_tensor: Output slices [slices, rows, cols]
//   reduction_dim_size: Size of the reduction dimension (for indexing)
template<int BlockSize, int Slices, 
         class InEngine, class InLayout, 
         class ShiftEngine, class ShiftLayout,
         class OutEngine, class OutLayout>
__launch_bounds__(BlockSize, 2) __global__ void 
slice_kernel(cute::Tensor<InEngine, InLayout> in_tensor, 
             cute::Tensor<ShiftEngine, ShiftLayout> shift_tensor, 
             cute::Tensor<OutEngine, OutLayout> out_tensor, 
             int32_t reduction_dim_size) {
    using in_datatype = typename InEngine::value_type;
    using out_datatype = typename OutEngine::value_type;

    const auto tid = threadIdx.x + blockIdx.x * BlockSize;

    // Calculate which matrix element this thread processes
    auto semantic_dim = tid / reduction_dim_size;
    auto reduction_dim = tid % reduction_dim_size;

    // Decompose the double precision value into multiple int8_t slices
    // using the appropriate scaling factor for this row/column
    const auto slices = slices_from_fp64<out_datatype, Slices>(in_tensor(semantic_dim, reduction_dim), shift_tensor(semantic_dim));
    const auto local_fragment = cute::make_tensor(slices.data(), cute::Layout<cute::Int<Slices>> {});
    
    // Store all slices for this matrix element
    cute::copy(local_fragment, out_tensor(cute::_, semantic_dim, reduction_dim));
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
//   AEngine, ALayout: A matrix slice tensor types
//   BEngine, BLayout: B matrix slice tensor types  
//   Beta: Beta scalar type
//   CEngine, CLayout: C matrix tensor types (double precision)
//   AShiftEngine, AShiftLayout: A scaling factor tensor types
//   BShiftEngine, BShiftLayout: B scaling factor tensor types
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
         class Alpha,
         class AEngine,
         class ALayout,
         class BEngine,
         class BLayout,
         class Beta,
         class CEngine,
         class CLayout,
         class AShiftEngine,
         class AShiftLayout,
         class BShiftEngine,
         class BShiftLayout,
         int32_t Slices>
__launch_bounds__(BLAS::max_threads_per_block, 1) __global__ void fused_epilogue_kernel(Alpha                                          alpha,
                                                                                        cute::Tensor<AEngine, ALayout>           const gmem_a,
                                                                                        cute::Tensor<BEngine, BLayout>           const gmem_b,
                                                                                        Beta                                           beta,
                                                                                        cute::Tensor<CEngine, CLayout>                 gmem_c_fp64,
                                                                                        cute::Tensor<AShiftEngine, AShiftLayout> const gmem_shift_a,
                                                                                        cute::Tensor<BShiftEngine, BShiftLayout> const gmem_shift_b) {
    extern __shared__ __align__(16) char smem[];

    // ================================
    // 1. SETUP AND TENSOR PREPARATION
    // ================================

    using alignment = cublasdx::alignment_of<BLAS>;
    constexpr auto gemm_arr = cute::make_tuple(
        std::integral_constant<cublasdx::arrangement, cublasdx::arrangement_of_v_a<BLAS>>{}, 
        std::integral_constant<cublasdx::arrangement, cublasdx::arrangement_of_v_b<BLAS>>{}, 
        std::integral_constant<cublasdx::arrangement, cublasdx::arrangement_of_v_c<BLAS>>{});

    // Determine which tile this thread block is responsible for
    const auto block_coord = example::get_block_coord(gemm_arr);

    // ================================
    // 2. SHARED MEMORY ALLOCATION
    // ================================
    
    // Slice shared memory into tensors for proper alignment in 2-stage pipelining
    // We need space for:
    //   - Current stage A and B tiles (s_a, s_b)
    //   - Next stage A and B tiles (s_a_n, s_b_n) 
    //   - Scaling factors for this tile (smem_shift_a, smem_shift_b)
    constexpr int tile_m = cublasdx::size_of_v_m<BLAS>;
    constexpr int tile_n = cublasdx::size_of_v_n<BLAS>;

    auto [s_a, s_b, s_a_n, s_b_n, smem_shift_a, smem_shift_b] =
        cublasdx::shared_memory::slice<typename AEngine::value_type, 
                                       typename BEngine::value_type, 
                                       typename AEngine::value_type, 
                                       typename BEngine::value_type, 
                                       typename AShiftEngine::value_type, 
                                       typename BShiftEngine::value_type>(
            smem,
            cublasdx::alignment_of_v_a<BLAS>,
            BLAS::suggest_layout_smem_a(),
            cublasdx::alignment_of_v_b<BLAS>,
            BLAS::suggest_layout_smem_b(),
            cublasdx::alignment_of_v_a<BLAS>,
            BLAS::suggest_layout_smem_a(),
            cublasdx::alignment_of_v_b<BLAS>,
            BLAS::suggest_layout_smem_b(),
            cublasdx::alignment_of_v_a<BLAS>,
            cute::make_layout(cute::Int<tile_m>()),
            cublasdx::alignment_of_v_b<BLAS>,
            cute::make_layout(cute::Int<tile_n>()));

    // Load scaling factors for this tile into shared memory
    cublasdx::copy<BLAS, 16>(gmem_shift_a(cute::_, cute::get<0>(block_coord)), smem_shift_a);
    cublasdx::copy<BLAS, 16>(gmem_shift_b(cute::_, cute::get<1>(block_coord)), smem_shift_b);
    cublasdx::copy_wait();

    // ================================
    // 3. REGISTER FRAGMENT SETUP
    // ================================

    auto partitioner = BLAS().suggest_partitioner();
    auto tile_c_fp64_gmem = example::get_block_tile_c<BLAS>(gmem_c_fp64, block_coord);

    // ============================================
    // 4. OZAKI SCHEME DIAGONAL ITERATION
    // ============================================
    
    // Iterate over diagonals in reverse order (highest power of 2 first)
    // This ensures proper accumulation order for numerical stability
#pragma unroll 1
    for (auto diag = (Slices - 1); diag >= 0; --diag) {

        // Initialize accumulator for this diagonal
        auto d_frag = partitioner.make_accumulator_fragment();
        cublasdx::clear(d_frag);

        // ==========================================
        // 5. SLICE COMBINATION COMPUTATION
        // ==========================================
        
        // Compute all slice combinations that contribute to this diagonal
        // For diagonal d, we compute: A_slice[i] * B_slice[d-i] for i = 0 to d
#pragma unroll 1
        for (auto term = 0; term <= diag; ++term) {
            const auto slice_row = term;         // A slice index
            const auto slice_col = diag - term;  // B slice index

            // Get tiles for this slice combination
            const auto tile_slice_a_gmem = example::get_block_tile_slice_a<BLAS>(gmem_a(slice_row, cute::_, cute::_), block_coord);
            const auto tile_slice_b_gmem = example::get_block_tile_slice_b<BLAS>(gmem_b(slice_col, cute::_, cute::_), block_coord);

            // =========================================
            // 6. 2-STAGE MEMORY PIPELINE FOR GEMM
            // =========================================

            const auto k_stages = cute::get<2>(cute::shape(tile_slice_a_gmem.layout()));

            // Load first stage into shared memory pipeline
            constexpr auto static_first_stage_index = cute::Int<0> {};
            cublasdx::copy<BLAS, alignment::a>(
                example::get_tile_from_slice(tile_slice_a_gmem, static_first_stage_index), s_a);
            cublasdx::copy<BLAS, alignment::b>(
                example::get_tile_from_slice(tile_slice_b_gmem, static_first_stage_index), s_b);

            // ==========================================
            // 7. EXECUTE GEMM WITH MEMORY PIPELINING
            // ==========================================

#pragma unroll 1
            for (int stage = 1; stage <= k_stages; stage++) {
                // Wait for previous stage to complete
                cublasdx::copy_wait();

                // Load next stage (if not the last iteration)  
                if (stage < k_stages) {
                    cublasdx::copy<BLAS, alignment::a>(example::get_tile_from_slice(tile_slice_a_gmem, stage), s_a_n);
                    cublasdx::copy<BLAS, alignment::b>(example::get_tile_from_slice(tile_slice_b_gmem, stage), s_b_n);
                }

                // Perform GEMM on current stage data and accumulate results
                BLAS().execute(s_a, s_b, d_frag);

                // Swap buffers for next iteration
                example::swap(s_a_n, s_a);
                example::swap(s_b_n, s_b);
            }
        } /* end of slice combination loop */

        // ========================================
        // 8. RESULT RECONSTRUCTION AND EPILOGUE  
        // ========================================
        
        // Convert accumulated int32_t results back to double precision
        // and apply appropriate scaling based on slice positions
        auto d_fp64_frag = cublasdx::make_fragment_like<double>(d_frag);
        auto c_fp64_frag = cublasdx::make_fragment_like(d_fp64_frag);
        
        // Load existing C values
        cublasdx::copy_fragment<alignment::c>(tile_c_fp64_gmem, c_fp64_frag, partitioner);

        // Process each element in the register fragment
#pragma unroll
        for (int i = 0; i < cublasdx::size(d_frag); ++i) {
            const auto [global_x, global_y] = partitioner.map_fragment_index(i);
            const auto shift_a_elem = smem_shift_a(global_x);
            const auto shift_b_elem = smem_shift_b(global_y);
            
            // Convert int32_t slice result back to double precision
            // with appropriate scaling for this diagonal and element
            d_fp64_frag(i) = nth_slice_to_fp64<int32_t, int8_t>(
                diag, d_frag(i), shift_a_elem + shift_b_elem);
        }

        // Apply alpha/beta scaling and accumulate into C
        // Use beta only for the first diagonal (highest order), then just add (beta=1.0)
        cublasdx::axpby(alpha, d_fp64_frag, (diag == Slices - 1) ? beta : 1.0, c_fp64_frag);
        
        // Store results back to global memory
        cublasdx::copy_fragment<alignment::c>(c_fp64_frag, tile_c_fp64_gmem, partitioner);
    }
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
//   Various tensor engine and layout types for inputs/outputs
//
// Returns:
//   Tuple of (execution_time_ms, results_vector)
template<int Arch,
         class Params,
         class GEMMShape,
         class GEMMArr,
         class AShiftEngine,
         class AShiftLayout,
         class BShiftEngine,
         class BShiftLayout,
         class Alpha,
         class ASliceEngine,
         class ASliceLayout,
         class BSliceEngine,
         class BSliceLayout,
         class Beta,
         class CEngine,
         class CLayout>
auto slice_matmul_and_epilogue(GEMMShape  gemm_shape,
                               GEMMArr    gemm_arrangement,
                               cute::Tensor<AShiftEngine, AShiftLayout> const d_tensor_shift_a,
                               cute::Tensor<BShiftEngine, BShiftLayout> const d_tensor_shift_b,
                               Alpha      alpha,
                               cute::Tensor<ASliceEngine, ASliceLayout> const d_tensor_slice_a,
                               cute::Tensor<BSliceEngine, BSliceLayout> const d_tensor_slice_b,
                               Beta       beta,
                               cute::Tensor<CEngine, CLayout> const d_tensor_c,
                               cudaStream_t stream) {
    // ================================
    // Type definitions and validation
    // ================================
    
    using SliceAValueType = typename ASliceEngine::value_type;  // int8_t
    using SliceBValueType = typename BSliceEngine::value_type;  // int8_t  
    using AccValueType    = typename AShiftEngine::value_type;  // int32_t
    using OutputType      = typename CEngine::value_type;       // double

    static_assert(std::is_same_v<SliceAValueType, SliceBValueType>, "A and B slices must be of the same type");

    constexpr typename Params::tile_shape tile_shape = {};
    constexpr typename Params::cta_shape  cta_shape  = {};

    // Timing events for performance measurement
    cudaEvent_t start, stop;
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&start));
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&stop));

    // Extract problem dimensions
    const unsigned m = cute::get<0>(gemm_shape);
    const unsigned n = cute::get<1>(gemm_shape);
    const unsigned k = cute::get<2>(gemm_shape);

    /* ===================================================================== */
    /*                        Prepare cuBLASDx kernel                        */
    /* ===================================================================== */

    // Extract memory layout arrangements
    constexpr auto global_arrangement_a       = cute::get<0>(gemm_arrangement);
    constexpr auto global_arrangement_b       = cute::get<1>(gemm_arrangement);
    constexpr auto global_arrangement_c       = cute::get<2>(gemm_arrangement);

    // ================================
    // cuBLASDx precision configuration
    // ================================

    // Compute precision (use Tensor Cores of this precision)
    using a_compute_precision = SliceAValueType;  // int8_t for slices
    using b_compute_precision = SliceBValueType;  // int8_t for slices
    using c_compute_precision = AccValueType;     // int32_t for accumulation

    // Number type, either real or complex
    constexpr auto type = cublasdx::type::real;

    // Create data type, based on precision and type (real / complex)
    using a_compute_value_type = example::get_value_type_t<a_compute_precision, type>;
    using b_compute_value_type = example::get_value_type_t<b_compute_precision, type>;
    using c_compute_value_type = example::get_value_type_t<c_compute_precision, type>;

    // ======================================
    // Configurable cuBLASDx tile properties
    // ======================================

    // Extract tile dimensions from parameters
    constexpr unsigned int tile_m = cute::get<0>(tile_shape);
    constexpr unsigned int tile_n = cute::get<1>(tile_shape);
    constexpr unsigned int tile_k = cute::get<2>(tile_shape);

    // Number of threads to compute the tile
    constexpr unsigned int tile_threads = cute::size(cta_shape);

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

    using BLAS = decltype(cublasdx::Size<tile_m, tile_n, tile_k>() +
                          cublasdx::Precision<SliceAValueType, SliceBValueType, AccValueType>() +
                          cublasdx::Type<type>() + cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<tile_arr_a, tile_arr_b, tile_arr_c_fp64>() + cublasdx::Block() +
                          cublasdx::BlockDim<tile_threads>() +
                          cublasdx::Alignment<cublasdx_alignment, cublasdx_alignment, cublasdx_alignment>() +
                          cublasdx::experimental::StaticBlockDim() + // Experimental: Runtime block dim is equal to operator block dim
                          cublasdx::SM<Arch>());

    // ================================
    // Grid configuration
    // ================================

    constexpr bool reverse_block_coord =
        (global_arrangement_a == cublasdx::row_major) and (global_arrangement_b == cublasdx::row_major);

    dim3 grid_dim = cute::conditional_return<reverse_block_coord>(dim3 {n / tile_n, m / tile_m, 1},
                                                                  dim3 {(m / tile_m), (n / tile_n), 1});

    // ================================
    // Shared memory size calculation
    // ================================

    // Calculate shared memory requirements for all tensors
    auto shared_memory_size =
        cublasdx::make_shared_storage_calculator()
            .add(cublasdx::alignment_of_v_a<BLAS>, sizeof(SliceAValueType), BLAS::suggest_layout_smem_a())
            .add(cublasdx::alignment_of_v_b<BLAS>, sizeof(SliceBValueType), BLAS::suggest_layout_smem_b())
            .add(cublasdx::alignment_of_v_a<BLAS>, sizeof(SliceAValueType), BLAS::suggest_layout_smem_a())
            .add(cublasdx::alignment_of_v_b<BLAS>, sizeof(SliceBValueType), BLAS::suggest_layout_smem_b())
            .add(maximal_alignment, sizeof(int32_t), tile_m)  // shift_a
            .add(maximal_alignment, sizeof(int32_t), tile_n)  // shift_b
            .get();

    // ================================
    // Kernel preparation
    // ================================

    // Extract tensor types for kernel template instantiation
    using a_engine = cute::remove_cvref_t<decltype(d_tensor_slice_a.engine())>;
    using a_layout = cute::remove_cvref_t<decltype(d_tensor_slice_a.layout())>;
    using b_engine = cute::remove_cvref_t<decltype(d_tensor_slice_b.engine())>;
    using b_layout = cute::remove_cvref_t<decltype(d_tensor_slice_b.layout())>;
    using c_engine = cute::remove_cvref_t<decltype(d_tensor_c.engine())>;
    using c_layout = cute::remove_cvref_t<decltype(d_tensor_c.layout())>;
    using a_shift_engine = cute::remove_cvref_t<decltype(d_tensor_shift_a.engine())>;
    using a_shift_layout = cute::remove_cvref_t<decltype(d_tensor_shift_a.layout())>;
    using b_shift_engine = cute::remove_cvref_t<decltype(d_tensor_shift_b.engine())>;
    using b_shift_layout = cute::remove_cvref_t<decltype(d_tensor_shift_b.layout())>;
    
    auto kernel = fused_epilogue_kernel<BLAS,
                                        Alpha,  /* alpha */
                                        a_engine, /* AEngine */
                                        a_layout, /* ALayout */
                                        b_engine, /* BEngine */
                                        b_layout, /* BLayout */
                                        Beta,    /* beta */
                                        c_engine, /* CEngine */
                                        c_layout, /* CLayout */
                                        a_shift_engine, /* AShiftEngine */
                                        a_shift_layout, /* AShiftLayout */
                                        b_shift_engine, /* BShiftEngine */
                                        b_shift_layout, /* BShiftLayout */
                                        Params::slices>;

    // Set dynamic shared memory size
    CUDA_CHECK_AND_EXIT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));

    // =======================================
    // Execute kernel for correctness results
    // =======================================

    // First run to get correct results
    kernel<<<grid_dim, BLAS::block_dim, shared_memory_size, stream>>>(alpha,
                                                                        d_tensor_slice_a,
                                                                        d_tensor_slice_b,
                                                                        beta,
                                                                        d_tensor_c,
                                                                        d_tensor_shift_a,
                                                                        d_tensor_shift_b);
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());
    
    // Copy results to host
    std::vector<OutputType> results (cute::cosize(d_tensor_c.layout()));
    CUDA_CHECK_AND_EXIT(cudaMemcpy(results.data(), d_tensor_c.data().get(), results.size() * sizeof(OutputType), cudaMemcpyDeviceToHost));

    /* ===================================================================== */
    /*                           Performance measurement                      */
    /* ===================================================================== */

    // Warm-up runs
    for (auto warm_up = 0; warm_up < Params::kernel_warm_up_repeats; ++warm_up) {
        kernel<<<grid_dim, BLAS::block_dim, shared_memory_size, stream>>>(alpha,
                                                                          d_tensor_slice_a,
                                                                          d_tensor_slice_b,
                                                                          beta,
                                                                          d_tensor_c,
                                                                          d_tensor_shift_a,
                                                                          d_tensor_shift_b);
    }

    CUDA_CHECK_AND_EXIT(cudaStreamSynchronize(stream));
    CUDA_CHECK_AND_EXIT(cudaEventRecord(start, stream));

    // Performance measurement runs
    for (auto perf_run = 0; perf_run < Params::kernel_repeats; ++perf_run) {
        kernel<<<grid_dim, BLAS::block_dim, shared_memory_size, stream>>>(alpha,
                                                                          d_tensor_slice_a,
                                                                          d_tensor_slice_b,
                                                                          beta,
                                                                          d_tensor_c,
                                                                          d_tensor_shift_a,
                                                                          d_tensor_shift_b);
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
