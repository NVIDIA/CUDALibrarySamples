#cuBLASDx DGEMM Emulation using Ozaki Scheme

This example demonstrates how to emulate double precision GEMM(DGEMM) operations using multiple lower precision GEMM operations through the **Ozaki scheme**. This technique allows achieving double precision accuracy while leveraging the performance benefits of lower precision tensor operations.

The code closely follows the implementation described in "DGEMM on Integer Matrix Multiplication Unit" and introduces a more efficient slicing algorithm. Our approach uses unsigned magnitudes for the slices and encodes the sign of the double-precision values in the leading slice, thereby saving (slices-1) bits of storage compared to the reference implementation and reducing the total number of operations required to compute the matrix product.

## Overview

The Ozaki scheme is a numerical method that decomposes high-precision floating-point numbers into multiple lower-precision components, performs computations on these components, and then reconstructs the high-precision result. This example specifically:

- **Input**: Double precision matrices (FP64)
- **Computation**: Multiple int8 GEMM operations using cuBLASDx
- **Output**: Double precision result with high accuracy
- **Comparison**: Validates against native cuBLAS DGEMM

## Mathematical Foundation

### The Ozaki Decomposition

For double precision values `a` and `b`, the Ozaki scheme represents them as:

```
a = Σ(i=0 to slices-1) a_i * 2^(shift_a - i*8)
b = Σ(j=0 to slices-1) b_j * 2^(shift_b - j*8)
```

Where:
- `a_i, b_j` are int8 slice values
- `shift_a, shift_b` are scaling factors determined by the maximum values in each row/column
- `8` represents the number of bits per int8 slice

### Matrix Multiplication

The product `a * b` becomes:
```
a * b = ΣΣ a_i * b_j * 2^(shift_a + shift_b - (i+j)*8)
```

This allows computing the double precision result using multiple int8 GEMM operations with appropriate scaling.

## Algorithm Steps

### 1. Preprocessing
- **Purpose**: Determine optimal scaling factors
- **Operation**: Find maximum absolute value in each row of A and each column of B
- **Output**: Exponent shifts for optimal int8 representation
- **Kernel**: `max_reduce_kernel`

### 2. Slicing  
- **Purpose**: Decompose FP64 values into int8 slices
- **Operation**: Convert each double precision element into multiple int8 components
- **Output**: Slice tensors `[slices, rows, cols]` for both A and B matrices
- **Kernel**: `slice_kernel`

### 3. Matrix Multiplication
- **Purpose**: Compute products of slice combinations
- **Operation**: Diagonal iteration over slice pairs with cuBLASDx GEMM
- **Pattern**: 
  - Diagonal 0: `A_slice[0] * B_slice[0]`
  - Diagonal 1: `A_slice[0] * B_slice[1] + A_slice[1] * B_slice[0]`  
  - Diagonal 2: `A_slice[0] * B_slice[2] + A_slice[1] * B_slice[1] + A_slice[2] * B_slice[0]`
  - etc.
- **Kernel**: `fused_epilogue_kernel`

### 4. Reconstruction
- **Purpose**: Combine slice results back to double precision
- **Operation**: Scale and accumulate results with proper powers of 2
- **Output**: Final FP64 result matrix

## Implementation Highlights

### Memory Efficiency
- **2-stage pipelining**: Overlaps data movement with computation
- **Shared memory optimization**: Efficient tile loading and reuse
- **Register blocking**: Maximizes computational intensity

### Precision Management
- **Adaptive scaling**: Per-row/column scaling factors for optimal precision
- **Numerical stability**: Reverse diagonal order for proper accumulation
- **Error minimization**: Careful reconstruction to maintain FP64 accuracy

### Performance Optimization
- **Tensor Core utilization**: int8 operations on modern GPUs
- **Memory coalescing**: Optimized memory access patterns
- **Tile configuration**: Configurable tile sizes for different problem sizes
- **Adaptive swizzled layouts**: Enabling LDSM and reducing shared memory bank conflicts

## Configuration Options

### Ozaki Scheme Parameters
```cpp
constexpr int slices = 7;  // Number of slices (more = higher precision, more computation)
```

    ## #Tile Configuration
```cpp

    using tile_shape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>;
    using cta_shape  = cute::Shape<cute::Int<128>, cute::Int<1>, cute::Int<1>>;
```

    ## #Problem Sizes
```cpp std::vector<problem_shape>
        problems = {
            {1024, 1024, 1024}, // Small problem for quick testing
            {2048, 2048, 2048}  // Larger problem for performance evaluation
};
```

## Building and Running

### Prerequisites
- CUDA toolkit (12.0 or later)
- cuBLASDx library
- GPU with Tensor Core support (recommended)

## Performance Analysis

### Metrics Reported
- **Preprocessing time**: Time for scaling factor computation
- **Slicing time**: Time for FP64 to int8 decomposition  
- **Matrix multiplication time**: Core GEMM computation time
- **End-to-end time**: Total emulation time including all steps
- **TFLOPS**: Throughput in tera floating-point operations per second
- **Error analysis**: Relative error compared to native cuBLAS


## Educational Value

### Key Learning Points
1. **Numerical precision techniques**: Understanding how high precision can be achieved with lower precision operations
2. **Memory hierarchy optimization**: Efficient use of shared memory and register files
3. **Kernel fusion**: Combining multiple operations into single kernels for better performance  
4. **Error analysis**: Quantifying and managing numerical errors in iterative algorithms
5. **cuBLASDx usage**: Advanced techniques for tensor core programming

### Code Organization
- `dgemm_emulation.cu`: Main orchestration and performance comparison
- `emulation_kernels.hpp`: CUDA kernel implementations  
- `slicing.hpp`: Utility functions for FP64 ↔ int8 conversion
- `debug_printer.hpp`: Debugging and visualization utilities

## Limitations and Considerations

### Current Limitations
- **Square matrices only**: Educational implementation supports M=N=K only
- **Memory overhead**: Requires storage for all slice combinations
- **Compute overhead**: Multiple GEMM operations vs single native GEMM

### Performance Considerations
- **Slice count trade-off**: More slices = higher accuracy but more computation
- **Tile size optimization**: Different optimal configurations for different problem sizes
- **Memory bandwidth**: May be memory-bound for smaller problems

### Numerical Considerations  
- **Scaling factor quality**: Critical for maintaining precision
- **Accumulation order**: Affects final numerical accuracy
- **Problem conditioning**: Some matrices may be challenging to decompose effectively

## Future Extensions

This example is under active development and will be extended with performance optimizations and new features.

## References

- Ootomo, H., Ozaki, K., and Yokota, R. "DGEMM on Integer Matrix Multiplication Unit." [arXiv:2306.11975](https://arxiv.org/pdf/2306.11975), 2023.
- cuBLASDx Documentation: [NVIDIA cuBLASDx Library](https://docs.nvidia.com/cuda/cublasdx/)
- Tensor Core Programming Guide: [NVIDIA Tensor Cores](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-operations)
