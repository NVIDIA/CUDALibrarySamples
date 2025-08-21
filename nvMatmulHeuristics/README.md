# nvMatmulHeuristics

This directory contains samples demonstrating the usage of nvMatmulHeuristics.

More informations on nvMatmulHeuristics can be found here:

- https://docs.nvidia.com/cuda/nvidia-matmul-heuristics 
- https://developer.nvidia.com/nvidia-matmul-heuristics-downloads

## C++ Samples

### How to build

To build any of the C++ sample (replace `${SAMPLE}` by the appropriate sample name)

```bash
g++ -std=c++17 ${SAMPLE}.cpp -o ${SAMPLE} -lnvMatmulHeuristics -I${NVMMH_HOME}/include -L${NVMMH_HOME}/lib
```

where `NVMMH_HOME` points to the root of the nvMatmulHeuristics installation.

All C++ samples can be run as-is without any arguments.

### 1 - GEMM Heuristics

Located in `1_gemm_heuristics.cpp`.
Shows how to use nvMatmulHeuristics to get the top-8 GEMM kernel configurations. 
Note that `runtime` refers to the estimated GEMM kernel runtime, not the heuristics runtime.

Sample output:

```bash
nvMatmulHeuristics version 0.1.0

[1] CTA: 128 128 64, Warp: 64 64 64, Instr: 16 8 16, Stages: 5, SplitK: 108, Raster: 0, SwizzleFactor: 1, estimated GEMM runtime: 0.264863 ms
[2] CTA: 128 128 32, Warp: 64 64 32, Instr: 16 8 16, Stages: 8, SplitK: 108, Raster: 0, SwizzleFactor: 1, estimated GEMM runtime: 0.264863 ms
[3] CTA: 128 128 32, Warp: 32 32 32, Instr: 16 8 16, Stages: 8, SplitK: 108, Raster: 0, SwizzleFactor: 1, estimated GEMM runtime: 0.264863 ms
[4] CTA: 128 64 64, Warp: 32 64 64, Instr: 16 8 16, Stages: 6, SplitK: 54, Raster: 0, SwizzleFactor: 1, estimated GEMM runtime: 0.246302 ms
[5] CTA: 128 64 32, Warp: 64 32 32, Instr: 16 8 16, Stages: 8, SplitK: 54, Raster: 0, SwizzleFactor: 1, estimated GEMM runtime: 0.246302 ms
[6] CTA: 64 64 64, Warp: 32 32 64, Instr: 16 8 16, Stages: 8, SplitK: 27, Raster: 0, SwizzleFactor: 1, estimated GEMM runtime: 0.239468 ms
[7] CTA: 256 128 64, Warp: 64 64 64, Instr: 16 8 16, Stages: 3, SplitK: 108, Raster: 0, SwizzleFactor: 1, estimated GEMM runtime: 0.407087 ms
[8] CTA: 256 128 32, Warp: 64 64 32, Instr: 16 8 16, Stages: 6, SplitK: 108, Raster: 0, SwizzleFactor: 1, estimated GEMM runtime: 0.407087 ms
```

### 2 - Discovery

Located in `2_discovery.cpp`.
Shows how to use the discovery process in nvMatmulHeuristics. 
Kernel benchmarking is not present in this sample and is replaced by a placeholder querying the nvMatmulHeuristics timing model instead.

Sample output:

```bash
nvMatmulHeuristics version 0.1.0

[1] Problem: M: 2272, N: 16, K: 16, layout: NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR, Config: CTA: 16 16 16, Warp: 16 16 16, Instr: 16 8 16, Stages: 1, SplitK: 1, Raster: 0, SwizzleFactor: 1
[2] Problem: M: 18080, N: 18080, K: 16, layout: NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR, Config: CTA: 128 128 16, Warp: 64 64 16, Instr: 16 8 16, Stages: 1, SplitK: 1, Raster: 0, SwizzleFactor: 1
[3] Problem: M: 18080, N: 18064, K: 16, layout: NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR, Config: CTA: 128 128 16, Warp: 64 64 16, Instr: 16 8 16, Stages: 1, SplitK: 1, Raster: 0, SwizzleFactor: 1
[4] Problem: M: 18080, N: 16, K: 18080, layout: NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR, Config: CTA: 128 16 128, Warp: 32 16 128, Instr: 16 8 16, Stages: 2, SplitK: 1, Raster: 0, SwizzleFactor: 1
...
```

### 3 - Energy Discovery

Located in `3_energy_discovery.cpp`. 
Demonstrates how to use nvMatmulHeuristics for energy-aware kernel discovery and selection.

Sample output:

```bash
nvMatmulHeuristics version 0.1.0

Estimated Energy Before: 0
[1] Problem: M: 512, N: 6912, K: 64, layout: NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR, Config: CTA: 256 128 64, Warp: 64 64 32, Instr: 16 8 16, Stages: 5, SplitK: 1, Raster: 1, SwizzleFactor: 1
...
[43] Problem: M: 32768, N: 32768, K: 32768, layout: NVMMH_MATMUL_LAYOUT_TN_ROW_MAJOR, Config: CTA: 128 256 64, Warp: 64 64 32, Instr: 16 8 16, Stages: 5, SplitK: 1, Raster: 0, SwizzleFactor: 1
Estimated Energy After: 6.45915
```

### 4 - Kernel Runtime Estimation

Located in `4_runtime_estimation.cpp`
Shows how to use nvMatmulHeuristics to estimate the runtime of a GEMM kernel configuration. This sample demonstrates:

- Creating a matmul problem
- Getting a kernel configuration using heuristics
- Estimating the runtime for the configuration
- Displaying detailed kernel configuration information

Sample output:

```bash
Problem: M=1024, N=2048, K=4096, Layout=NN
Kernel Configuration:
  CTA: 128x128x32
  Warp: 64x64x32
  Instr: 16x8x16
  SplitK: 1
  Load Stages: 8
  Grid Swizzle: 1
  CTA Order: 0
  Cluster Config: 1x1
Estimated Runtime: 0.000046 seconds
```

## Python Samples

### How to install

Install nvMatmulHeuristics through

```bash
pip install nvidia-matmul-heuristics
```

### 5 - GEMM configurations

Demonstrates how to use nvMatmulHeuristics to get GEMM configurations for a given problem size and hardware. 
This sample shows the basic usage of the public API.

Usage:

```bash
python3 5_get_configs.py -M 1024 -N 1024 -K 1024 --gpu A100_SXM_80GB
```

Sample output

```
Getting 8 configurations for:
Problem size: M=1024, N=1024, K=1024
Batch size: 1
GPU: A100_SXM_80GB
Layout: NN_ROW
Backend: CUTLASS3
Precision: HSH

Successfully loaded internal discovery set.
Found 8 configurations:

Configuration 1:
  Kernel: layout(NN_ROW) stages(6) cta(128 80 64) warp(32 40 64) instr(16 8 16) splitK(1) swizz(1) ctaOrder(0) cluster(1 1)
  Estimated runtime: 0.020084 ms
...
Configuration 8:
  Kernel: layout(NN_ROW) stages(2) cta(64 256 128) warp(64 64 128) instr(16 8 16) splitK(1) swizz(1) ctaOrder(0) cluster(1 1)
  Estimated runtime: 0.025779 ms
```

### 6 - GEMM configurations with extended interface

Demonstrates the usage of the extended interface (`NvMatmulHeuristicsInterfaceEx`) which provides additional features like automatic discovery set loading and per-call precision specification. 
This example shows how to track implicitly loaded discovery sets and use the simplified GPU-based initialization. 
`HSH` below indicates FP16 input, FP16 output and FP32 accumulation.

Usage:

```bash
python3 6_get_configs_ex.py -M 1024 -N 1024 -K 1024 --layout NN_ROW_MAJOR --gpu H100_SXM --precision HSH
```

Key features demonstrated:

- Automatic hardware descriptor management
- Implicit discovery set loading with tracking
- Per-call precision specification
- Simplified GPU-based initialization

Sample output:

```bash
Getting 8 configurations for:
Problem size: M=1024, N=1024, K=1024
Batch size: 1
GPU: H100_SXM
Layout: NN_ROW
Backend: CUTLASS3
Precision: HSH
Auto-load discovery sets: True


Getting configurations with precision (HSH)...

Implicitly loaded discovery sets:
  - Target: CUTLASS3, Precision: HSH, Layout: NN_ROW

Found 8 configurations with HSH precision:

Configuration 1:
  Kernel: layout(NN_ROW) stages(4) cta(128 64 128) warp(64 32 128) instr(64 8 16) splitK(1) swizz(1) ctaOrder(0) cluster(1 2)
  Estimated runtime: 0.017216 ms
...
Configuration 8:
  Kernel: layout(NN_ROW) stages(8) cta(64 128 64) warp(64 32 32) instr(64 8 16) splitK(1) swizz(1) ctaOrder(0) cluster(2 1)
  Estimated runtime: 0.017216 ms
```

### 7 - SMEM carveout

Demonstrates how to use the SMEM carveout feature to reserve a portion of shared memory for other purposes. This example compares kernel configurations with and without SMEM carveout to show how it
affects the kernel parameters.

Usage:

```bash
python3 7_smem_carveout.py -M 2560 -N 2560 -K 2560 --layout NN_ROW_MAJOR --gpu H100_SXM --smem-carveout 60000
```

Example output:

```
Comparing configurations with and without SMEM carveout:
Problem size: M=2560, N=2560, K=2560
GPU: H100_SXM
Layout: NN_ROW
Backend: CUTLASS3
Precision: HSH
SMEM Carveout: 60000 bytes


Configuration without SMEM carveout:
  CTA Tile: 256x128x64
  Warp Tile: 64x64x32
  Instruction Tile: 64x8x16
  Split K: 1
  Stages: 4
  CTA swizzling: 1
  CTA Order: 0
  Cluster Config: 1x2
  Estimated Runtime: 0.095070 ms

Configuration with SMEM carveout:
  CTA Tile: 256x128x64
  Warp Tile: 64x64x32
  Instruction Tile: 64x8x16
  Split K: 1
  Stages: 3
  CTA swizzling: 1
  CTA Order: 0
  Cluster Config: 1x2
  Estimated Runtime: 0.095070 ms
```

In this example, we can see how the SMEM carveout affects the kernel configuration: the number of stages decrease from 4 to 2 to compensate.
The estimated runtime remains the same.
