# cuBLASMp Library API examples

## Description

This folder demonstrates cuBLASMp library API usage. Each sample is a self-contained program that initializes MPI, creates a process grid, and runs a distributed linear algebra operation with performance timing.

### Samples

**Tensor Parallelism Matmul** (communication-overlapped variants):

| Sample | Description |
|--------|-------------|
| [tp_matmul](tp_matmul.cu) | Tensor parallelism example covering AllGather + GEMM and GEMM + ReduceScatter |
| [matmul_ag](matmul_ag.cu) | AllGather + GEMM with configurable data types and scaling |
| [matmul_rs](matmul_rs.cu) | GEMM + ReduceScatter with configurable data types and scaling |
| [matmul_ar](matmul_ar.cu) | GEMM + AllReduce with configurable data types and scaling |

**PBLAS-style operations** (2D block-cyclic distribution):

| Sample | Description |
|--------|-------------|
| [gemm](gemm.cu) | General matrix-matrix multiply (GEMM) |
| [trsm](trsm.cu) | Triangular solve (TRSM) |
| [trmm](trmm.cu) | Triangular matrix-matrix multiply (TRMM) |
| [syrk](syrk.cu) | Symmetric rank-k update (SYRK) |
| [syr2k](syr2k.cu) | Symmetric rank-2k update (SYR2K) |
| [syrkx](syrkx.cu) | Extended symmetric rank-k update (SYRKX) |
| [symm](symm.cu) | Symmetric matrix-matrix multiply (SYMM) |
| [geadd](geadd.cu) | General matrix addition |
| [tradd](tradd.cu) | Triangular matrix addition |
| [gemr2d](gemr2d.cu) | General matrix redistribution between block-cyclic layouts |

### Supported OSes

* Linux

### Supported CPU Architectures

* x86_64
* arm64-sbsa

### Supported Compute Capabilities

* CUDA 12.x
    * [Compute Capability 7.0 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 7.5 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 8.0 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 9.0 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 10.0 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 12.0 ](https://developer.nvidia.com/cuda-gpus)

* CUDA 13.x
    * [Compute Capability 7.5 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 8.0 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 9.0 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 10.0 ](https://developer.nvidia.com/cuda-gpus)
    * [Compute Capability 12.0 ](https://developer.nvidia.com/cuda-gpus)

### Documentation

[cuBLASMp documentation](https://docs.nvidia.com/cuda/cublasmp)

## Usage

### Prerequisites

cuBLASMp is distributed through [NVIDIA Developer Zone](https://developer.nvidia.com/cublasmp-downloads), PyPI ([CUDA 12](https://pypi.org/project/nvidia-cublasmp-cu12/), [CUDA 13](https://pypi.org/project/nvidia-cublasmp-cu13/)), [Conda](https://anaconda.org/nvidia/libcublasmp), [conda-forge](https://anaconda.org/conda-forge/cublasmp) and [HPC SDK](https://developer.nvidia.com/hpc-sdk). cuBLASMp requires CUDA Toolkit and NCCL to be installed on the system. The samples require a C++17 compatible compiler and MPI (HPC-X recommended).

### Build Steps

```bash
git clone https://github.com/NVIDIA/CUDALibrarySamples.git
cd CUDALibrarySamples/cuBLASMp
mkdir build && cd build

export HPCXROOT=<path/to/hpcx>
export CUBLASMP_HOME=<path/to/cublasmp>
export NCCL_HOME=<path/to/nccl>
source ${HPCXROOT}/hpcx-mt-init-ompi.sh
hpcx_load

cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;90;100;120" \
    -DCUBLASMP_INCLUDE_DIRECTORIES=${CUBLASMP_HOME}/include \
    -DCUBLASMP_LIBRARIES=${CUBLASMP_HOME}/lib/libcublasmp.so \
    -DNCCL_INCLUDE_DIRECTORIES=${NCCL_HOME}/include \
    -DNCCL_LIBRARIES=${NCCL_HOME}/lib/libnccl.so
make -j
```

### Running

The number of MPI processes must equal the process grid size (`p * q`). All samples accept `-help` for a full list of options.

**Tensor Parallelism Matmul** (1D grid, processes along one dimension):

```bash
# AllGather + GEMM with FP16
mpirun -n 2 ./matmul_ag -typeA fp16 -typeB fp16 -typeD fp16 -transA t -transB n

# GEMM + ReduceScatter with FP16
mpirun -n 2 ./matmul_rs -typeA fp16 -typeB fp16 -typeD fp16 -transA t -transB n

# GEMM + AllReduce with FP16
mpirun -n 2 ./matmul_ar -typeA fp16 -typeB fp16 -typeD fp16 -transA t -transB n

# End-to-end tensor parallelism
mpirun -n 2 ./tp_matmul
```

**PBLAS-style operations** (2D grid with `-p` rows and `-q` columns):

```bash
mpirun -n 2 ./gemm -p 2 -q 1
mpirun -n 2 ./trsm -p 2 -q 1
mpirun -n 2 ./trmm -p 2 -q 1
mpirun -n 2 ./syrk -p 2 -q 1
mpirun -n 2 ./syr2k -p 2 -q 1
mpirun -n 2 ./syrkx -p 2 -q 1
mpirun -n 2 ./symm -p 2 -q 1
mpirun -n 2 ./geadd -p 2 -q 1
mpirun -n 2 ./tradd -p 2 -q 1
mpirun -n 2 ./gemr2d -p 2 -q 1
```

### Common Options

Individual operations may use only a subset of these options, and not every datatype or scaling-mode combination is valid for every sample.

| Option | Description |
|--------|-------------|
| `-m`, `-n`, `-k` | Matrix dimensions |
| `-mbA`, `-nbA`, `-mbB`, `-nbB`, `-mbC`, `-nbC` | Block sizes for the distributed matrices |
| `-ia`, `-ja`, `-ib`, `-jb`, `-ic`, `-jc` | 1-based starting indices of the operated submatrices |
| `-p`, `-q` | Process grid dimensions (p rows, q columns) |
| `-typeA`, `-typeB`, `-typeC`, `-typeD` | Data types (`fp16`, `bf16`, `fp32`, `fp64`, `fp8_e4m3`, `fp8_e5m2`, `fp4_e2m1`, `cfp32`, `cfp64`) |
| `-transA`, `-transB` | Transpose operations (`n`, `t`, `c`) |
| `-scaleA`, `-scaleB`, `-scaleD`, `-scaleDOut` | Scaling modes (`scalar_fp32`, `vec16_ue4m3`, `vec32_ue8m0`, `outer_vec_fp32`, `vec128_fp32`, `blk128x128_fp32`) |
| `-gridLayout` | Process grid layout (`c` column-major, `r` row-major) |
| `-emulationStrategy` | FP emulation strategy (`default`, `performant`, `eager`) |
| `-checkResult` | Enable result verification (`true` or `false`; default: `true`) |
| `-no-check` | Disable result verification |
| `-cycles` | Number of iterations for timing |
| `-warmup` | Number of warmup iterations |
| `-verbose` | Print detailed output |
| `-help` | Print all available options |

### Matmul Scaling Modes

The `matmul_ag`, `matmul_rs`, and `matmul_ar` samples support both Hopper FP8 scaling modes
(`vec128_fp32`, `blk128x128_fp32`, `outer_vec_fp32`) and Blackwell block scaling modes
(`vec32_ue8m0`, `vec16_ue4m3`). Support for a given datatype and scaling-mode combination depends on the
GPU architecture, CUDA Toolkit, and cuBLASLt support available at runtime.
