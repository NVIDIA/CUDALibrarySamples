# cuBLASLt Library API examples

## Description
This folder demonstrates cuBLASLt library API usage.

## Samples
- [LtDgemmPresetAlgo](LtDgemmPresetAlgo/)
    
    Sample wrapper executing double precision gemm with a predefined algorithm using cublasLtMatmul, nearly a drop-in
    replacement for cublasDgemm, with addition of the workspace to support split-K algorithms.

- [LtHSHgemmStridedBatchSimple](LtHSHgemmStridedBatchSimple/)

    Sample wrapper executing mixed precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasGemmEx,
    with addition of the workspace to support split-K algorithms.

- [LtIgemmTensor](LtIgemmTensor/)

    Use cublasLtMatmul to perform tensor-op Igemm with memory order transforms on all buffers.

- [LtPlanarComplex](LtPlanarComplex/)

    Use cublasLtMatmul to perform tensor-op Cgemm using planar complex memory layout and half-precision inputs.

- [LtSgemm](LtSgemm/)

    Sample wrapper executing single precision gemm with cublasLtMatmul, nearly a drop-in replacement for cublasSgemm,
    with addition of the workspace to support split-K algorithms.

- [LtSgemmCustomFind](LtSgemmCustomFind/)

    Sample wrapper running through multiple algo and config attributes combination for single precision gemm using cublasLt low-level API.

- [LtSgemmSimpleAutoTuning](LtSgemmSimpleAutoTuning/)

    Sample wrapper executing single precision gemm algorithm auto tuning by querying cublasLt heuristics for best algorithms,
    iterate over the results and pick the algorithm that have the best performance for the given problem.
    
## Supported SM Architectures
[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes
Linux & Windows

## Supported CPU Architecture
x86_64

## Tested Platforms:
 - Windows 10
 - Ubuntu 18.04

## cuBLAS Documentation
[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html)

## Prerequisites
- A Linux/Windows system with NVIDIA driver of version 450.41 and above.
- [CUDA 11.0 toolkit](https://developer.nvidia.com/cuda-downloads).
- CMake 3.10 and above
- Compiler with C++ 11 or above capabilities

## Build Steps
        git clone https://github.com/NVIDIA/CUDALibrarySamples.git
        cd CUDALibrarySamples/cuBLASLt
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j

On Windows, instead of running the last build step, open the Visual Studio Solution that was created and build.