# cuSolverDx Library - API Examples

All examples are shipped within [cuSolverDx package](https://developer.nvidia.com/cusolverdx-downloads), and the github examples could be more frequently updated than the examples shipped in the release packages.  

## Description

This folder demonstrates cuSolverDx APIs usage.

* [cuSolverDx API documentation](https://docs.nvidia.com/cuda/cusolverdx/index.html)

## Requirements

* [See cuSolverDx requirements](https://docs.nvidia.com/cuda/cusolverdx/get_started/requirement.html)
* CMake 3.18 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

* You may specify `CUSOLVERDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to mathDx package (XX.Y - version of the package)

```
mkdir build && cd build
cmake -DCUSOLVERDX_CUDA_ARCHITECTURES=80-real -Dmathdx_ROOT=<path_of_mathdx>/nvidia-mathdx-25.01.0/nvidia/mathdx/25.01 ../example/cusolverdx
make
// Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cusolverdx/examples/index.html) section of the cuSolverDx documentation.
