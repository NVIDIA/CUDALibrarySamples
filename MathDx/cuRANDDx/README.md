# cuRANDDx Library - API Examples

All examples are shipped within [cuRANDDx package](https://developer.nvidia.com/curanddx-downloads).

## Description

This folder demonstrates cuRANDDx APIs usage.

* [cuRANDDx API documentation](https://docs.nvidia.com/cuda/curanddx/index.html)

## Requirements

* [See cuRANDDx requirements](https://docs.nvidia.com/cuda/curanddx/get_started/requirement.html)
* CMake 3.18 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build

* You may specify `CURANDDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to mathDx package (XX.Y - version of the package)

```
mkdir build && cd build
cmake -DCURANDDX_CUDA_ARCHITECTURES=80-real -Dmathdx_ROOT=<path_of_mathdx>/nvidia-mathdx-25.01.0/nvidia/mathdx/25.01 ../example/cuRANDDx
make
// Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/curanddx/index.html) section of the cuRANDDx documentation.

|            Example                |                                  Description                                                      |
|-----------------------------------|---------------------------------------------------------------------------------------------------|
| Introduction Example              | Introduction example (Philox RNG)                                                                 |
| PCG Generator Example             | Generate sequence of 32-bit random values using PCG generator                                     |
| Separate Initialization Kernel    | Initialize RNG states in a separate kernel                                                        |
| Skipping Values                   | Example of using skip methods                                                                     |
| Quasirandom Generator             | Example of using Sobol quasirandom generator                                                      |
| NVRTC Example                     | Using cuRANDDx with NVRTC                                                                         |
