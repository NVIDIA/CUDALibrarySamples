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
cmake -DCURANDDX_CUDA_ARCHITECTURES=80-real -Dmathdx_ROOT=<path_of_mathdx>/nvidia-mathdx-25.06.0/nvidia/mathdx/25.06 ../example/cuRANDDx
make
// Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/curanddx/index.html) section of the cuRANDDx documentation.

|              Group           |            Example                |                                  Description                                                      |
|------------------------------|-----------------------------------|---------------------------------------------------------------------------------------------------|
| Introduction Examples        | philox_thread_api                 | Introduction example with Philox random number generator                                          |
| Thread API Examples          | xorwow_init_and_generate_thread_api | XORWOW random number generator with initialization and generation                               |
|                              | sobol_thread_api                  | Sobol quasi-random number generator                                                               |
|                              | pcg_thread_api                    | PCG random number generator                                                                       |
|                              | mrg_two_distributions_thread_api  | Multiple random number distributions using MRG32k3a generator                                     |
| NVRTC Examples               | nvrtc_pcg_thread_api              | Using cuRANDDx with NVTRC runtime compilation and nvJitLink runtime linking                       |
