# cuRANDDx Library - API Examples

All examples are shipped within [cuRANDDx package](https://developer.nvidia.com/curanddx-downloads).

## Description

This folder demonstrates cuRANDDx APIs usage.

* [cuRANDDx API documentation](https://docs.nvidia.com/cuda/curanddx/index.html)

## Requirements

* [See cuRANDDx requirements](https://docs.nvidia.com/cuda/curanddx/get_started/requirement.html)
* CMake 3.23 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Turing (SM75) or newer architecture

## Build

* You may specify `CURANDDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to mathDx package (XX.Y - version of the package)

```
mkdir build && cd build
cmake -DCURANDDX_CUDA_ARCHITECTURES=80-real -Dmathdx_ROOT=<path_of_mathdx>/mathdx/XX.Y ..
make
# Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/curanddx/index.html) section of the cuRANDDx documentation.

| Group                 | Example                             | Description                                                                                          |
| --------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Introduction Examples | philox_thread_api                   | Introduction example explaining the basics of cuRANDDx using the Philox generator                    |
| Thread API Examples   | pcg_thread_api                      | Use the PCG generator to create 32-bit random numbers matching NVPL RAND with strict ordering        |
|                       | philox_random_bits_thread_api       | Use the Philox generator to generate a sequence of random bits                                       |
|                       | xorwow_init_and_generate_thread_api | Set up generator states first, then generate random numbers in later kernels                         |
|                       | mrg_two_distributions_thread_api    | Generate two sequences of different distributions in a single kernel using skip functions (MRG32k3a) |
|                       | sobol_thread_api                    | Generate quasirandom numbers using the 64-bit scrambled Sobol generator                              |
| NVRTC Examples        | nvrtc_pcg_thread_api                | Use cuRANDDx with NVRTC runtime compilation                                                          |
