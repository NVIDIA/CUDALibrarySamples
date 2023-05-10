# cuSPARSE Generic APIs - `Hardware Memory Compression`

## Description

The sample demonstrates how to optimize *sparse vector - dense vector scaling and sum* (`cusparseAxpby`) by exploiting NVIDIA Ampere architecture *Hardware Memory Compression*

[cuSPARSE Optimization Notes](https://docs.nvidia.com/cuda/cusparse/index.html#optimization-notes)

[Nsight Compute](https://developer.nvidia.com/nsight-compute) can be used to understand the effect of the memory compression

```bash
nv-nsight-cu-cli --metrics lts__gcomp_input_sectors_compression_achieved_algo_sdc4to1.sum,lts__gcomp_input_sectors_compression_achieved_algo_sdc4to2.sum,fbpa__dram_read_sectors.sum,fbpa__dram_write_sectors.sum,lts__average_gcomp_input_sector_compression_rate.pct ./compression_example
```

## Building

* Command line
    ```bash
    nvcc -I<cuda_toolkit_path>/include compression_example.c -o compression_example -lcusparse -lcuda
    ```

* Linux
    ```bash
    make
    ```

* Windows/Linux
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
    On Windows, instead of running the last build step, open the Visual Studio Solution that was created and build.

## Support

* **Supported SM Architectures:** SM 8.0, SM 8.6, SM 8.9, SM 9.0
* **Supported OSes:** Linux, Windows, QNX, Android
* **Supported CPU Architectures**: x86_64, ppc64le, arm64
* **Supported Compilers**: gcc, clang, Intel icc, IBM xlc, Microsoft msvc, Nvidia HPC SDK nvc
* **Language**: `C++14`

## Prerequisites

* [CUDA 11.0 toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [CMake 3.9](https://cmake.org/download/) or above on Windows
