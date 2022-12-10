# cuSPARSE APIs - `cusparseSgpsvInterleavedBatch`

## Description

The example solves two penta-diagonal systems and assumes data layout is NOT interleaved format. Before calling `gpsvInterleavedBatch`, `cublasXgeam` is used to transform the data layout, from aggregate format to interleaved format. If the user can prepare interleaved format, no need to transpose the data.

[cusparseSgpsvInterleavedBatch Documentation](https://docs.nvidia.com/cuda/cusparse/index.html#gpsvInterleavedBatch)

## Building

* Command line
    ```bash
    nvcc -I<cuda_toolkit_path>/include gpsvInterleavedBatch_example.c -o gpsvInterleavedBatch_example -lcusparse
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

* **Supported SM Architectures:** SM 3.5, SM 3.7, SM 5.0, SM 5.2, SM 5.3, SM 6.0, SM 6.1, SM 6.2, SM 7.0, SM 7.2, SM 7.5, SM 8.0, SM 8.6, SM 8.9, SM 9.0
* **Supported OSes:** Linux, Windows, QNX, Android
* **Supported CPU Architectures**: x86_64, ppc64le, arm64
* **Supported Compilers**: gcc, clang, Intel icc, IBM xlc, Microsoft msvc, Nvidia HPC SDK nvc
* **Language**: `C99`

## Prerequisites

* [CUDA 11.0 toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [CMake 3.9](https://cmake.org/download/) or above on Windows

## Usage

```
$  ./gpsvInterleavedBatch_example
```

Sample output:

```
bufferSize = 64
==== x1 = inv(A1)*b1
x1[0] = -0.059152
x1[1] = 0.342773
x1[2] = -0.129464
x1[3] = 0.198242
|b1 - A1*x1| = 7.152557E-07

==== x2 = inv(A2)*b2
x2[0] = -0.001203
x2[1] = 0.279155
x2[2] = -0.041607
x2[3] = 0.089761
|b2 - A2*x2| = 4.768372E-07
```
