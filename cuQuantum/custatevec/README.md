# cuStateVec - Samples

# Install

## Linux 

You can use make or cmake to compile the cuStateVec samples. Options for CUSTATEVEC_ROOT can be skipped if cuStateVec is in the CUDA installation folder. 

With make

```
export CUSTATEVEC_ROOT=<path_to_custatevec_root>
make -j8
```

With cmake

```
mkdir build && cd build
cmake .. -DCUSTATEVEC_ROOT=<path_to_custatevec_root>
make -j8
```

# Support

* **Supported SM Architectures:** SM 7.0, SM 7.5, SM 8.0, SM 8.6
* **Supported OSes:** Linux
* **Supported CPU Architectures**: x86_64, arm64
* **Language**: `C++11`

# Prerequisites

* [CUDA 11.4 toolkit](https://developer.nvidia.com/cuda-downloads) (or above) and compatible driver (see [CUDA Driver Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions)).
* [CMake 3.13](https://cmake.org/download/) or above on Windows
