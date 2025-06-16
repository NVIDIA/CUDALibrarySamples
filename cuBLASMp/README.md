# cuBLASMp Library API examples

## Description

This folder demonstrates cuBLASMp library API usage.

### Samples

* [ PMATMUL ](pmatmul.cu)

* [ PMATMUL_AR ](pmatmul_ar.cu)

* [ PGEMM ](pgemm.cu)

* [ PTRSM ](ptrsm.cu)

* [ PSYRK ](psyrk.cu)

* [ PGEADD ](pgeadd.cu)

* [ PTRADD ](ptradd.cu)

### Supported OSes

* Linux

### Supported CPU Architecture

* x86_64
* arm64-sbsa

### Supported Compute Capabilities

* [Compute Capability 7.0 ](https://developer.nvidia.com/cuda-gpus)

* [Compute Capability 8.0 ](https://developer.nvidia.com/cuda-gpus)

* [Compute Capability 9.0 ](https://developer.nvidia.com/cuda-gpus)

* [Compute Capability 10.0 ](https://developer.nvidia.com/cuda-gpus)

* [Compute Capability 12.0 ](https://developer.nvidia.com/cuda-gpus)

### Documentation

[cuBLASMp documentation](https://docs.nvidia.com/cuda/cublasmp)

## Usage

### Prerequisites

cuBLASMp is distributed through [NVIDIA Developer Zone](https://developer.nvidia.com/cublasmp-downloads), [PyPI](https://pypi.org/project/nvidia-cublasmp-cu12/), [Conda](https://anaconda.org/nvidia/libcublasmp) and [HPC SDK](https://developer.nvidia.com/hpc-sdk). cuBLASMp requires CUDA Toolkit, NCCL and NVSHMEM to be installed on the system. The samples require C++11 compatible compiler and MPI (used from HPC-X in the Build Steps).

### Build Steps

    git clone https://github.com/NVIDIA/CUDALibrarySamples.git
    cd CUDALibrarySamples/cuBLASMp
    mkdir build
    cd build
    export HPCXROOT=<path/to/hpcx>
    export CUBLASMP_HOME=<path/to/cublasmp>
    export NCCL_HOME=<path/to/nccl>
    export NVSHMEM_HOME=<path/to/nvshmem>
    source ${HPCXROOT}/hpcx-mt-init-ompi.sh
    hpcx_load
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="70;80;90;100;120" -DCUBLASMP_INCLUDE_DIRECTORIES=${CUBLASMP_HOME}/include -DCUBLASMP_LIBRARIES=${CUBLASMP_HOME}/lib/libcublasmp.so -DNCCL_INCLUDE_DIRECTORIES=${NCCL_HOME}/include -DNCCL_LIBRARIES=${NCCL_HOME}/lib/libnccl.so -DNVSHMEM_INCLUDE_DIRECTORIES=${NVSHMEM_HOME}/include -DNVSHMEM_HOST_LIBRARIES=${NVSHMEM_HOME}/lib/libnvshmem_host.so -DNVSHMEM_DEVICE_LIBRARIES=${NVSHMEM_HOME}/lib/libnvshmem_device.a
    make -j

### Running

Run examples with mpirun command and number of processes according to process grid values, i.e.

`mpirun -n 2 ./pmatmul`

`mpirun -n 2 ./pmatmul_ar`

`mpirun -n 2 ./pgemm`

`mpirun -n 2 ./ptrsm`

`mpirun -n 2 ./psyrk`

`mpirun -n 2 ./pgeadd`

`mpirun -n 2 ./ptradd`
