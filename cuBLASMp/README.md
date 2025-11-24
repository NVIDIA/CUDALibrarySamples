# cuBLASMp Library API examples

## Description

This folder demonstrates cuBLASMp library API usage.

### Samples

* [ TP_MATMUL ](tp_matmul.cu)
* [ MATMUL_AG ](matmul_ag.cu)
* [ MATMUL_RS ](matmul_rs.cu)
* [ MATMUL_AR ](matmul_ar.cu)
* [ GEMM ](gemm.cu)
* [ TRSM ](trsm.cu)
* [ SYRK ](syrk.cu)
* [ GEADD ](geadd.cu)
* [ TRADD ](tradd.cu)
* [ GEMR2D ](gemr2d.cu)

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

cuBLASMp is distributed through [NVIDIA Developer Zone](https://developer.nvidia.com/cublasmp-downloads), PyPI ([CUDA 12](https://pypi.org/project/nvidia-cublasmp-cu12/), [CUDA 13](https://pypi.org/project/nvidia-cublasmp-cu13/)), [Conda](https://anaconda.org/nvidia/libcublasmp) and [HPC SDK](https://developer.nvidia.com/hpc-sdk). cuBLASMp requires CUDA Toolkit, NCCL and NVSHMEM to be installed on the system. The samples require C++11 compatible compiler and MPI (used from HPC-X in the Build Steps).

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
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;90;100;120" -DCUBLASMP_INCLUDE_DIRECTORIES=${CUBLASMP_HOME}/include -DCUBLASMP_LIBRARIES=${CUBLASMP_HOME}/lib/libcublasmp.so -DNCCL_INCLUDE_DIRECTORIES=${NCCL_HOME}/include -DNCCL_LIBRARIES=${NCCL_HOME}/lib/libnccl.so -DNVSHMEM_INCLUDE_DIRECTORIES=${NVSHMEM_HOME}/include -DNVSHMEM_HOST_LIBRARIES=${NVSHMEM_HOME}/lib/libnvshmem_host.so -DNVSHMEM_DEVICE_LIBRARIES=${NVSHMEM_HOME}/lib/libnvshmem_device.a
    make -j

### Running

Run examples with mpirun command and number of processes according to process grid values, i.e.

`mpirun -n 2 ./tp_matmul`

`mpirun -n 2 ./matmul_ag -typeA fp16 -typeB fp16 -typeD fp16 -transA t -transB n`

`mpirun -n 2 ./matmul_rs -typeA fp16 -typeB fp16 -typeD fp16 -transA t -transB n`

`mpirun -n 2 ./matmul_ar -typeA fp16 -typeB fp16 -typeD fp16 -transA t -transB n`

`mpirun -n 2 ./gemm -p 2 -q 1`

`mpirun -n 2 ./trsm -p 2 -q 1`

`mpirun -n 2 ./syrk -p 2 -q 1`

`mpirun -n 2 ./geadd -p 2 -q 1`

`mpirun -n 2 ./tradd -p 2 -q 1`

`mpirun -n 2 ./gemr2d -p 2 -q 1`
