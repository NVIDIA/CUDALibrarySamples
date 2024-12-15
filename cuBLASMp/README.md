# cuBLASMp Library API examples

## Description 

This folder demonstrates cuBLASLMp library API usage.

### Samples

* [ PMATMUL ](pmatmul.cu)

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

### Supported SM Architectures

* [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)

* [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

* [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

### Documentation

[cuBLASMp documentation](https://docs.nvidia.com/cuda/cublasmp)

## Usage

### Prerequisites

cuBLASMp is distributed through [NVIDIA Developer Zone](https://developer.nvidia.com/cublasmp-downloads) and also as a part of [HPC SDK](https://developer.nvidia.com/hpc-sdk). cuBLASMp requires CUDA Toolkit, HPC-X, NVSHMEM, NCCL and GDRCOPY to be installed on the system. The samples require C++11 compatible compiler. 

### Build Steps

    git clone https://github.com/NVIDIA/CUDALibrarySamples.git
    cd CUDALibrarySamples/cuBLASMp
    mkdir build
    cd build
    export HPCXROOT=<path/to/hpcx>
    export CUBLASMP_HOME=<path/to/cublasmp>
    export CAL_HOME=<path/to/libcal>
    export NVSHMEM_HOME=<path/to/nvshmem>
    source ${HPCXROOT}/hpcx-mt-init-ompi.sh
    hpcx_load
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="70;80;90" -DCUBLASMP_INCLUDE_DIRECTORIES=${CUBLASMP_HOME}/include -DCUBLASMP_LIBRARIES=${CUBLASMP_HOME}/lib/libcublasmp.so -DCAL_INCLUDE_DIRECTORIES=${CAL_HOME}/include -DCAL_LIBRARIES=${CAL_HOME}/lib/libcal.so -DNVSHMEM_INCLUDE_DIRECTORIES=${NVSHMEM_HOME}/include -DNVSHMEM_HOST_LIBRARIES=${NVSHMEM_HOME}/lib/libnvshmem_host.so -DNVSHMEM_DEVICE_LIBRARIES=${NVSHMEM_HOME}/lib/libnvshmem_device.a
    make -j

### Running

Run examples with mpi command and number of processes according to process grid values, i.e.

`mpirun -n 2 ./pmatmul`

`mpirun -n 2 ./pgemm`

`mpirun -n 2 ./ptrsm`

`mpirun -n 2 ./psyrk`

`mpirun -n 2 ./pgeadd`

`mpirun -n 2 ./ptradd`
