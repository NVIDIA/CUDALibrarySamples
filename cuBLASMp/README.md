# cuBLASMp Library API examples

## Description 

This folder demonstrates cuBLASLMp library API usage.

### Samples

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

cuBLASMp is distributed through [NVIDIA Developer Zone](https://developer.nvidia.com/cublasmp-downloads) and also as a part of [HPC SDK](https://developer.nvidia.com/hpc-sdk). cuBLASMp requires CUDA Toolkit, HPC-X, NCCL and GDRCOPY to be installed on the system. The samples require C++11 compatible compiler. 

### Build Steps

    git clone https://github.com/NVIDIA/CUDALibrarySamples.git
    cd CUDALibrarySamples/cuBLASMp
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j

### Running

Run examples with mpi command and number of processes according to process grid values, i.e.

`mpirun -n 2 ./pgemm`

`mpirun -n 2 ./ptrsm`

`mpirun -n 2 ./psyrk`

`mpirun -n 2 ./pgeadd`

`mpirun -n 2 ./ptradd`
