# cuSOLVERMp Library API examples

## Description 

Here we provide examples of cuSOLVERMp library API usage

### Key Concepts

Distributed decompositions and linear system solutions

### Examples

[Dense matrix LU factorization and linear system solve](mp_getrf_getrs.cpp)

[Dense matrix Cholesky factorization and linear system solve](mp_potrf_potrs.cpp)

Examples are bootstrapped by MPI and use it to set up distributed data. Those examples are intended just to show how API is used and not for performance benchmarking. For same reasons process grid is hardcoded to `2x1` in the examples, however you can change it to other values in following lines:
```
/* Define grid of processors */
    const int numRowDevices = 2;
    const int numColDevices = 1;
```

Based on your distributed setup you can choose how your GPU devices are mapped to processes - change following line in the example to suit your needs:
`const int localDeviceId = getLocalRank();`

In these samples each process will use CUDA device ID equal to the local MPI rank ID of the process.

### Supported OSes

Linux

### Supported CPU Architecture

x86_64

### Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus)

[SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

### Documentation

[cuSOLVERMp documentation](https://docs.nvidia.com/hpc-sdk/index.html)

## Usage

### Prerequisites

Samples require c++11 compatible compiler. 
cusolverMp is distributed as a part of [HPC SDK](https://developer.nvidia.com/hpc-sdk) starting with version 21.11 and requires 
HPC SDK to be installed in the system. Also you need to set up `HPCX` environment which is part of `HPC SDK` using one of the provided scripts before building and running examples, i.e.:
```
HPCSDKVER=21.11
HPCSDKARCH=Linux_x86_64
HPCSDKPATH=/opt/nvidia/hpc_sdk
HPCSDKROOT=$HPCSDKPATH/$HPCSDKARCH/$HPCSDKVER
source $HPCSDKROOT/comm_libs/hpcx/latest/hpcx-init-ompi.sh
hpcx_load
```

### Building

Build examples using `make` command:

`make HPCSDKVER=21.11 CUDAVER=11.5 all`

### Running

Run examples with mpi command and number of processes according to process grid values, i.e.

`mpirun -n 2 ./mp_getrf_getrs`

`mpirun -n 2 ./mp_potrf_potrs`
