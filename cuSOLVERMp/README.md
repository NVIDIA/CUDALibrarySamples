# cuSOLVERMp Library API examples

## Description 

Here we provide examples of cuSOLVERMp library API usage

### Key Concepts

Distributed decompositions and linear system solutions

### Examples

[Dense matrix LU factorization and linear system solve](mp_getrf_getrs.c)

[Dense matrix Cholesky factorization and linear system solve](mp_potrf_potrs.c)

[Dense matrix Symmetric Eigensolver](mp_syevd.c)

[Dense matrix Symmetric Generalized Eigensolver](mp_sygvd.c)

[Dense matrix QR factorization](mp_geqrf.c)

[Dense matrix QR factorization and linear system solve](mp_gels.c)

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

[SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

### Documentation

[cuSOLVERMp documentation](https://docs.nvidia.com/cuda/cusolvermp/index.html)

## Usage

### Prerequisites

cuSOLVERMp is distributed through [NVIDIA Developer Zone](https://developer.nvidia.com/cusolvermp) and also as a part of [HPC SDK](https://developer.nvidia.com/hpc-sdk). cuSOLVERMp requires CUDA Toolkit, HPC-X, NCCL and GDRCOPY to be installed on the system. The samples require c++11 compatible compiler. 

### Building

Build examples using `make` command:

`make`

### Running

Run examples with mpi command and number of processes according to process grid values, i.e.

`mpirun -n 2 ./mp_getrf_getrs`

`mpirun -n 2 ./mp_potrf_potrs`

`mpirun -n 2 ./mp_syevd`

`mpirun -n 2 ./mp_sygvd`

`mpirun -n 2 ./mp_geqrf`

`mpirun -n 2 ./mp_gels`
