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

Examples are bootstrapped by MPI and use it to set up distributed data. Those examples are intended just to show how API is used and not for performance benchmarking. For same reasons process grid defaults to `2x1` in the examples, however you can change it using `-p` and `-q` command line parameters.

Based on your distributed setup you can choose how your GPU devices are mapped to processes - change following line in the example to suit your needs:
`const int localDeviceId = getLocalRank();`

In these samples each process will use CUDA device ID equal to the local MPI rank ID of the process.

### Supported OSes

* Linux

### Supported CPU Architecture

* x86_64
* arm64-sbsa

### Supported Compute Capabilities

[Compute Capability 7.0 ](https://developer.nvidia.com/cuda-gpus)

[Compute Capability 8.0 ](https://developer.nvidia.com/cuda-gpus)

[Compute Capability 9.0 ](https://developer.nvidia.com/cuda-gpus)

[Compute Capability 10.0 ](https://developer.nvidia.com/cuda-gpus)

[Compute Capability 12.0 ](https://developer.nvidia.com/cuda-gpus)

### Documentation

[cuSOLVERMp documentation](https://docs.nvidia.com/cuda/cusolvermp/index.html)

## Usage

### Prerequisites

cuSOLVERMp is distributed through [NVIDIA Developer Zone](https://developer.nvidia.com/cusolvermp), PyPI ([CUDA 12](https://pypi.org/project/nvidia-cusolvermp-cu12), [CUDA 13](https://pypi.org/project/nvidia-cusolvermp-cu13)), [Conda](https://anaconda.org/nvidia/cusolvermp) and [HPC SDK](https://developer.nvidia.com/hpc-sdk). cuSOLVERMp requires CUDA Toolkit and NCCL to be installed on the system. The samples require C++11 compatible compiler.

### Building

```
git clone https://github.com/NVIDIA/CUDALibrarySamples.git
cd CUDALibrarySamples/cuSOLVERMp
mkdir build
cd build
export HPCXROOT=<path/to/hpcx>
export CUSOLVERMP_HOME=<path/to/cusolvermp>
export NCCL_HOME=<path/to/nccl>
source ${HPCXROOT}/hpcx-mt-init-ompi.sh
hpcx_load
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;90;100;120" -DCUSOLVERMP_INCLUDE_DIRECTORIES=${CUSOLVERMP_HOME}/include -DCUSOLVERMP_LINK_DIRECTORIES=${CUSOLVERMP_HOME}/lib/ -DNCCL_INCLUDE_DIR=${NCCL_HOME}/include -DNCCL_LIBRARIES=${NCCL_HOME}/lib/libnccl.so
make -j
```

### Running

Run examples with mpi command and number of processes according to process grid values, i.e.

`mpirun -n 2 ./mp_getrf_getrs`

`mpirun -n 4 ./mp_getrf_getrs -p 2 -q 2`

`mpirun -n 2 ./mp_potrf_potrs`

`mpirun -n 4 ./mp_potrf_potrs_fp32 -p 2 -q 2 -n 8192 -nbA 1024 -mbA 1024 -nbB 1024 -mbB 1024 -ia 1 -ja 1 -ib 1 -jb 1`

`mpirun -n 4 ./mp_potrf_potrs_fp32emulation -p 2 -q 2 -n 8192 -nbA 1024 -mbA 1024 -nbB 1024 -mbB 1024 -ia 1 -ja 1 -ib 1 -jb 1`

`mpirun -n 2 ./mp_syevd`

`mpirun -n 4 ./mp_syevd -grid_layout C -p 2 -q 2`

`mpirun -n 4 ./mp_syevd -grid_layout R -p 2 -q 2`

`mpirun -n 2 ./mp_sygvd`

`mpirun -n 2 ./mp_geqrf`

`mpirun -n 2 ./mp_gels`
