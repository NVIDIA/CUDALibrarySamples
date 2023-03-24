# cuFFTMp C++ and Fortran Code Samples

## Requirements
- HPC SDK 23.3 and up
- CUDA 11.0 and up
- A system with at least two Hopper (SM90), Ampere (SM80) or Volta (SM70) GPU.
    - The `c2c_pencils` and `r2c_c2r_pencils` samples require at least 4 GPUs.

Please see the "Hardware and software requirements" sections of the [documentation](https://docs.nvidia.com/hpc-sdk/cufftmp/usage/requirements.html) for the full list of requirements.

## Quick start for C++ samples
The following environmental variables need to be defined in order to build and run the samples, for example:
 - `MPI_HOME=/hpc_sdk/Linux_x86_64/.../comm_libs/hpcx/latest/ompi`, the path to your MPI installation and should contain a `lib` and `include` folder
 - `CUFFT_LIB=/hpc_sdk/Linux_x86_64/.../math_libs/lib64/`, where `libcufftMp.so` is located
 - `CUFFT_INC=/hpc_sdk/Linux_x86_64/.../math_libs/include/cufftmp`, where all the cuFFT and cuFFTMp headers files are located
 - `NVSHMEM_LIB=/hpc_sdk/Linux_x86_64/.../comm_libs/nvshmem/lib`, where all the NVSHMEM libraries, such as `libnvshmem_host.so`, are located
 - `NVSHMEM_LIB=/hpc_sdk/Linux_x86_64/.../comm_libs/nvshmem/include`, where all the NVSHMEM headers, such as `nvshmem.h`, are located

Note that cuFFTMp requires a specific version of NVSHMEM, as indicated [here](https://docs.nvidia.com/hpc-sdk/cufftmp/usage/nvshmem_and_cufftmp.html). If HPC SDK contains multiple versions of NVSHMEM, compatible versions are available in `/hpc_sdk/Linux_x86_64/.../lib/compat/` and `/hpc_sdk/Linux_x86_64/.../include/compat/`. Note that NVSHMEM can also be downloaded individually from [here](https://docs.nvidia.com/nvshmem/install-guide/index.html).

As cuFFTMp is released in HPC SDK 22.3 and up, to build and run the samples or your applications with cuFFTMp it is highly recommended to have $MPI_HOME, $CUFFT_LIB, $CUFFT_INC, and $NVSHMEM_LIB all pointing to the same HPC SDK version.

If you have to use an older version of HPC SDK (21.9 or 21.11), you can find the early-access version of cuFFTMp in [cuFFTMP EA](https://developer.nvidia.com/cudamathlibraryea). 

Then build and run the C2C sample by:
```
$ cd samples/c2c
$ make run
Hello from rank 1/2 using GPU 1 transform of size 16 x 16 x 16, local size 8 x 16 x 16
Hello from rank 0/2 using GPU 0 transform of size 16 x 16 x 16, local size 8 x 16 x 16
Shuffled (Y-Slabs) GPU data, global 3D index [0 8 0], local index 0, rank 1 is (-13.323235,-48.004234)
[...]
Shuffled (Y-Slabs) GPU data, global 3D index [0 0 9], local index 9, rank 0 is (15.618601,-9.228624)
Relative Linf error on rank 0, 3.226381e-07
Relative Linf error on rank 1, 3.109569e-07
PASSED on rank 1
PASSED on rank 0
```
If you see `PASSED`, the test ran successfully.

- You can repeat the same procedure for the other samples
  - `samples/c2c_pencils` 
  - `samples/c2c_no_descriptors`
  - `samples/r2c_c2r`
  - `samples/r2c_c2r_shared_scratch`
  - `samples/r2c_c2r_pencils`
  - `samples/r2c_c2r_no_descriptors`
  - `samples/reshape`

## Fortran samples
A Fortran wrapper library for cuFFTMp is provided in [Fortran_wrappers_nvhpc](Fortran_samples/Fortran_wrappers_nvhpc/) subfolder. The wrapper library will be included in HPC SDK 22.5 and later. 
The Fortran samples can be built and run similarly with `make run` in each of the directories:  
- `Fortran_samples/c2c`
- `Fortran_samples/c2c_pencils`
- `Fortran_samples/r2c_c2r`
- `Fortran_samples/r2c_c2r_shared_scratch`
- `Fortran_samples/r2c_c2r_pencils`
- `Fortran_samples/reshape`

## General tips

### No Infiniband?
Those samples use NVSHMEM. If the system doesn't have Infiniband, you can use
```
NVSHMEM_REMOTE_TRANSPORT=none
```
to avoid Infiniband initialization-related errors. This will then fallback to p2p (single-node) only.

### MPI non compatible with provided bootstrap in HPC SDK?
In case a custom MPI, other than the MPI implementations provided in HPC SDK, is used, the bootstrapping plugin may fail with an error such as
```
src/bootstrap/bootstrap_loader.cpp:46: NULL value Bootstrap unable to load 'nvshmem_bootstrap_mpi.so'
libmpi.so.40: cannot open shared object file: No such file or directory
src/bootstrap/bootstrap.cpp:26: non-zero status: -1 bootstrap_loader_init returned error
src/init/init.cpp:90: non-zero status: 7 bootstrap_init failed
src/init/init.cpp:501: non-zero status: 7 nvshmem_bootstrap failed
```
indicating it cannot load `libmpi.so.40`, most likely because a non-compatible version of MPI is used to link with the nvshmem bootstrapping library. 

In this case a custom bootstrap library can be built to enable users to use its own MPI implementation. We include an [extra_bootstraps](extra_bootstraps/) folder in the samples to help creating the custom bootstrap library. Find more information at the "Bootstrapping Mechanism" session of the [documentation](https://docs.nvidia.com/hpc-sdk/cufftmp).


### Container
HPC SDK containers contain all the required dependencies. For instance,
```
docker pull nvcr.io/nvidia/nvhpc:23.3-devel-cuda_multi-ubuntu20.04
```
