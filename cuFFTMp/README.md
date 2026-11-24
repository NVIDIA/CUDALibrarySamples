# cuFFTMp C++ and Fortran Code Samples

## Requirements
- cuFFTMp 12.6.0 or later, from the standalone packages or from the HPC SDK. HPC SDK 26.11 and earlier releases contain an earlier cuFFTMp version.
- CUDA 13.0 and up
- A system with at least two GPUs of compute capability 7.5 or higher (Turing or later).
    - The `c2c_pencils`, `r2c_c2r_pencils`, `r2c_c2r_pencils_cufftMpMakePlan` and `r2c_c2r_slabs_GROMACS` samples require at least 4 GPUs.
- NVSHMEM 3.8.0 or later for the NVSHMEM samples
- NCCL 2.30.7 or later for the NCCL samples (`*_nccl`). These samples do not use NVSHMEM.

Please see the "Hardware and software requirements" sections of the [documentation](https://docs.nvidia.com/cuda/cufftmp/usage/requirements.html) for the full list of requirements.

## Quick start for C++ samples
The following environmental variables need to be defined in order to build and run the samples, for example:
 - `MPI_HOME=/hpc_sdk/Linux_x86_64/.../comm_libs/hpcx/latest/ompi`, the path to your MPI installation and should contain a `lib` and `include` folder
 - `CUFFT_LIB=/hpc_sdk/Linux_x86_64/.../math_libs/lib64/`, where `libcufftMp.so` is located
 - `CUFFT_INC=/hpc_sdk/Linux_x86_64/.../math_libs/include/cufftmp`, where all the cuFFT and cuFFTMp headers files are located
 - `NVSHMEM_LIB=/hpc_sdk/Linux_x86_64/.../comm_libs/nvshmem/lib`, where all the NVSHMEM libraries, such as `libnvshmem_host.so.3`, are located
 - `NVSHMEM_INC=/hpc_sdk/Linux_x86_64/.../comm_libs/nvshmem/include`, where all the NVSHMEM headers, such as `nvshmem.h`, are located
 - `NCCL_LIB=/hpc_sdk/Linux_x86_64/.../comm_libs/nccl/lib`, where `libnccl.so.2` is located (NCCL samples only)
 - `NCCL_INC=/hpc_sdk/Linux_x86_64/.../comm_libs/nccl/include`, where `nccl.h` is located (NCCL samples only)

If you use the standalone cuFFTMp packages instead of the HPC SDK, set `CUFFT_INC` to the `include` directory of the package and `CUFFT_LIB` to the directory of `libcufftMp.so` (for example, `lib` in the tarball). See the installation section of the [documentation](https://docs.nvidia.com/cuda/cufftmp/).

Note that cuFFTMp requires a specific version of NVSHMEM, as indicated [here](https://docs.nvidia.com/cuda/cufftmp/usage/nvshmem_and_cufftmp.html). If HPC SDK contains multiple versions of NVSHMEM, compatible versions are available in `/hpc_sdk/Linux_x86_64/.../lib/compat/` and `/hpc_sdk/Linux_x86_64/.../include/compat/`. Note that NVSHMEM can also be downloaded individually from [here](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html).

If you use the cuFFTMp of the HPC SDK, set `MPI_HOME`, `CUFFT_LIB`, `CUFFT_INC`, `NVSHMEM_LIB` and `NVSHMEM_INC` to directories of the same HPC SDK version.

Then build and run the C2C sample by:
```
$ cd samples/c2c
$ make run
Hello from rank 1/2 using GPU 1 transform of size 16 x 16 x 16, local size 8 x 16 x 16
Hello from rank 0/2 using GPU 0 transform of size 16 x 16 x 16, local size 8 x 16 x 16
GPU data (after first transform): global 3D index [0 0 0], local index 0, rank 0 is (47.305470,11.079344)
[...]
GPU data (after first transform): global 3D index [0 8 9], local index 9, rank 1 is (-63.778786,18.937965)
PASSED with L2 error 1.978186e-07 < 1.000000e-06
```
If you see `PASSED`, the test ran successfully.

The `run` targets of the C++ samples call `mpirun -oversubscribe`. `-oversubscribe` is an Open MPI option. With a different MPI, start the executable with the launcher of that MPI. Use the `LD_LIBRARY_PATH` value and the number of ranks from the `run` target of the sample.

- You can repeat the same procedure for the other samples
  - `samples/c2c_pencils` 
  - `samples/c2c_no_descriptors`
  - `samples/c2c_no_descriptors_cufftMpMakePlan`
  - `samples/r2c_c2r`
  - `samples/r2c_c2r_shared_scratch`
  - `samples/r2c_c2r_pencils`
  - `samples/r2c_c2r_pencils_cufftMpMakePlan`
  - `samples/r2c_c2r_slabs_GROMACS`
  - `samples/r2c_c2r_no_descriptors`
  - `samples/reshape`
- The following samples use the NCCL communication backend (`CUFFT_COMM_NCCL`) instead of NVSHMEM. They need `NCCL_INC` and `NCCL_LIB`, but not `NVSHMEM_INC` and `NVSHMEM_LIB`:
  - `samples/c2c_no_descriptors_nccl`
  - `samples/r2c_c2r_shared_scratch_nccl`
  - `samples/reshape_nccl`

## Fortran samples
A Fortran wrapper library for cuFFTMp is provided in [Fortran_wrappers_nvhpc](Fortran_samples/Fortran_wrappers_nvhpc/) subfolder. The wrapper library will be included in HPC SDK 22.5 and later. 
The Fortran samples can be built and run similarly with `make run` in each of the directories:  
- `Fortran_samples/c2c`
- `Fortran_samples/c2c_no_descriptors`
- `Fortran_samples/c2c_pencils`
- `Fortran_samples/r2c_c2r`
- `Fortran_samples/r2c_c2r_no_descriptors`
- `Fortran_samples/r2c_c2r_pencils`
- `Fortran_samples/r2c_c2r_shared_scratch`
- `Fortran_samples/reshape`

Note: Some Fortran samples that use NVSHMEM APIs to manage memory, including `c2c_no_descriptors`, `r2c_c2r_no_descriptors` and `r2c_c2r_shared_scratch` will fail to build using HPC SDK 23.9. It is a known issue. The workaround is to use a different version of HPC SDK, for example 23.11 or later.

The Fortran wrappers do not support the NCCL backend.

## General tips

### No Infiniband?
This tip applies to the NVSHMEM samples. If the system doesn't have Infiniband, you can use
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

In this case a custom bootstrap library can be built to enable users to use its own MPI implementation. For custom bootstrap plug-ins, please refer to the NVSHMEM documentation [here](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html#using-nvshmem-with-your-mpi-or-openshmem-program) for more details.
