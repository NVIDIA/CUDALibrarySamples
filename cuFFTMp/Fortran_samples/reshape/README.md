# Fortran Standalone Reshape Sample

This sample shows how to use the reshape API to re-distribute data accross GPUs.

Requirement:
- HPC SDK 21.9 and up
- `mpif90` and `mpicc` should be in your `$PATH`

To build and run:
```
export CUFFT_LIB=/path/to/cufftMp/lib/
export CUFFT_INC=/path/to/cufftMp/include/
cd reshape
make run

 Hello from rank             0  gpu id            0 size            2
           Input data on rank 0:  0  1  4  5  8  9 12 13
 Expected output data on rank 0:  0  1  2  3  4  5  6  7
 Hello from rank             1  gpu id            1 size            2
           Input data on rank 1:  2  3  6  7 10 11 14 15
 Expected output data on rank 1:  8  9 10 11 12 13 14 15
          Output data on rank 0:  0  1  2  3  4  5  6  7
          Output data on rank 1:  8  9 10 11 12 13 14 15
 >>>> PASSED on rank             1
 >>>> PASSED on rank             0
