# Fortran R2C_C2R Sample 

This sample shows a simple example of R2C/C2R distributed FFT computation, by performing a real-to-complex FFT -> element-wise transform -> complex-to-real FFT -> correctness verification. It illustrates the meaning of the `CUFFT_XT_INPLACE` and `CUFFT_XT_INPLACE_SHUFFLED` data distribution. The sample also shows how to use openACC as an alternative to cuda kernels to scale device arrays.

Requirement:
- HPC SDK 21.9 and up
- `mpif90` and `mpicc` should be in your `$PATH`

To build and run:
```
export CUFFT_LIB=/path/to/cufftMp/lib/
export CUFFT_INC=/path/to/cufftMp/include/

cd r2c_c2r
make run
[...]
Hello from rank             0  gpu id            0 size            2
local_rshape          :          258          256          128
local_permuted_cshape :          129          128          256
shape of u is           258          256          128
shape of u_permuted is           129          128          256
Hello from rank             1  gpu id            1 size            2
shape of u is           258          256          128
shape of u_permuted is           129          128          256
initial data on             1  max_norm is    0.9999999
initial data on             0  max_norm is    0.9999999
   after scaling 1  max_norm is                0.00036766
   after scaling 0  max_norm is                0.49997514
       after C2R 1  max_norm is                1.00000000  max_diff is      0.00000107
  Relative Linf on rank 1           is                0.00000107
       after C2R 0  max_norm is                1.00000000  max_diff is      0.00000107
  Relative Linf on rank 0           is                0.00000107
>>>> PASSED on rank             1
>>>> PASSED on rank             0
```
