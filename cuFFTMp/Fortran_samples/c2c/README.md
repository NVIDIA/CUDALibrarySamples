# Fortran C2C Samples 

This sample shows a simple example of complex-to-complex distributed FFT computation, by performing a FORWARD FFT -> element-wise transform -> INVERSE FFT -> correctness verification. It illustrates the meaning of the `CUFFT_XT_INPLACE` and `CUFFT_XT_INPLACE_SHUFFLED` data distribution. The sample also shows how to use openACC as an alternative to cuda kernels to scale device arrays. 

Requirement:
- HPC SDK 21.9 and up
- `mpif90` and `mpicc` should be in your `$PATH`

To build and run:
```
export CUFFT_LIB=/path/to/cufftMp/lib/
export CUFFT_INC=/path/to/cufftMp/include/

cd c2c
make run
[...]

 Hello from rank             0  gpu id            0 size            2
 local_cshape          :          256          256          128
 local_permuted_cshape :          256          128          256
 Hello from rank             1  gpu id            1 size            2
 initial data on             0  max_norm is    0.9999999
 initial data on             1  max_norm is    0.9999999
after C2C forward 1  max_norm is             9671.17089844
    after scaling 1  max_norm is                0.00057645
after C2C forward 0  max_norm is          8389904.00000000
    after scaling 0  max_norm is                0.50007725
after C2C inverse 1  max_norm is                0.99999988  max_diff is      0.00000105
   Relative Linf on rank 1           is                0.00000105
after C2C inverse 0  max_norm is                0.99999988  max_diff is      0.00000105
   Relative Linf on rank 0           is                0.00000105
 >>>> PASSED on rank             1
 >>>> PASSED on rank             0

```
