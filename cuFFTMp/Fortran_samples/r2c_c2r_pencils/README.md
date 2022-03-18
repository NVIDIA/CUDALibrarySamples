# Fortran R2C/C2R Sample using custom user distributions (pencils)

This sample shows a Complex-to-complex distributed FFT computation using a custom user distribution and `cufftXtSetDistribution` API.
It assumes that
- the input data is distributed using a pencil decomposition along Z;
- the output data is distributed using a pencil decomposition along Z.

This example requires 4 GPUs.

Requirement:
- HPC SDK 21.9 and up
- `mpif90` and `mpicc` should be in your `$PATH`

To build and run:
```
export CUFFT_LIB=/path/to/cufftMp/lib/
export CUFFT_INC=/path/to/cufftMp/include/

cd r2c_c2r_pencils
make run
[...]
 Hello from rank             0  gpu id            0 size            4
 Hello from rank             1  gpu id            1 size            4
 Hello from rank             2  gpu id            2 size            4
 Hello from rank             3  gpu id            3 size            4
[...]
 shape of u is            66           32           32
 shape of u_permuted is            33           32           32
 initial data on             1  max_norm is    0.9999951
 local_rshape_in  :           66           32           32
 local_cshape_out :           33           32           32
   Relative Linf on rank 2           is                0.00000064
        after C2R 3  max_norm is                0.99999487  max_diff is      0.00000064
   Relative Linf on rank 3           is                0.00000064
        after C2R 1  max_norm is                0.99999487  max_diff is      0.00000064
   Relative Linf on rank 1           is                0.00000064
        after C2R 0  max_norm is                0.99999487  max_diff is      0.00000064
   Relative Linf on rank 0           is                0.00000064
 >>>> PASSED on rank             1
 >>>> PASSED on rank             3
 >>>> PASSED on rank             2
 >>>> PASSED on rank             0
```
