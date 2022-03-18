# Fortran R2C_C2R Sample with workarea sharing

This sample shows how to compute a distributed R2C-C2R transform using shared scratch workarea between the two plans.
It is otherwise identical to the other, simpler R2C-C2R sample.

Requirement:
- HPC SDK 21.9 and up
- `mpif90` and `mpicc` should be in your `$PATH`

To build and run:
```
export CUFFT_LIB=/path/to/cufftMp/lib/
export CUFFT_INC=/path/to/cufftMp/include/

cd r2c_c2r_shared_scratch
make run
[...]
 Hello from rank             0  gpu id            0 size            2
 Hello from rank             1  gpu id            1 size            2
 local_rshape          :          258          256          128
 local_permuted_cshape :          129          128          256
 shape of u is           258          256          128
 shape of u_permuted is           129          128          256
[...]
        after C2R 0  max_norm is                1.00000000  max_diff is      0.00000107
   Relative Linf on rank 0           is                0.00000107
        after C2R 1  max_norm is                1.00000000  max_diff is      0.00000107
   Relative Linf on rank 1           is                0.00000107
 >>>> PASSED on rank             0
 >>>> PASSED on rank             1
```
