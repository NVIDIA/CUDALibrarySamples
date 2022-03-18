# Fortran C2C Sample using custom user distributions (pencils)

This sample shows a Complex-to-complex distributed FFT computation using custom user distributions and `cufftXtSetDistribution` API.
It assumes that
- the input data is distributed using a pencil decomposition along Z;
- the output data is distributed using a pencil decomposition along Y.

This example requires 4 GPUs.

Requirement:
- HPC SDK 21.9 and up
- `mpif90` and `mpicc` should be in your `$PATH`

To build and run:
```
export CUFFT_LIB=/path/to/cufftMp/lib/
export CUFFT_INC=/path/to/cufftMp/include/

cd c2c_pencils
make run
[...]
Hello from rank             0  gpu id            0 size            4
Hello from rank             1  gpu id            1 size            4
Hello from rank             2  gpu id            2 size            4
Hello from rank             3  gpu id            3 size            4
           my rank3     lower        32        32         0
           my rank3     upper        64        64        64
           my rank3   strides      2048        64         1
initial data on             3  max_norm is    0.9999951
           my rank2     lower        32         0         0
           my rank2     upper        64        32        64
           my rank2   strides      2048        64         1
initial data on             2  max_norm is    0.9999951
           my rank1     lower         0        32         0
           my rank1     upper        32        64        64
           my rank1   strides      2048        64         1
initial data on             1  max_norm is    0.9999951
local_cshape_in (z,y,x) z fast  :           64           32           32
local_cshape_out (z,y,x) z fast:           32           64           32
           my rank0     lower         0         0         0
           my rank0     upper        32        32        64
           my rank0   strides      2048        64         1
 initial data on             0  max_norm is    0.9999951
[...]
after C2C inverse 2  max_norm is                0.99999511  max_diff is      0.00000066
   Relative Linf on rank 2           is                0.00000066
after C2C inverse 3  max_norm is                0.99999511  max_diff is      0.00000066
   Relative Linf on rank 3           is                0.00000066
after C2C inverse 1  max_norm is                0.99999511  max_diff is      0.00000066
   Relative Linf on rank 1           is                0.00000066
after C2C inverse 0  max_norm is                0.99999511  max_diff is      0.00000066
   Relative Linf on rank 0           is                0.00000066
 >>>> PASSED on rank             3
 >>>> PASSED on rank             1
 >>>> PASSED on rank             2
 >>>> PASSED on rank             0
```
