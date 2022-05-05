# R2C_C2R Sample using a custom user distributions (pencils)

This sample shows how to compute a distributed R2C-C2R transform with custom data distribution.
It explains how data is distributed according to `CUFFT_XT_FORMAT_DISTRIBUTED_INPUT` and `CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT`.

This example requires 4 GPUs.

To build and run:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ make run
Hello from rank 1/4 using GPU 1
Input data, global 3D index [0,2,0], local index 0, rank 1 is -0.165956
[...]
GPU data, global 3D index [3 1 1], local index 7, rank 2 is (2.301149,-0.832995)
Relative Linf error on rank 2, 6.286432e-08
[...]
PASSED on rank 1
[...]
```
