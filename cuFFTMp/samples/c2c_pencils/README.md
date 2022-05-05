# C2C using a custom user distributions (pencils)

This sample shows a Complex-to-complex distributed FFT computation using a custom user distribution and `cufftXtSetDistribution`.
It assumes that
- the input data is distributed using a pencil decomposition along Z;
- the output data is distributed using a pencil decomposition along Y.
This example requires 4 GPUs.

To build and run:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ make run
Hello from rank 3/4 using GPU 3
Hello from rank 1/4 using GPU 1
Hello from rank 0/4 using GPU 0
Hello from rank 2/4 using GPU 2
input data, global 3D index [2,0,0], local index 0, rank 2 is (-0.12801,-0.629836)
input data, global 3D index [2,0,1], local index 1, rank 2 is (-0.948148,0.863082)
[...]
output, global 3D index [0,0,2], local index 0, rank 1 is (-8.45704,12.8481)
output, global 3D index [0,0,3], local index 1, rank 1 is (3.18903,28.6322)
[...]
```