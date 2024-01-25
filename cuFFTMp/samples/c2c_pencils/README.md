# C2C using a custom user distributions (pencils)
## Sample description
This sample is similar to [samples/c2c](../c2c/README.md), where it performs
- C2C forward transform
- [Scaling/normalization](../common/README.md)
- C2C backward transform.
  
But this sample assumes pencil decomposition layout:
- the input data is distributed using a pencil decomposition in X and Y, along Z;
- the output data is distributed using a pencil decomposition in X and Z, along Y.

This is achieved using a custom user-defined distribution and `cufftXtSetDistribution`.

## Build and run
This example requires 4 GPUs.

See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.

Example code snippet:
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