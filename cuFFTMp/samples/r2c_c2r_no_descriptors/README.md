# R2C_C2R Sample without descriptors
## Sample description
This sample is simiar to [samples/r2c_c2r](../r2c_c2r/README.md), where it performs
- R2C forward transform
- [Scaling/normalization](../common/README.md)
- C2R backward transform.

But this sample *does not use multi-GPU descriptors*.

## Build and run
See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.

Example code snippet:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ make run
Hello from rank 1/2 using GPU 1
Hello from rank 0/2 using GPU 0
GPU data (after first transform): global 3D index [0 0 0], local index 0, rank 0 is (3.783546,0.000000)
GPU data (after first transform): global 3D index [0 0 1], local index 1, rank 0 is (-0.385746,-1.599692)
GPU data (after first transform): global 3D index [0 0 2], local index 2, rank 0 is (-4.507643,0.000000)
GPU data (after first transform): global 3D index [1 0 0], local index 3, rank 0 is (0.778320,0.000000)
GPU data (after first transform): global 3D index [1 0 1], local index 4, rank 0 is (0.205157,1.710624)
GPU data (after first transform): global 3D index [1 0 2], local index 5, rank 0 is (1.806104,0.000000)
GPU data (after first transform): global 3D index [0 1 0], local index 0, rank 1 is (3.289261,0.000000)
GPU data (after first transform): global 3D index [0 1 1], local index 1, rank 1 is (-1.492967,2.346867)
GPU data (after first transform): global 3D index [0 1 2], local index 2, rank 1 is (0.645631,0.000000)
GPU data (after first transform): global 3D index [1 1 0], local index 3, rank 1 is (-2.242222,0.000000)
GPU data (after first transform): global 3D index [1 1 1], local index 4, rank 1 is (0.342550,-0.446430)
GPU data (after first transform): global 3D index [1 1 2], local index 5, rank 1 is (0.671049,0.000000)
PASSED with L2 error 5.600862e-08 < 1.000000e-06
```
