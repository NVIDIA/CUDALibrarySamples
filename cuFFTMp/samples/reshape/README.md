# Standalone Reshape Sample

This sample shows how to use the reshape API to re-distribute data accross GPUs.

To build and run:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ make run
Hello from rank 1/2 using GPU 1
Hello from rank 0/2 using GPU 0
Input data on rank 0: 0 1 4 5 8 9 12 13
Expected output data on rank 0: 0 1 2 3 4 5 6 7
Output data on rank 0: 0 1 2 3 4 5 6 7
Input data on rank 1: 2 3 6 7 10 11 14 15
Expected output data on rank 1: 8 9 10 11 12 13 14 15
Output data on rank 1: 8 9 10 11 12 13 14 15
PASSED
PASSED
```