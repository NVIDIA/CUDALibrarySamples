# Standalone Reshape Sample

This sample shows how to use the reshape API to re-distribute data accross GPUs.

## Sample description
Assume we have a 4x5 "global" world with the following data

|   | y  |    |    |    |    |
|---|----|----|----|----|----|
| x | 0  | 1  | 2  | 3  | 4  |
|   | 5  | 6  | 7  | 8  | 9  | 
|   | 10 | 11 | 12 | 13 | 14 |
|   | 15 | 16 | 17 | 18 | 19 |

Initially, data is distributed as follow

|   | y  |    |    |    |    | 
|---|----|----|----|----|----|
| x | 0  | 1  | 2  | 3  | 4  |
|   | 5  | 6  | 7  | 8  | 9  | 
|   | 10 | 11 | 12 | 13 | 14 |
|   | 15 | 16 | 17 | 18 | 19 |
|   | rank 0 | rank 0 | rank 0 | rank 1| rank 1|

where every rank owns part of it, stored in row major format
    
and we wish to redistribute it as
|   | y  |    |    |    |    |      |
|---|----|----|----|----|----|------|
| x | 0  | 1  | 2  | 3  | 4  | rank 0 |
|   | 5  | 6  | 7  | 8  | 9  | rank 0 |
|   | 10 | 11 | 12 | 13 | 14 | rank 1 |
|   | 15 | 16 | 17 | 18 | 19 | rank 1 |

where every rank should own part of it, stored in row major format

To do so, we describe the data using boxes where every box is described as low-high, where low and high are the lower (index inclusive) and upper (index exclusive) corners, respectively

- The "global/world" box is of size 4x5
- The input boxes are  [ (0,0)--(4,3), (0,3)--(4,5) ]
- The output boxes are [ (0,0)--(2,5), (2,0)--(4,5) ]

### Note
- This sample executes an *asymmetric* reshape. However, NVSHMEM requires *symmetric* heap allocation. For this reason, we need to allocate the maximum buffer size of both input and output buffers on all GPUs (`12 * sizeof(int)` in this case, where 12 is the number of elements in the input buffer on rank 0). 
- This sample uses `cufftMpExecReshapeAsync` API, which executes communication calls to write on memory buffers on remote GPUs. This means the user is responsible for making sure ``data_in``, ``data_out`` and ``workspace`` are available on all other GPUs before calling the API. More precisely, when one GPU starts to execute the API `cufftMpExecReshapeAsync` in a stream, the destination buffer and scratch buffer need to be ready on all other GPUs to prevent race conditions when writing to memory buffers on remote GPUs. This can be achieved by, for instance, placing synchronization points before the API. 
- At the end of `cufftMpExecReshapeAsync`, the memory buffers on all GPUs are available. 

## Build and run
See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.

Example code snippet:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ make run
Hello from rank 0/2 using GPU 0
Hello from rank 1/2 using GPU 1
Input data on rank 0: 0 1 2 5 6 7 10 11 12 15 16 17
Expected output data on rank 0: 0 1 2 3 4 5 6 7 8 9
Output data on rank 0: 0 1 2 3 4 5 6 7 8 9
Input data on rank 1: 3 4 8 9 13 14 18 19
Expected output data on rank 1: 10 11 12 13 14 15 16 17 18 19
Output data on rank 1: 10 11 12 13 14 15 16 17 18 19
PASSED
PASSED
```