# Standalone Reshape Sample

This sample shows how to use the reshape API to re-distribute data accross GPUs.

## Sample description
Assume we have a 4x4 "global" world with the following data

|   | y  |    |    |    |
|---|----|----|----|----|
| x | 0  | 1  | 2  | 3  |
|   | 4  | 5  | 6  | 7  |
|   | 8  | 9  | 10 | 11 |
|   | 12 | 13 | 14 | 15 |

Initially, data is distributed as follow

|   | y  |    |    |    |
|---|----|----|----|----|
| x | 0  | 1  | 2  | 3  |
|   | 4  | 5  | 6  | 7  |
|   | 8  | 9  | 10 | 11 |
|   | 12 | 13 | 14 | 15 |
|   | rank 0 | rank 0 | rank 1 | rank 1|

where every rank owns part of it, stored in row major format
    
and we wish to redistribute it as
|   | y  |    |    |    |      |
|---|----|----|----|----|------|
| x | 0  | 1  | 2  | 3  |rank 0|
|   | 4  | 5  | 6  | 7  |rank 0|
|   | 8  | 9  | 10 | 11 |rank 1|
|   | 12 | 13 | 14 | 15 |rank 1|

where every rank should own part of it, stored in row major format

To do so, we describe the data using boxes where every box is described as low-high, where low and high are the lower (index inclusive) and upper (index exclusive) corners, respectively

- The "global/world" box is of size 4x4
- The input boxes are  [ (0,0)--(4,2), (0,2)--(4,4) ]
- The output boxes are [ (0,0)--(2,4), (2,0)--(4,4) ]

## Build and run
See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.

Example code snippet:
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