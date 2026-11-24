# Standalone Reshape Sample with NCCL

This sample shows how to use the reshape API to re-distribute data across GPUs with the NCCL communication backend (`CUFFT_COMM_NCCL`).

## Sample description
This sample does the same reshape as [samples/reshape](../reshape/README.md): a 4x5 "global" world of integers moves from a distribution in columns to a distribution in rows, on 2 ranks.

The differences with [samples/reshape](../reshape/README.md) are:
- The application creates an `ncclComm_t` and gives a pointer to it to `cufftMpMakeReshape` with `CUFFT_COMM_NCCL`.
- The input, output and workspace buffers come from `cudaMalloc`. The input and output buffers have the local size of each rank. With NVSHMEM, all ranks must allocate the same size on the symmetric heap.
- With NCCL, the reshape always needs a workspace. Thus `cufftMpGetReshapeSize` returns a workspace size that is not zero, and the application must supply this workspace to `cufftMpExecReshapeAsync`.
- No barrier (such as `nvshmemx_sync_all_on_stream`) is necessary before `cufftMpExecReshapeAsync`. NCCL writes the data into the output buffer of a rank in the stream order of that rank.
- The application destroys the communicator with `ncclCommDestroy` after it destroys the reshape handle. cuFFTMp does not destroy the communicator.

This sample does not use NVSHMEM.

## Build and run
See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.
This sample also needs NCCL 2.30.7 or later. Set `NCCL_INC` and `NCCL_LIB` to the NCCL `include` and `lib` directories.
Each rank must use a different GPU.

Example code snippet:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ CUFFT_INC=/path/to/cufftmp/include CUFFT_LIB=/path/to/cufftmp/lib NCCL_INC=/path/to/nccl/include NCCL_LIB=/path/to/nccl/lib make run
Hello from rank 0/2 using GPU 0
Hello from rank 1/2 using GPU 1
Rank 0 needs a workspace of 88 B
Rank 1 needs a workspace of 88 B
Input data on rank 0: 0 1 2 5 6 7 10 11 12 15 16 17
Expected output data on rank 0: 0 1 2 3 4 5 6 7 8 9
Output data on rank 0: 0 1 2 3 4 5 6 7 8 9
Input data on rank 1: 3 4 8 9 13 14 18 19
Expected output data on rank 1: 10 11 12 13 14 15 16 17 18 19
Output data on rank 1: 10 11 12 13 14 15 16 17 18 19
PASSED
PASSED
```
