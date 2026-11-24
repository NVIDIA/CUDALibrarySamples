# R2C_C2R with Scratch Sharing and NCCL
## Sample description

This sample is similar to [samples/r2c_c2r_shared_scratch](../r2c_c2r_shared_scratch/README.md), where it performs
- R2C forward transform
- [Scaling/normalization](../common/README.md)
- C2R backward transform

with two plans that share one scratch buffer. But this sample uses the *NCCL communication backend* (`CUFFT_COMM_NCCL`) instead of NVSHMEM.

The steps are:
1. Create (but do not make) the plans
2. Call `cufftSetAutoAllocation(plan, false)` on all plans, on all ranks
3. Call `cufftMpMakePlan3d(plan, ..., &nccl_comm, CUFFT_COMM_NCCL, &scratch_size)` on all plans and get the scratch size of each plan
4. Compute the maximum scratch size across the plans
5. Allocate memory with `cudaMalloc`
6. Call `cufftSetWorkArea(plan, buffer)` on all plans
7. Call `cufftExec`, `cufftXtMemcpy`, etc.
8. Destroy the plans and free the memory

Differences with the NVSHMEM sample:
- The scratch buffer comes from `cudaMalloc`. NCCL does not use a symmetric heap, thus each rank can allocate a different size. No `MPI_Allreduce` is necessary.
- `cufftSetAutoAllocation(plan, false)` must come before `cufftMpMakePlan3d`. If a plan allocates its own work area, `cufftSetWorkArea` can return `CUFFT_INVALID_VALUE`.
- A plan without its own work area can need more scratch than a plan with its own work area (on one node with 2 GPUs and peer-to-peer access, 144 B instead of 48 B for each plan in this sample), and it can be slower. Thus two plans with their own work areas can use less memory than two plans that share one work area. Compare the `workSize` values on your system.
- The application destroys the communicator with `ncclCommDestroy` after it destroys the plans. cuFFTMp does not destroy the communicator.

This sample does not use NVSHMEM.

## Build and run
See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.
This sample also needs NCCL 2.30.7 or later. Set `NCCL_INC` and `NCCL_LIB` to the NCCL `include` and `lib` directories.
Each rank must use a different GPU. The sample selects the GPU with `rank % ndevices`. This gives each rank its own GPU if the ranks on each node are consecutive and each node has sufficient GPUs for its ranks. The default placement of Open MPI `mpirun` gives consecutive ranks on each node. With other placements (for example `mpirun --map-by node`), two ranks on a node can select the same GPU. `ncclCommInitRank` then returns `ncclInvalidUsage`. In this case, select the GPU with the rank of the process in its node (`MPI_Comm_split_type` with `MPI_COMM_TYPE_SHARED`). Refer to "Create the NCCL communicator" on the "NCCL and cuFFTMp" page of the cuFFTMp documentation.

Example code snippet:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ CUFFT_INC=/path/to/cufftmp/include CUFFT_LIB=/path/to/cufftmp/lib NCCL_INC=/path/to/nccl/include NCCL_LIB=/path/to/nccl/lib make run
Hello from rank 0/2 using GPU 0
Hello from rank 1/2 using GPU 1
Allocated 144 B of user scratch at 0x8d4eea02000 on rank 0/2
Allocated 144 B of user scratch at 0x11998ea02000 on rank 1/2
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
