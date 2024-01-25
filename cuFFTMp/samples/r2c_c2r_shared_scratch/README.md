# R2C_C2R with Scratch Sharing
## Sample description

This sample is simiar to [samples/r2c_c2r](../r2c_c2r/README.md), where it performs
- R2C forward transform
- [Scaling/normalization](../common/README.md)
- C2R backward transform.
  
This sample illustrates how to share scratch memory among plans to minimize memory usage.

This can be done as the following:
1. Create (but don't make) plans
2. Call `cufftSetAutoAllocation(plan, false)` on all plans
3. Call `cufftMakePlan3d(plan, ..., scratch_size)` on all plans and retrieve the required scratch size per plan
4. Compute the maximum scratch size accros plans *AND* accross MPI ranks (see note below on `nvshmem_malloc`)
5. Allocate memory using `nvshmem_malloc`
6. Call `cufftSetWorkArea(plan, buffer)` on all plans
7. Call `cufftExec`, `cufftXtMemcpy`, etc
8. Free memory using `nvshmem_free`
9. Destroy the plan
    
Note that `nvshmem_malloc` requires the same "size" argument on every MPI rank
- Hence, if scratch_size is not identical on every rank, the max accross ranks should be used. (See https://docs.nvidia.com/hpc-sdk/nvshmem/api/docs/gen/api/memory.html#c.nvshmem_malloc)
- Except for FFT kernels that don't require any scratch (like powers of 2), there is no guarantees that cuFFT requires the same amount of scratch on all ranks Hence, the user should compute the max across MPI ranks (e.g. using `MPI_Allreduce`) and pass this to `nvshmem_malloc`.
  
## Build and run
See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.

Example code snippet:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ make run
Hello from rank 0/2 using GPU 0
Hello from rank 1/2 using GPU 1
Allocated 48 B of user scratch at 0x10027a00600 on rank 1/2
Allocated 48 B of user scratch at 0x10027a00600 on rank 0/2
GPU data (after first transform): global 3D index [0 1 0], local index 0, rank 1 is (3.289261,0.000000)
GPU data (after first transform): global 3D index [0 1 1], local index 1, rank 1 is (-1.492967,2.346867)
GPU data (after first transform): global 3D index [0 1 2], local index 2, rank 1 is (0.645631,0.000000)
GPU data (after first transform): global 3D index [1 1 0], local index 3, rank 1 is (-2.242222,0.000000)
GPU data (after first transform): global 3D index [1 1 1], local index 4, rank 1 is (0.342550,-0.446430)
GPU data (after first transform): global 3D index [1 1 2], local index 5, rank 1 is (0.671049,0.000000)
GPU data (after first transform): global 3D index [0 0 0], local index 0, rank 0 is (3.783546,0.000000)
GPU data (after first transform): global 3D index [0 0 1], local index 1, rank 0 is (-0.385746,-1.599692)
GPU data (after first transform): global 3D index [0 0 2], local index 2, rank 0 is (-4.507643,0.000000)
GPU data (after first transform): global 3D index [1 0 0], local index 3, rank 0 is (0.778320,0.000000)
GPU data (after first transform): global 3D index [1 0 1], local index 4, rank 0 is (0.205157,1.710624)
GPU data (after first transform): global 3D index [1 0 2], local index 5, rank 0 is (1.806104,0.000000)
PASSED with L2 error 5.600862e-08 < 1.000000e-06
```
