# R2C_C2R Sample using a custom user distributions (pencils)
## Sample description
This sample is simiar to [samples/r2c_c2r](../r2c_c2r/README.md), where it performs
- R2C forward transform
- [Scaling/normalization](../common/README.md)
- C2R backward transform.

But this sample assumes pencil decomposition layout
- Input data are real pencils in X & Y, along Z. Strides are packed and in-place (i.e., real is padded)
- Output data are complex pencils in X & Z, along Y (picked arbitrarily). Strides are packed. For best performances, the local dimension in the input (Z, here) and output (Y, here) should be different to ensure cuFFTMp will only perform two communication phases. If Z was also local in the output, cuFFTMp would perform three communication phases, decreasing performances. cuFFTMp will try to minimize the number of communication phases whenever possible. In some cases, such as Slabs-X to Slabs-Y, cuFFTMp will only perform a single communication phase.

This sample explains how data is distributed according to `CUFFT_XT_FORMAT_DISTRIBUTED_INPUT` and `CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT`.

## Build and run
This example requires 4 GPUs.

See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.

Example code snippet:
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ make run
Hello from rank 1/4 using GPU 1
Hello from rank 3/4 using GPU 3
Hello from rank 2/4 using GPU 2
Hello from rank 0/4 using GPU 0
Input data, global 3D index [0,2,0], local index 0, rank 1 is -0.165956
[...]
GPU data (after first transform): global 3D index [0 4 3], local index 9, rank 1 is (0.412567,-9.293055)
PASSED with L2 error 1.156259e-07 < 1.000000e-06
[...]
```
