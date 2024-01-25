# R2C_C2R Sample using a custom user distributions (slabs)
## Sample description
This sample is simiar to [samples/r2c_c2r](../r2c_c2r/README.md), where it performs
- R2C forward transform
- [Scaling/normalization](../common/README.md)
- C2R backward transform.

But this sample assumes slab decomposition layout
- Input data are real slabs distributed along X. 
- Output data are complex slabs distributed along Y.

This sample also mimics the use case of cuFFTMp in GROMACS (for PME ranks <= 8).
Additional functionalities are added in this sample for comparing with cuFFTMp in GROMACS:
- By default, 10 cycles (with 1 warm-up cycle) of (R2C + [Scaling](../common/README.md) + C2R) transform are performed.
- Timings are reported.
- Printing in scaling kernel is turned off for performance reasons.
- Correctness is verified (with relaxed tolerance based on the number of cycles).

Refer to the following links for more information
- [GROMACS Official Site](https://www.gromacs.org/)
- [Using cuFFTMp in GROMACS](https://manual.gromacs.org/nightly/install-guide/index.html#using-cufftmp)
- [Implementation of cuFFTMp backend in GROMACS](https://github.com/gromacs/gromacs/blob/main/src/gromacs/fft/gpu_3dfft_cufftmp.cpp#L110)


## Build and run

This sample agrees with cuFFTMp usage in GROMACS up to 8 GPUs (PME ranks), but can be ran on any number of GPUs.

See [Requirements](../../README.md) and [Quick start for C++ samples](../../README.md) for hardware/software requirements and build instructions.

Example code snippet (e.g., on 4 GPUs):
```
$ MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/ make run
Hello from rank 0/4 using GPU 0
<...>
For a total of 10 cycles
Average time for 1 cycle (GPU timer): 0.765747 ms
Average time for 1 cycle (CPU timer): 0.768102 ms
Average time for each individual step (GPU timer):
C2R: 0.336384 ms
Scaling: 0.0120832 ms
R2C: 0.41728 ms
PASSED with L2 error 3.362441e-06 < 1.100000e-05
<...>
```

