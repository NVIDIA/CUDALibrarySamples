# cuSOLVER batched sparse QR example 1

## Description

This chapter demonstrates a usage of cuSOLVER csrqr function two examples to perform batched sparse QR solver.

The example 1 performs batched sparse QR to solver a set of linear systems.

The example 2 performs batched sparse QR to solver a set of linear systems, but we assume device memory is not enough, so we need to cut 17 matrices into several chunks and compute each chunk by batched sparse QR.

_**A**<sub>i</sub>x<sub>i</sub> = b<sub>i</sub>_

All matrices A<sub>i</sub> are small perturbations of
```
A = | 1.0 | 0.0 | 0.0 | 0.0 |
    | 0.0 | 2.0 | 0.0 | 0.0 |
    | 0.0 | 0.0 | 3.0 | 0.0 |
    | 0.1 | 0.1 | 0.1 | 4.0 |
```

All right-hand side vectors b<sub>i</sub> are small pertubation of the Matlab vector *ones(4,1)*.
We assume device memory is big enough to compute all matrices in one pass.

## Supported SM Architectures

All GPUs supported by CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)  

## Supported OSes

Linux  
Windows  

## Supported CPU Architecture

x86_64  
ppc64le  
arm64-sbsa

## CUDA APIs involved
- [cusolverSpXcsrqrAnalysisBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csrqrbatched)
- [cusolverSpDcsrqrBufferInfoBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csrqrbatched)
- [cusolverSpDcsrqrsvBatched API](https://docs.nvidia.com/cuda/cusolver/index.html#cusolver-lt-t-gt-csrqrbatched)

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Make sure that CMake finds expected CUDA Toolkit. If that is not the case you can add argument `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to cmake command.

## Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open cusolver_examples.sln project in Visual Studio and build
```

# Usage 1
```
$  ./cusolver_csrqr_example1
```

Sample example output:

```
numerical factorization needs working space 209664 bytes
batchId 0: sup|bj - Aj*xj| = 6.661338E-16
batchId 1: sup|bj - Aj*xj| = 1.110223E-15
batchId 2: sup|bj - Aj*xj| = 4.440892E-16
batchId 3: sup|bj - Aj*xj| = 4.440892E-16
batchId 4: sup|bj - Aj*xj| = 6.661338E-16
batchId 5: sup|bj - Aj*xj| = 6.661338E-16
batchId 6: sup|bj - Aj*xj| = 8.881784E-16
batchId 7: sup|bj - Aj*xj| = 4.440892E-16
batchId 8: sup|bj - Aj*xj| = 1.110223E-15
batchId 9: sup|bj - Aj*xj| = 4.440892E-16
batchId 10: sup|bj - Aj*xj| = 2.220446E-16
batchId 11: sup|bj - Aj*xj| = 1.332268E-15
batchId 12: sup|bj - Aj*xj| = 6.661338E-16
batchId 13: sup|bj - Aj*xj| = 6.661338E-16
batchId 14: sup|bj - Aj*xj| = 2.220446E-16
batchId 15: sup|bj - Aj*xj| = 6.661338E-16
batchId 16: sup|bj - Aj*xj| = 4.440892E-16
x0[0] = 9.936533E-01
x0[1] = 4.996754E-01
x0[2] = 3.333555E-01
x0[3] = 2.032261E-01

x1[0] = 9.954397E-01
x1[1] = 5.023664E-01
x1[2] = 3.332114E-01
x1[3] = 2.011510E-01

x2[0] = 9.952372E-01
x2[1] = 5.000749E-01
x2[2] = 3.338207E-01
x2[3] = 2.033641E-01

x3[0] = 1.001597E+00
x3[1] = 5.009494E-01
x3[2] = 3.342649E-01
x3[3] = 2.024054E-01

x4[0] = 9.924708E-01
x4[1] = 4.997259E-01
x4[2] = 3.349327E-01
x4[3] = 2.012184E-01

x5[0] = 1.006178E+00
x5[1] = 5.018936E-01
x5[2] = 3.335885E-01
x5[3] = 2.038061E-01

x6[0] = 9.985129E-01
x6[1] = 5.008987E-01
x6[2] = 3.348651E-01
x6[3] = 2.022812E-01

x7[0] = 9.989101E-01
x7[1] = 4.994274E-01
x7[2] = 3.351732E-01
x7[3] = 2.014273E-01

x8[0] = 1.002587E+00
x8[1] = 4.994259E-01
x8[2] = 3.360536E-01
x8[3] = 2.022084E-01

x9[0] = 9.988026E-01
x9[1] = 4.994268E-01
x9[2] = 3.352765E-01
x9[3] = 2.035007E-01

x10[0] = 9.965219E-01
x10[1] = 5.029465E-01
x10[2] = 3.333555E-01
x10[3] = 2.036849E-01

x11[0] = 1.003989E+00
x11[1] = 5.017441E-01
x11[2] = 3.343099E-01
x11[3] = 2.018912E-01

x12[0] = 9.966307E-01
x12[1] = 4.995508E-01
x12[2] = 3.339314E-01
x12[3] = 2.022344E-01

x13[0] = 1.003777E+00
x13[1] = 5.000499E-01
x13[2] = 3.361865E-01
x13[3] = 2.026724E-01

x14[0] = 9.990064E-01
x14[1] = 5.019747E-01
x14[2] = 3.331890E-01
x14[3] = 2.042594E-01

x15[0] = 1.005984E+00
x15[1] = 5.019228E-01
x15[2] = 3.337102E-01
x15[3] = 2.007337E-01

x16[0] = 1.002490E+00
x16[1] = 5.033651E-01
x16[2] = 3.353435E-01
x16[3] = 2.031494E-01
```

# Usage 2
```
$  ./cusolver_csrqr_example2
```

Sample example output:

```
batchSizeMax = 2
batchSizeMax = 4
batchSizeMax = 8
batchSizeMax = 16
batchSizeMax = 17
numerical factorization needs internal data 272 bytes
numerical factorization needs working space 201984 bytes
current batchSize = 2
current batchSize = 2
current batchSize = 2
current batchSize = 2
current batchSize = 2
current batchSize = 2
current batchSize = 2
current batchSize = 2
current batchSize = 1
batchId 0: sup|bj - Aj*xj| = 6.661338E-16
batchId 1: sup|bj - Aj*xj| = 1.110223E-15
batchId 2: sup|bj - Aj*xj| = 4.440892E-16
batchId 3: sup|bj - Aj*xj| = 4.440892E-16
batchId 4: sup|bj - Aj*xj| = 6.661338E-16
batchId 5: sup|bj - Aj*xj| = 6.661338E-16
batchId 6: sup|bj - Aj*xj| = 8.881784E-16
batchId 7: sup|bj - Aj*xj| = 4.440892E-16
batchId 8: sup|bj - Aj*xj| = 1.110223E-15
batchId 9: sup|bj - Aj*xj| = 4.440892E-16
batchId 10: sup|bj - Aj*xj| = 2.220446E-16
batchId 11: sup|bj - Aj*xj| = 1.332268E-15
batchId 12: sup|bj - Aj*xj| = 6.661338E-16
batchId 13: sup|bj - Aj*xj| = 6.661338E-16
batchId 14: sup|bj - Aj*xj| = 2.220446E-16
batchId 15: sup|bj - Aj*xj| = 6.661338E-16
batchId 16: sup|bj - Aj*xj| = 4.440892E-16
x0[0] = 9.936533E-01
x0[1] = 4.996754E-01
x0[2] = 3.333555E-01
x0[3] = 2.032261E-01

x1[0] = 9.954397E-01
x1[1] = 5.023664E-01
x1[2] = 3.332114E-01
x1[3] = 2.011510E-01

x2[0] = 9.952372E-01
x2[1] = 5.000749E-01
x2[2] = 3.338207E-01
x2[3] = 2.033641E-01

x3[0] = 1.001597E+00
x3[1] = 5.009494E-01
x3[2] = 3.342649E-01
x3[3] = 2.024054E-01

x4[0] = 9.924708E-01
x4[1] = 4.997259E-01
x4[2] = 3.349327E-01
x4[3] = 2.012184E-01

x5[0] = 1.006178E+00
x5[1] = 5.018936E-01
x5[2] = 3.335885E-01
x5[3] = 2.038061E-01

x6[0] = 9.985129E-01
x6[1] = 5.008987E-01
x6[2] = 3.348651E-01
x6[3] = 2.022812E-01

x7[0] = 9.989101E-01
x7[1] = 4.994274E-01
x7[2] = 3.351732E-01
x7[3] = 2.014273E-01

x8[0] = 1.002587E+00
x8[1] = 4.994259E-01
x8[2] = 3.360536E-01
x8[3] = 2.022084E-01

x9[0] = 9.988026E-01
x9[1] = 4.994268E-01
x9[2] = 3.352765E-01
x9[3] = 2.035007E-01

x10[0] = 9.965219E-01
x10[1] = 5.029465E-01
x10[2] = 3.333555E-01
x10[3] = 2.036849E-01

x11[0] = 1.003989E+00
x11[1] = 5.017441E-01
x11[2] = 3.343099E-01
x11[3] = 2.018912E-01

x12[0] = 9.966307E-01
x12[1] = 4.995508E-01
x12[2] = 3.339314E-01
x12[3] = 2.022344E-01

x13[0] = 1.003777E+00
x13[1] = 5.000499E-01
x13[2] = 3.361865E-01
x13[3] = 2.026724E-01

x14[0] = 9.990064E-01
x14[1] = 5.019747E-01
x14[2] = 3.331890E-01
x14[3] = 2.042594E-01

x15[0] = 1.005984E+00
x15[1] = 5.019228E-01
x15[2] = 3.337102E-01
x15[3] = 2.007337E-01

x16[0] = 1.002490E+00
x16[1] = 5.033651E-01
x16[2] = 3.353435E-01
x16[3] = 2.031494E-01
```
