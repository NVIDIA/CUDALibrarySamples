# cuRAND Host APIs - `curandCreateGenerator[Host] - XORWOW`

## Description

This code demonstrates a usage of cuRAND `curandCreateGenerator[Host]` to generate XORWOW pseudorandom generated numbers

Example 1 generating uniform distribution.

Example 2 generating normal distribution.

Example 3 generating log-normal distribution.

Example 4 generating Poisson distribution.

See documentation for further details.

## Supported SM Architectures

All GPUs supported by CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)  

## Supported OSes

Linux  

## Supported CPU Architecture

x86_64  
ppc64le  
arm64-sbsa

## CUDA APIs involved
- [curandCreateGeneratorHost API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g35b6e9396d5b54b52ba9053496ad4ff4)
- [curandCreateGenerator API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g56ff2b3cf7e28849f73a1e22022bcbfd)
- [curandSetPseudoRandomGeneratorSeed API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gbcd2982aa3d53571b8ad12d8188b139b)
- [curandSetGeneratorOffset API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb21ba987f85486e552797206451b0939)
- [curandSetGeneratorOrdering API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gf1aa05715d726f94002d03237405fc5d)
- [curandGenerateUniform API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g5df92a7293dc6b2e61ea481a2069ebc2)
- [curandGenerateNormal API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gb9280e447ef04e1dec4611720bd0eb69)
- [curandGenerateLogNormal API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g3569cc960eb1a31357752fc813e21f49)
- [curandGeneratePoisson API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g425c7c13db4444e6150d159bb1417f05)

# Building (make)

# Prerequisites
- A Linux system with recent NVIDIA drivers.
- [CMake](https://cmake.org/download) version 3.18 minimum

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
Make sure that CMake finds expected CUDA Toolkit. If that is not the case you can add argument `-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` to cmake command.

# Usage #1
```
$  ./curand_xorwow_uniform_example
```

Sample example output:

```
Host
0.145468
0.820181
0.550399
0.294830
0.914733
0.868979
0.321921
0.782857
0.011302
0.285450
=====
Device
0.145468
0.820181
0.550399
0.294830
0.914733
0.868979
0.321921
0.782857
0.011302
0.285450
=====
```

# Usage #2
```
$  ./curand_xorwow_normal_example
```

Sample example output:

```
Host
2.561947
-2.603156
0.438303
2.127093
-1.022666
1.828079
3.916651
-0.124273
0.702847
0.209629
=====
Device
2.561947
-2.603156
0.438303
2.127093
-1.022666
1.828078
3.916651
-0.124272
0.702847
0.209629
=====
```

# Usage #3
```
$  ./curand_xorwow_lognormal_example
```

Sample example output:

```
Host
12.961029
0.074040
1.550074
8.390443
0.359635
6.221920
50.231937
0.883139
2.019494
1.233221
=====
Device
12.961033
0.074040
1.550074
8.390439
0.359635
6.221916
50.231937
0.883139
2.019495
1.233220
=====
```

# Usage #4
```
$  ./curand_xorwow_poisson_example
```

Sample example output:

```
Host
14
7
10
12
6
7
12
8
3
12
8
13
=====
Device
14
7
10
12
6
7
12
8
3
12
8
13
=====
```