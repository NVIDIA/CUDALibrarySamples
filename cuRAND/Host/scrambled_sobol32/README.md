# cuRAND Host APIs - `curandCreateGenerator[Host] - Scrambled SOBOL32`

## Description

This code demonstrates a usage of cuRAND `curandCreateGenerator[Host]` to generate scrambled SOBOL32 quasirandom generated numbers

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
- [curandSetQuasiRandomGeneratorDimensions API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1gd00db3478a788b5823038481495bb6ab)
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
$  ./curand_scrambled_sobol32_uniform_example
```

Sample example output:

```
Host
0.882941
0.046950
0.332641
0.731132
0.554493
0.390621
0.229930
0.831301
0.755963
0.185704
=====
Device
0.882941
0.046950
0.332641
0.731132
0.554493
0.390621
0.229930
0.831301
0.755963
0.185704
=====
```

# Usage #2
```
$  ./curand_scrambled_sobol32_normal_example
```

Sample example output:

```
Host
3.379633
-2.350356
0.134736
2.232481
1.274041
0.444599
-0.478156
2.918642
2.386749
-0.787681
=====
Device
3.379633
-2.350356
0.134736
2.232480
1.274041
0.444599
-0.478156
2.918642
2.386749
-0.787681
=====
```

# Usage #3
```
$  ./curand_scrambled_sobol32_lognormal_example
```

Sample example output:

```
Host
29.359985
0.095335
1.144234
9.322964
3.575273
1.559865
0.619926
18.516123
10.878075
0.454898
=====
Device
29.359985
0.095335
1.144234
9.322962
3.575273
1.559865
0.619926
18.516119
10.878075
0.454898
=====
```

# Usage #4
```
$  ./curand_scrambled_sobol32_poisson_example
```

Sample example output:

```
Host
14
5
8
12
10
9
8
13
12
7
10
11
=====
Device
14
5
8
12
10
9
8
13
12
7
10
11
=====
```