# cuRAND Host APIs - `curandCreateGenerator[Host] - SOBOL64`

## Description

This code demonstrates a usage of cuRAND `curandCreateGenerator[Host]` to generate SOBOL64 quasirandom generated numbers

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
- [cucurandGeneratePoisson API](https://docs.nvidia.com/cuda/curand/group__HOST.html#group__HOST_1g425c7c13db4444e6150d159bb1417f05)

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
$  ./curand_sobol64_uniform_example
```

Sample example output:

```
Host
0.000000
0.500000
0.750000
0.250000
0.375000
0.875000
0.625000
0.125000
0.187500
0.687500
=====
Device
0.000000
0.500000
0.750000
0.250000
0.375000
0.875000
0.625000
0.125000
0.187500
0.687500
=====
```

# Usage #2
```
$  ./curand_sobol64_normal_example
```

Sample example output:

```
Host
-11.675915
1.000000
2.348979
-0.348979
0.362721
3.300699
1.637279
-1.300699
-0.774293
1.977553
=====
Device
-11.675915
1.000000
2.348979
-0.348979
0.362721
3.300699
1.637279
-1.300699
-0.774293
1.977553
=====
```

# Usage #3
```
$  ./curand_sobol64_lognormal_example
```

Sample example output:

```
Host
0.000008
2.718282
10.474874
0.705408
1.437235
27.131590
5.141160
0.272341
0.461030
7.225041
=====
Device
0.000008
2.718282
10.474874
0.705408
1.437235
27.131592
5.141160
0.272341
0.461030
7.225041
=====
```

# Usage #4
```
$  ./curand_sobol64_poisson_example
```

Sample example output:

```
Host
0
10
12
8
9
14
11
6
7
11
15
9
=====
Device
0
10
12
8
9
14
11
6
7
11
15
9
=====
```