# cuRAND Host APIs - `curandCreateGenerator[Host] - PHILOX`

## Description

This code demonstrates a usage of cuRAND `curandCreateGenerator[Host]` to generate PHILOX pseudorandom generated numbers

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
$  ./curand_philox_uniform_example
```

Sample example output:

```
Host
0.127208
0.853469
0.265649
0.796055
0.816736
0.324165
0.781055
0.058918
0.543999
0.923496
0.441505
0.258264
=====
Device
0.127208
0.853469
0.265649
0.796055
0.816736
0.324165
0.781055
0.058918
0.543999
0.923496
0.441505
0.258264
=====

```

# Usage #2
```
$  ./curand_philox_normal_example
```

Sample example output:

```
Host
-2.232981
3.458311
-2.121081
1.929243
2.136910
0.428207
1.508686
2.310768
-0.020449
2.956819
3.553997
0.867263
=====
Device
-2.232981
3.458309
-2.121081
1.929241
2.136910
0.428208
1.508686
2.310768
-0.020450
2.956819
3.553997
0.867263
=====

```

# Usage #3
```
$  ./curand_philox_lognormal_example
```

Sample example output:

```
Host
0.107208
31.763277
0.119902
6.884295
8.473219
1.534504
4.520785
10.082164
0.979758
19.236683
34.952755
2.380386
=====
Device
0.107208
31.763233
0.119902
6.884281
8.473219
1.534505
4.520784
10.082161
0.979758
19.236685
34.952747
2.380388
=====

```
# Usage #4
```
$  ./curand_philox_poisson_example
```

Sample example output:

```
Host
15
7
12
8
7
12
8
17
10
6
11
12
=====
Device
15
7
12
8
7
12
8
17
10
6
11
12
=====
```