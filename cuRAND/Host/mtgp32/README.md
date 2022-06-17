# cuRAND Host APIs - `curandCreateGenerator[Host] - MTGP32`

## Description

This code demonstrates a usage of cuRAND `curandCreateGenerator[Host]` to generate MTGP32 pseudorandom generated numbers

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
$  ./curand_mtgp32_uniform_example
```

Sample example output:

```
Host
0.786752
0.152112
0.934182
0.986567
0.988028
0.304493
0.926525
0.432048
0.346986
0.840585
=====
Device
0.786752
0.152112
0.934182
0.986567
0.988028
0.304493
0.926525
0.432048
0.346986
0.840585
=====
```

# Usage #2
```
$  ./curand_mtgp32_normal_example
```

Sample example output:

```
Host
1.415674
0.817308
1.588038
6.369638
-1.367920
-0.242695
3.866537
2.142646
-0.471517
0.531867
=====
Device
1.415674
0.817308
1.588038
6.369638
-1.367920
-0.242695
3.866537
2.142646
-0.471517
0.531867
=====
```

# Usage #3
```
$  ./curand_mtgp32_lognormal_example
```

Sample example output:

```
Host
4.119263
2.264397
4.894138
583.846680
0.254636
0.784510
47.776665
8.521959
0.624055
1.702106
=====
Device
4.119263
2.264397
4.894139
583.846680
0.254636
0.784510
47.776642
8.521960
0.624055
1.702106
=====
```

# Usage #4
```
$  ./curand_mtgp32_poisson_example
```

Sample example output:

```
Host
1.415674
0.817308
1.588038
6.369638
-1.367920
-0.242695
3.866537
2.142646
-0.471517
0.531867
=====
Device
1.415674
0.817308
1.588038
6.369638
-1.367920
-0.242695
3.866537
2.142646
-0.471517
0.531867
=====
```