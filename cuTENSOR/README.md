# cuTENSOR - Samples#

* [Documentation](https://docs.nvidia.com/cuda/cutensor/index.html)

# Install

## Linux 

You can use make or cmake to compile the cuTENSOR samples.

With make

```
export CUTENSOR_ROOT=<path_to_cutensor_root>
make
```

With cmake

```
cmake -S. -B build -DCUTENSOR_ROOT=<path_to_cutensor_root>
make --build build -j 8
```

## Windows

We recommend using cmake with Ninja generator to compile:

```
cmake -S. -B build -DCUTENSOR_ROOT=<path_to_cutensor_root> -G Ninja
cmake --build build
```

To run the examples, make sure the library files are located in a directory included in your %PATH%
