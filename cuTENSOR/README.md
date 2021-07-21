# cuTENSOR - Samples#

* [Documentation](https://docs.nvidia.com/cuda/cutensor/index.html)

# Install

## Linux 

You can use make or cmake to compile the cuTENSOR samples.

With make

```
export CUTENSOR_ROOT=<path_to_cutensor_root>
make -j8
```

With cmake

```
mkdir build && cd build
cmake .. -DCUTENSOR_ROOT=<path_to_cutensor_root>
make -j8
```

## Windows

We recommend using cmake with Ninja generator to compile:

```
mkdir build && cd build
cmake .. -DCUTENSOR_ROOT=<path_to_cutensor_root> -G Ninja
ninja
```

To run the examples, make sure the library files are located in a directory included in your %PATH%
