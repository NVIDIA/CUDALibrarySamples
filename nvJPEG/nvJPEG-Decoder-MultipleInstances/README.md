# nvJPEG Image decoding example using multiple decode states

## Description

This example demonstrates how to use multiple decode states of the nvJPEG library to achieve optimal decode throughputs.
The example unifies the following techniques in a single framework:
- Host-side parallelization: assigning decode states to multiple threads
- Device-side parallelization: submitting decode jobs on multiple CUDA streams
- Hardware parallelization: utilizing all hardware decoder engines (with as few as a single CPU thread)
- Host-device parallelization: double-buffering to optimally overlap host and device executions
- Backend parallelization: utilizing hardware engines, the CUDA cores, and CPU threads concurrently
- Overhead minimization: padding of buffer sizes to avoid frequent reallocations
- Throughput optimization: automatic load balancing among backends
- Throughput optimization: automatic search for the best number of CPU threads

This example can also be used for benchmarking purpose, to determine the maximum decode throughput that can be achieved with nvJPEG on any single GPU (with or without hardware decode engines).

Detailed information about nvJPEG's API can be found at
[nvJPEG documentation](https://docs.nvidia.com/cuda/nvjpeg/index.html).

## Supported platforms

### GPU architectures
- Turing (SM 7.5)
- Ampere (SM 8.0, 8.6, and 8.7)
- Ada Lovelace (SM 8.9)
- Hopper (SM 9.0)
- Blackwell (SM 10.0, 10.3, 11.0, 12.0 and 12.1)

More information about the architectures and compute capabilities can be found at [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

### Operating systems

- Linux (x64 and aarch64)
- Windows (x64 and aarch64)

## Building the example
### Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- The [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
- A C++ compiler with C++ 20 support.
- CMake 3.17 or later

### Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
```
Open the project in Visual Studio and build

### Cross-compile to aarch64 (for Jetson Orin/Thor)
```
$ mkdir build
$ cd build
$ cmake -DCROSS_COMPILE_AARCH64=ON ..
$ make
```

## Usage
Prior to running, prepare a directory with input JPEG files.
For benchmarking purposes, it is recommended to use only one JPEG file in the input directory, so that buffers can be created once and cached, thus memory allocation and file I/O overheads are not taken into account.

```
-i indir [-n nimages] [-s nstates] [-j nthreads] [-b backends] [-r nruns] [-o outdir]
        
(REQUIRED)  -i indir: Directory to take JPEG images from.
(OPTIONAL)  -n nimages: Number of images to decode.
             If not provided, decode all images in the input directory.
             Will be automatically adjusted to be at least number of states.
(OPTIONAL)  -s nstates: Number of states.
             If not provided, use twice the number of hardware engines.
             Will be automatically adjusted to be at least number of threads.
(OPTIONAL)  -j nthreads: Number of CPU threads.
             If not provided, automatically find the best number of threads to use.
             Use 0 to set to the number of CPU cores on the system.
(OPTIONAL)  -b backends: any of
             cpu/gpu/hardware/cpu gpu/cpu hardware/gpu hardware/cpu gpu hardware.
(OPTIONAL)  -r nruns: Run this many times and pick the one with the maximum throughput.
(OPTIONAL)  -o outdir: Directory to write decoded images in BMP format.
```

### Example commands:

- Decode all images from directory "img"
```
-i img
```
- Decode all images from directory "img" using 1 thread 
```
-i img -j 1
```
- Decode 4000 images (with potential repetitions) from directory "img" (add -n).
For benchmarking purposes, it is recommended to put just one image in the "img" directory.
This image will be decoded 4000 times.
```
-i img -n 4000
```
- Write outputs to the current directory (add -o)
```
-i img -o .
```
- Use only the hardware backend (add -b)
```
-i img -b hardware
```
- Use both gpu and hardware backends (with automatic load-balancing)
```
-i img -b gpu hardware
```
- Use cpu, gpu and hardware backends (with automatic load-balancing)
```
-i img -b cpu gpu hardware
```
- Use all supported backends (omit -b)
```
-i img
```
- Use as many threads as CPU hardware threads (-j 0)
```
-i img -j 0
```
- Use two threads and 8 states (add -s)
```
-i img -j 2 -s 8
```
- Use one thread and twice as many states as hardware JPEG decode engines (provide -s 0 or just obmit it)
```
-i img -j 1
-i img -j 1 -s 0
```
- Use one thread with all available hardware JPEG decode engines
```
-i img -j 1 -b hardware
```
- Automatically detect the best number of CPU threads to use (omit -j)
```
-i img
```
- Automatically detect the best number of CPU threads to use but start with 8 states
```
-i img -s 8
```
- Benchmark individual backends (add -r to run multiple times and pick the best)
```
-i img -n 4000 -b cpu -r 4
-i img -n 4000 -b gpu -r 4
-i img -n 4000 -b hardware -r 4
```
- Benchmark automatic load-balancing among backends
```
-i img -n 4000 -b cpu gpu hardware -r 4
```

### Example output
```
------------------------------------------------
GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU
Num hardware decode engines: 4
Input: img720x480_420
Enabled backends: cpu gpu hardware
Throughput: 6079.03 images/s
Latency: 2.3015, 3.42781, 1.00983 ms
Percentage: 0.925, 0.575, 97.95 %
Num threads: 4
Num states: 4
Num runs: 4
```