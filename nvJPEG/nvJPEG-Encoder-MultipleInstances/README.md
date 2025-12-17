# nvJPEG Image encoding example using multiple encode states

## Description

This example demonstrates how to use multiple encode states of the nvJPEG library to achieve optimal encode throughputs.
The example unifies the following techniques in a single framework:
- Host-side parallelization: assigning encode states to multiple threads
- Device-side parallelization: submitting encode jobs on multiple CUDA streams
- Hardware parallelization: utilizing all hardware encode engines (with as few as a single CPU thread)
- Host-device parallelization: double-buffering to optimally overlap host and device executions
- Backend parallelization: utilizing hardware engines and the CUDA cores
- Overhead minimization: padding of buffer sizes to avoid frequent reallocations
- Throughput optimization: automatic load balancing among backends
- Throughput optimization: automatic search for the best number of CPU threads

This example can also be used for benchmarking purpose, to determine the maximum encode throughput that can be achieved with nvJPEG on any single GPU (with or without hardware encode engines).

Detailed information about nvJPEG's API can be found at
[nvJPEG documentation](https://docs.nvidia.com/cuda/nvjpeg/index.html).

## Supported platforms

### GPU architectures
- Turing (SM 7.5)
- Ampere (SM 8.0, 8.6, and 8.7)
- Ada Lovelace (SM 8.9)
- Hopper (SM 9.0)
- Blackwell (SM 10.0, 10.3, 11.0, 12.0 and 12.1)

Note that hardware JPEG encode engines are only available on Jetson Thor (SM 11.0).

More information about the architectures and compute capabilities can be found at [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

### Operating systems

- Linux (x64 and aarch64)
- Windows (x64 and aarch64)

## Building the example
### Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- The [CUDA toolkit](https://developer.nvidia.com/cuda-downloads), at least version 12.9.
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
Prior to running, prepare a directory with input BMP files.
For benchmarking purposes, it is recommended to use only one BMP file in the input directory, so that buffers can be created once and cached, thus memory allocation and file I/O overheads are not taken into account.

```
-i indir [-n nimages] [-s nstates] [-j nthreads] [-b backends] [-r nruns] [-d] [-o outdir] [-q quality] [-f] [-p]

(OPTIONAL)  -i indir: Directory to take BMP images from.
(OPTIONAL)  -n nimages: Number of images to encode
             If not provided, encode all images in the input directory.
             Will be adjusted to be at least the number of states.
(OPTIONAL)  -s nstates: Number of states.
             If not provided, use twice the number of hardware encode engines, if present.
             Will be adjusted to be at least the number of threads.
(OPTIONAL)  -j nthreads: Number of CPU threads.
             If not provided, automatically find the best number of threads to use.
             Use 0 to set to the number of CPU hardware threads on the system.
(OPTIONAL)  -b backends: any of gpu/hardware/gpu hardware
(OPTIONAL)  -r nruns: Run this many times and pick the one with the maximum throughput.
(OPTIONAL)  -o outdir: Directory to write encoded JPEG images.
(OPTIONAL)  -d: Download JPEG bitstream to the host.
(OPTIONAL)  -q quality: JPEG quality 1-100 (default 80).
(OPTIONAL)  -f: Enable optimized Huffman.
(OPTIONAL)  -p: Disable chroma subsampling.
```

### Example commands:
Encode all images from directory "img" 
```
-i img
```
Encode all images from directory "img" using 1 thread (add -j)
```
-i img -j 1
```
Encode 4000 images (with potential repetitions) from directory "img" (add -n)
For benchmarking purposes, it is recommended to put just one image in the "img" directory.
This image will be encoded 4000 times.
```
-i img -n 4000
```
Write JPEG outputs to the current directory (add -o)
```
-i img -o .
```
Set JPEG quality (add -q), default is 80
```
-i img -q 70
```
Enable optimized Huffman (add -f), default is off
```
-i img -f 
```
Disable chroma subsampling (add -p), default is on (use 4:2:0)
```
-i img -p
```
Use only the hardware backend (add -b) (NOTE: this only works on Thor)
```
-i img -b hardware
```
Use both gpu and hardware backends (with automatic load-balancing) (NOTE: this only works on Thor)
```
-i img -b gpu hardware
```
Use all supported backends (the same as not providing -b)
```
-i img
```
Use as many threads as CPU hardware threads (-j 0)
```
-i img -j 0
```
Use two threads and 8 states (add -s)
```
-i img -j 2 -s 8
```
Use one thread and twice as many states as hardware JPEG encode engines (provide -s 0 or just omit it)
```
-i img -j 1
-i img -j 1 -s 0
```
Automatically detect the best number of states and CPU threads to use (omit -j)
```
-i img
```
Automatically detect the best number of CPU threads to use but start with 8 states
```
-i img -s 8
```
Benchmark individual backends (add -r to run multiple times and pick the best)
```
-i img -n 4000 -b gpu -r 4
-i img -n 4000 -b hardware -r 4
```
Benchmark automatic load-balancing among backends
```
-i img -n 4000 -b gpu hardware -r 4
```

### Example output
```
------------------------------------------------
GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU
Num hardware encode engines: 0
Input: C:/Users/duongh/Desktop/nvjpeg-benchmark/PBR_ImageDataS/img1920x1080_420
Enabled backends: gpu
Subsampling: off
Optimized Huffman: off
Quality: 80
Throughput: 3018.87 images/s
Latency: 1.32759, 0 ms
Percentage: 100, 0 %
Num threads: 4
Num states: 4
Num runs: 1
```