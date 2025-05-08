# nvJPEG Image encoding example using multiple encoder states

## Description

This example demonstrates how to use multiple encoder states for JPEG encoding with the nvJPEG library.
Using multiple encoder states is useful to:
- Utilize all hardware encoder engines and/or
- Assign a different encoder engine to each CPU core for better parallelism

## Key Concepts

Image Encoding from NVJPEG Library

## Supported SM Architectures

- Pascal (SM 6.x)
- Volta (SM 7.0, and SM 7.2)
- Turing (SM 7.5)
- Ampere (SM 8.0, SM 8.6, and SM 8.7)
- Ada Lovelace (SM 8.9)
- Hopper (SM 9.0)
- Blackwell (SM 10.1 and SM 12.0)

More information can be found about the architectures and compute capabilities on the official NVIDIA website [here](https://developer.nvidia.com/cuda-gpus).

## Supported OSes

Linux
Windows

## Supported CPU Architecture

x86_64
aarch64

## CUDA APIs involved
[NVJPEG](https://docs.nvidia.com/cuda/nvjpeg/index.html)


# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open the project in Visual Studio and build
```

## Cross-compile to aarch64 (for Jetson Orin/Thor)
```
$ mkdir build
$ cd build
$ cmake -DCROSS_COMPILE_AARCH64=ON ..
$ Open the project in Visual Studio and build
```

# Usage
Three encoding modes are supported:
- Use single thread, and synchronize on the CPU after each image
- Use single thread, but do not synchronize on the CPU after each image
- Use multiple threads, and synchronize on the CPU on each thread after each image

In each mode, the user can optionally specify:
- The number of encode states (which is also the number of threads if multiple threads are used)
- The width of the image (the height is calculated based on 16:9 ratio)
- The input directory, from which we will read BMP files and encode them to JPEG
- The output directory (which, if not specified, no output JPEGs will be written)

./nvJPEGEncMultipleInstances

```
Usage: ./nvJPEGEncMultipleInstances -n nimages [-m mode] [-j nstates] [-w width] [-i indir] [-o outdir]
Parameters:
    -n nimages: Encode this many images
    -m mode:
        0 (single threaded, blocking)
        1 (single threaded, nonblocking)
        2 (multithreaded)
   -j nstates: Create this many encoder states (also the number of threads in multithreaded mode)
   -w width: Set the width of each image (the height will be set automatically assuming aspect ratio 16:9)
   -i indir: Read BMP images from this directory
   -o outdir: Write encoded images to this directory
```

Note that the `-n` option controls the total number of images generated, while the `-j` option controls the number of *unique* images generated (one for each encoder state).
For example, to generate 10 images from the same encoder state (so all images look the same), provide `-n 10 -j 1` (or just `-n 10` since `-j` is 1 by default).
Similarly, `-n 10 -j 2` will generate 10 images, of which 5 will look the same, and the other 5 will look the same.
If `-j` is larger than `-n`, we set `-n` to be the same as `-j` (e.g., `-n 1 -j 3` means the same as `-n 3 -j 3`, or just `-j 3`).

On a device with hardware encoder engines, `-j` also controls the number of engines that will be used.
If `-j` is larger than the number of engines available, we will loop around when assigning encoder states to hardware engines.

If an input directory is provided, we will read BMP images from it to encode to JPEG. If not, we will write randomly generated JPEG files.
The `-n` option works the same way in both cases, e.g., `-n 10` will read 10 images from the input directory. 
If `-n` is greater than the number of input images in the directory, we will loop around (so some images will be repeated).
To encode every BMP image in the input directory, provide `-n 0`.

Example:

- Encode single threaded blocking, 2 images, 1 states `./nvJPEGEncMultipleInstances -n 2`
- Encode single threaded blocking, 2 images, 1 states, write to current dir `./nvJPEGEncMultipleInstances -n 2 -o .`
- Encode single threaded blocking, 20 images, 2 states, write to current dir `./nvJPEGEncMultipleInstances -n 20 -j 2 -o .`
- Encode single threaded nonblocking, 20 images, 2 states, write to current dir `./nvJPEGEncMultipleInstances -n 20 -j 2 -m 1 -o .`
- Encode multithreaded, 20 images, 2 states (2 threads), write to current dir `./nvJPEGEncMultipleInstances -n 20 -j 2 -m 2 -o .`
- Encode single threaded nonblocking, 20 images, 2 state, from the `bmp` directory `nvJPEGEncMultipleInstances.exe -i bmp -n 20 -m 1`
- Encode multithreaded, 20 images, 20 states, from the `bmp` directory `nvJPEGEncMultipleInstances.exe -i bmp -j 20 -m 2`
- Encode multithreaded, all images, from the `bmp` directory `nvJPEGEncMultipleInstances.exe -i bmp -n 0 -m 2`
