# Flood Fill Algorithm using NPP+

## Description

This project demonstrates the implementation of the **flood fill** algorithm using the **Nvidia Performance Primitives (NPP+)** library. The flood fill algorithm is a popular image processing technique used to determine and fill connected regions of a specific color in a 2D array, often applied in computer graphics and image editing.

This example showcases the high-performance capability of NPP+ in handling flood fill operations on large images using NVIDIA GPUs.

## Key Concepts

- **Flood Fill**: The algorithm identifies connected regions in an image starting from a seed point and "fills" the area by changing the pixel values of all connected components.
  
## System Requirements

### Supported SM Architectures

This project supports the following Nvidia GPU architectures:

- [SM 7.0](https://developer.nvidia.com/cuda-gpus)
- [SM 7.2](https://developer.nvidia.com/cuda-gpus)
- [SM 7.5](https://developer.nvidia.com/cuda-gpus)
- [SM 8.0](https://developer.nvidia.com/cuda-gpus)

### Supported Operating Systems

- Linux
- Windows

### Supported CPU Architectures

- x86_64

## Getting Started with NPP+

To begin using NPP+, download the latest version of the library and install it on your Nvidia GPU-equipped machine.

- [Download NPP+](https://developer.nvidia.com/nppplus-downloads)

### NPP+ APIs Involved
This example uses the [NPP+](https://docs.nvidia.com/cuda/nppplus/introduction.html) library, which provides optimized functions for image processing tasks on Nvidia GPUs.

## Architecture

This project focuses on demonstrating:

- **Flood Fill Image Processing**: Efficient identification and filling of connected regions in an image, using GPU-accelerated operations.

## Prerequisites

To build and run this project, you'll need:

- A Linux or Windows system with a recent version of NVIDIA drivers installed.
- The [CUDA 12.0 Toolkit](https://developer.nvidia.com/cuda-downloads).

## Building the Project

### Build Instructions on Linux
```
$ mkdir build
$ cd build
$ cmake .. -DNPP_PLUS_PATH=/usr/lib/x86_64-linux-gnu 
$ make
```

## Build Instructions on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open floodFill.sln project in Visual Studio 15 2017 and build
```

## Running the Application

To run the flood fill algorithm, use the following command:

./floodFill -h

```
Usage: ./floodFill 

```
Sample Output:
```
```
$  ./floodFill 

./floodFill 

NPP Library Version 0.9.0
CUDA Driver  Version: 12.7
CUDA Runtime Version: 12.6

Input file load succeeded.
BoundsRect x 0 y 0 width 1023 height 444 count 257557 seed0 255 seed1 255 seed2 255. 
Input file load succeeded.
BoundsRect x 0 y 0 width 1023 height 439 count 343565 seed0 255 seed1 255 seed2 255. 
Input file load succeeded.
BoundsRect x 0 y 0 width 1023 height 443 count 343565 seed0 255 seed1 255 seed2 255. 
Input file load succeeded.
BoundsRect x 0 y 0 width 674 height 657 count 318444 seed0 149 seed1 205 seed2 229. 
Input file load succeeded.
BoundsRect x 0 y 0 width 674 height 658 count 318442 seed0 149 seed1 205 seed2 229. 



```

