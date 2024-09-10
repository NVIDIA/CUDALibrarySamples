# Image Euclidean Distance Transform (EDT) using NPP+

## Description

This project demonstrates the **Euclidean Distance Transform (EDT)** using the **Nvidia Performance Primitives (NPP+)** library. The Euclidean Distance Transform calculates the shortest distance from each pixel in a binary image to the nearest object or boundary, which is a key task in image segmentation and object detection.

## Key Concepts

- **Euclidean Distance Transform**: Determines the Euclidean distance between image elements and the nearest object or boundary.
- **Image Segmentation**: Separates an image into distinct regions based on object boundaries or other criteria.

## Supported SM Architectures

This example is optimized for the following Nvidia GPU architectures:

- [SM 7.0](https://developer.nvidia.com/cuda-gpus)
- [SM 7.2](https://developer.nvidia.com/cuda-gpus)
- [SM 7.5](https://developer.nvidia.com/cuda-gpus)
- [SM 8.0](https://developer.nvidia.com/cuda-gpus)

## Supported Operating Systems

- Linux
- Windows

## Supported CPU Architecture

- x86_64

## Getting Started with NPP+

To get started with this project, you will need to download and install the latest version of the NPP+ library:

- [Download NPP+](https://developer.nvidia.com/nppplus-downloads)

### NPP+ APIs Involved

This project uses key functions from the [NPP+](https://docs.nvidia.com/cuda/nppplus/introduction.html) library to efficiently compute the Euclidean distance transform on Nvidia GPUs.

## Architecture

This project consists of the following main component:

- **Euclidean Distance Transform (EDT)**: Computes the Euclidean distance between pixels and the nearest object or boundary.

## Prerequisites

Before building and running the project, ensure that your system meets the following requirements:

- A Linux or Windows system with an Nvidia GPU.
- The latest [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) installed.
- [CUDA 12.0 Toolkit and above](https://developer.nvidia.com/cuda-downloads) installed.

## Building the Project

## Build command on Linux
```
$ mkdir build
$ cd build
$ cmake .. -DNPP_PLUS_PATH=/usr/lib/x86_64-linux-gnu 
$ make
```


## Build command on Windows
```
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
$ Open distanceTransform.sln project in Visual Studio 15 2017 and build
```


# Running the Example
./distanceTransform -h
```
Usage: ./distanceTransform 
Parameters: 
	number-of-images	:	Use 2 number of images

```
Sample Output:
```
```
$  ./distanceTransform 

./distanceTransform 

NPP Library Version 0.9.0
CUDA Driver  Version: 12.7
CUDA Runtime Version: 12.6

Input file load succeeded.
Input file load succeeded.
Done!


```

