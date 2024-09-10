# Image Watershed Segmentation using NPP+

## Description

This project demonstrates **Image Segmentation using the Watershed Algorithm** using the **Nvidia Performance Primitives (NPP+)** library. The watershed algorithm divides an image into regions based on intensity gradients, making it useful for tasks such as object detection and image processing. This implementation follows the technique described in "Efficient 2D and 3D Watershed on Graphics Processing Unit: Block-Asynchronous Approaches Based on Cellular Automata" by Pablo Quesada-Barriuso and others.

## Key Concepts

- **Watershed-based Image Segmentation**: This technique segments an image into distinct regions based on variations in pixel intensities. It is commonly used in image processing for object detection and boundary extraction.

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

To start using NPP+ for image segmentation, download the latest version of the library and install it on a machine equipped with an Nvidia GPU.

- [Download NPP+](https://developer.nvidia.com/nppplus-downloads)

### NPP+ APIs Involved

This example utilizes the following key functions from the [NPP+](https://docs.nvidia.com/cuda/nppplus/introduction.html) library for efficient GPU-accelerated image processing.

## Architecture

This project includes the following main component:

- **Watershed Segmentation**: Performs image segmentation based on intensity gradients using the watershed algorithm.

## Prerequisites

Before building and running the example, make sure that your system meets the following requirements:

- A Linux or Windows system with an Nvidia GPU.
- The latest [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx).
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
$ Open watershedSegmentation.sln project in Visual Studio 15 2017 and build
```


# Running the Example
./watershedSegmentation -h
```
Usage: ./watershedSegmentation [-b number-of-batch]
Parameters: 
	number-of-batch	:	Use number of batch to process [default 3]

```
Sample Output:
```
```
$  ./watershedSegmentation -b 3

./watershedSegmentation 

NPP Library Version 0.9.0
CUDA Driver  Version: 12.7
CUDA Runtime Version: 12.6

Input file load succeeded.
CT_Skull_Segments_8Way_512x512_8u succeeded.
CT_Skull_CompressedSegmentLabels_8Way_512x512_32u succeeded.
CT_Skull_SegmentBoundaries_8Way_512x512_8u succeeded.
CT_Skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.
Input file load succeeded.
Rocks_Segments_8Way_512x512_8u succeeded.
Rocks_CompressedSegmentLabels_8Way_512x512_32u succeeded.
Rocks_SegmentBoundaries_8Way_512x512_8u succeeded.
Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.
Input file load succeeded.
Corn_Segments_8Way_614x461_8u succeeded.
Corn_CompressedSegmentLabels_8Way_614x461_32u succeeded.
Corn_SegmentBoundaries_8Way_614x461_8u succeeded.
Corn_SegmentsWithContrastingBoundaries_8Way_614x461_8u succeeded.


```

