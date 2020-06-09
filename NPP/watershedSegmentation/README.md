# Image watershed segmentation using NPP

## Description

This code demonstrates Image segmentation using watershed algorithm utility using NPP library.
Segments a grayscale image using the watershed segmentation technique described in "Efficient 2D and 3D Watershed on Graphics Processing Unit: Block-Asynchronous Approaches Based on Cellular Automata" by Pablo Quesada-Barriuso and others.

## Key Concepts

Image segmentation  

## Supported SM Architectures

 [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved
[NPP](https://docs.nvidia.com/cuda/npp/index.html)


# Architecture
- Image segmentation using watershed.

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- Install the [CUDA 11.0 toolkit](https://developer.nvidia.com/cuda-downloads).

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
$ Open watershedSegmentation.sln project in Visual Studio 15 2017 and build
```


# Usage
./watershedSegmentation -h
```
Usage: ./watershedSegmentation [-b number-of-batch]
Parameters: 
	number-of-batch	:	Use number of batch to process [default 3]

```
Example:
```
```
$  ./watershedSegmentation -b 3

./watershedSegmentation 

NPP Library Version 11.0.0
CUDA Driver  Version: 11.0
CUDA Runtime Version: 11.0

Input file load succeeded.
Lena_Segments_8Way_512x512_8u succeeded.
Lena_CompressedSegmentLabels_8Way_512x512_32u succeeded.
Lena_SegmentBoundaries_8Way_512x512_8u succeeded.
Lena_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.

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


```

