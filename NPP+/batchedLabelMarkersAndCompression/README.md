# Batched Image Connected Region Label Markers and Compression using NPP+

## Description

This project demonstrates **Labeling and Compression of Image Regions** using the **Nvidia Performance Primitives (NPP+)** library. It efficiently labels connected regions in images and compresses the labels for optimized processing. This is especially useful for processing large batches of images in GPU-accelerated environments.

## Key Concepts

- **Image Region Labeling**: Identifies connected regions in an image and assigns labels to each region.
- **Label Compression**: Compresses labeled regions to optimize memory usage and performance, reducing redundancy in the labeled regions.
  
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

To use this example, first download and install the latest version of NPP+ on your Nvidia GPU-equipped machine:

- [Download NPP+](https://developer.nvidia.com/nppplus-downloads)

### NPP+ APIs Involved

This example leverages key functions from the [NPP+](https://docs.nvidia.com/cuda/nppplus/introduction.html) library for efficient image processing on Nvidia GPUs.

## Architecture

This project includes the following components:

- **Labeling Connected Image Regions**: Labels all connected regions in an image.
- **Compression of Labeled Regions**: Compresses the labeled regions for efficient memory usage and faster processing.

## Prerequisites

Before building and running the example, ensure that your system meets the following requirements:

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
$ Open findContour.sln project in Visual Studio 15 2017 and build
```


# Example Command
./batchedLabelMarkersAndCompression -h
```
Usage: ./batchedLabelMarkersAndCompression [-b number-of-batch]
Parameters: 
	number-of-batch	:	Use number of batch to process [default 5]

```
Sample Output:
```
```
$  ./batchedLabelMarkersAndCompression -b 5

./batchedLabelMarkersAndCompression 

NPP Library Version 0.9.0
CUDA Driver  Version: 12.7
CUDA Runtime Version: 12.6

Input file load succeeded.
Lena_CompressedMarkerLabelsUF_8Way_512x512_32u succeeded, compressed label count is 572.
Input file load succeeded.
CT_Skull_CompressedMarkerLabelsUF_8Way_512x512_32u succeeded, compressed label count is 434.
Input file load succeeded.
PCB_METAL_CompressedMarkerLabelsUF_8Way_509x335_32u succeeded, compressed label count is 3733.
Input file load succeeded.
PCB2_CompressedMarkerLabelsUF_8Way_1024x683_32u succeeded, compressed label count is 1272.
Input file load succeeded.
PCB_CompressedMarkerLabelsUF_8Way_1280x720_32u succeeded, compressed label count is 1468.



```

