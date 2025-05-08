# Batched Image connected region Label Markers And Compression using NPP

## Description

This code demonstrates Batched Image connected region Label Markers and Compression utility using NPP library.

## Key Concepts
- Connected region labeling 
- Label marker compression 
- Batched image processing using GPU acceleration

## Supported SM Architectures

[SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved
[NPP](https://docs.nvidia.com/cuda/npp/index.html)


# Architecture
- Image connected region label markers and compression.

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- Install the [CUDA 12.0 toolkit](https://developer.nvidia.com/cuda-downloads).

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
$ Open batchedLabelMarkersAndCompression.sln project in Visual Studio 15 2017 and build
```


# Usage
./batchedLabelMarkersAndCompression -h
```
Usage: ./batchedLabelMarkersAndCompression [-b number-of-batch]
Parameters: 
	number-of-batch	:	Use number of batch to process [default 5]

```
Example:
```
```
$  ./batchedLabelMarkersAndCompression -b 5

./batchedLabelMarkersAndCompression 

NPP Library Version 12.4.0
CUDA Driver  Version: 12.9
CUDA Runtime Version: 12.9

Input file load succeeded.
Lena_CompressedMarkerLabelsUF_8Way_512x512_32u succeeded, compressed label count is 572.
Input file load succeeded.
CT_Skull_CompressedMarkerLabelsUF_8Way_512x512_32u succeeded, compressed label count is 435.
Input file load succeeded.
PCB_METAL_CompressedMarkerLabelsUF_8Way_509x335_32u succeeded, compressed label count is 3733.
Input file load succeeded.
PCB2_CompressedMarkerLabelsUF_8Way_1024x683_32u succeeded, compressed label count is 1271.
Input file load succeeded.
PCB_CompressedMarkerLabelsUF_8Way_1280x720_32u succeeded, compressed label count is 1468.


Lena_CompressedMarkerLabelsUFBatch_8Way_512x512_32u succeeded, compressed label count is 555.
CT_Skull_CompressedMarkerLabelsUFBatch_8Way_512x512_32u succeeded, compressed label count is 413.
PCB_METAL_CompressedMarkerLabelsUFBatch_8Way_509x335_32u succeeded, compressed label count is 3732.
PCB2_CompressedMarkerLabelsUFBatch_8Way_1024x683_32u succeeded, compressed label count is 1222.
PCB_CompressedMarkerLabelsUFBatch_8Way_1280x720_32u succeeded, compressed label count is 1454.

```
