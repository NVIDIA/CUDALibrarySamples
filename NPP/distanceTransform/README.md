# Image euclidean distance transform (EDT) using NPP

## Description

This code demonstrates Image euclidean distance transform (EDT) using NPP library.

## Key Concepts

Image segmentation  

## Supported SM Architectures

  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved
[NPP](https://docs.nvidia.com/cuda/npp/index.html)


# Architecture
- Image Euclidean Distance Transfrom (EDT).

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- Install the [CUDA 11.2 toolkit](https://developer.nvidia.com/cuda-downloads).

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
$ Open distanceTransform.sln project in Visual Studio 15 2017 and build
```


# Usage
./distanceTransform -h
```
Usage: ./distanceTransform 
Parameters: 
	number-of-images	:	Use 2 number of images

```
Example:
```
```
$  ./distanceTransform 

./distanceTransform 

NPP Library Version 11.3.2
CUDA Driver  Version: 11.2
CUDA Runtime Version: 11.2

Input file load succeeded.
Input file load succeeded.
Done!



Input Image
![dolphin1_Input_319x319_8u](/NPP/distanceTransform/dolphin1_Input_319x319_8u.jpg)

Distance Transform Image
![DistanceTransformTrue_Dolphin1_319x319_16u](/NPP/distanceTransform/DistanceTransformTrue_Dolphin1_319x319_16u.jpg)



```

