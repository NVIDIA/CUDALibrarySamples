# JPEG2000 Image decoding Example using nvJPEG 2000 Library

## Description

This code demonstrates how to pipleline the decoding of multiple JPEG 2000 images nvJPEG2000 library. 

## Key Concepts

Image Decoding from nvJPEG2000 Library

## Supported SM Architectures

  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved

[NVJPEG2000](https://docs.nvidia.com/cuda/nvjpeg2000/index.html)


# Building (make)

# Prerequisites
- A Linux system with recent NVIDIA drivers.
- Install the [CUDA 11.0 toolkit](https://developer.nvidia.com/cuda-downloads).
- CMake (3.13 or later)
- nvjpeg2k package


## Build command on Linux
```
$ mkdir build
$
$ cd build 
$
$ export CUDACXX= nvcc path
$
$ cmake ..  -DCMAKE_BUILD_TYPE=Release -DNVJPEG2K_PATH= nvjpeg2k location
#
# example  cmake .. -DCMAKE_BUILD_TYPE=Release -DNVJPEG2K_PATH=~/nvJPEG2kDecodeSample/libnvjpeg_2k
#
$ make
```



# Usage
./nvjpeg2k_dec_pipelined -h

```
Usage: ./nvjpeg2k_dec_pipelined -i images_dir [-b batch_size] [-t total_images] [-w warmup_iterations] [-o output_dir] Parameters: 
	images_dir	:	Path to single image or directory of images
	batch_size	:	Decode images from input by batches of specified size
	total_images	:	Decode these many images, if there are fewer images 
				in the input than total images, decoder will loop over the input
	warmup_iterations:	Run these many batches first without measuring performance
	output_dir	:	Write decoded images in BMP/PGM format to this directory

```