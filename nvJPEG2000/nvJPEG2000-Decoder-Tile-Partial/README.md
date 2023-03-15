# JPEG 2000 Partial Image decoding Example using nvJPEG2000 Library

## Description

This code demonstrates how to partially decode a multi tile image, when the decode window spans multiple tiles

## Key Concepts

Image Decoding from nvJPEG2000 Library

## Supported SM Architectures

  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64

## API Documentation

[nvJPEG2000](https://docs.nvidia.com/cuda/nvjpeg2000/index.html)


# Prerequisites
- Recent NVIDIA drivers.
- [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
- CMake (3.17 or later).
- Install [nvJPEG2000](https://developer.nvidia.com/nvjpeg2000/downloads).


## Build Instructions on Linux
```
$ mkdir build
$
$ cd build 
$
$ export CUDACXX=/usr/local/cuda/bin/nvcc
$
$ cmake ..  -DCMAKE_BUILD_TYPE=Release
# Use -DNVJPEG2K_PATH, if nvJPEG2000 is not installed at the default location.
# example  cmake .. -DCMAKE_BUILD_TYPE=Release -DNVJPEG2K_PATH=/opt/libnvjpeg_2k
#
$ make
```



# Usage
./nvj2k_decode_tile_partial -h

```
Usage: ./nvj2k_decode_tile_partial -i images_dir [-b batch_size] [-t total_images] [-w warmup_iterations] [-o output_dir] [-da x0,y0,x1,y1]Parameters: 
	images_dir	:	Path to single image or directory of images
	batch_size	:	Decode images from input by batches of specified size
	total_images	:	Decode these many images, if there are fewer images 
				in the input than total images, decoder will loop over the input
	warmup_iterations:	Run these many batches first without measuring performance
	output_dir	:	Write decoded images in BMP/PGM format to this directory
	-da x0,y0,x1,y1 : Decode Area of Interest. The  coordinates are relative to the image origin

```

