# TIFF Single ROI Image Decoding using nvTIFF Library

## Description

This code demonstrates Single Image and ROI decoding using nvTIFF library.

## Key Concepts

GPU accelerated TIFF Image decoding

## Supported SM Architectures

  [SM 6.0 + ](https://developer.nvidia.com/cuda-gpus) 

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, arm64-sbsa, aarch64-jetson

## APIs involved

[nvTIFF](https://docs.nvidia.com/cuda/nvtiff/index.html)


# Building (make)

# Prerequisites
- A Linux system with recent NVIDIA drivers.
- Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
- CMake (3.17 or later)
- [nvTIFF] (https://developer.nvidia.com/nvtiff-downloads).
- [nvComp] (https://developer.nvidia.com/nvcomp-download)

## Build command on Linux
```
$ mkdir build
$
$ cd build 
$
$ export CUDACXX=<nvcc location>
$
$ cmake ..  -DNVTIFF_PATH=<nvTIFF Library location>
#
# example  cmake .. -DNVTIFF_PATH=~/project/cudalibrarysamples-mirror/nvTIFF/libnvtiff
#
$ make
```

# Usage


Usage:
./nvtiff_decode_image -h

```
./nvtiff_decode_image -f <TIFF_FILE> [-image_id image_id]  [-o output_dir] [-roi offset_x, offset_y, roi_width, roi_height]
Parameters: 
    <TIFF_FILE>   :  TIFF file to decode.
    image_id      :  Image index(IFD location) within a TIFF file. Defaults to 0.
    output_dir	:	Write decoded images in PNM format to this directory.
    offset_x, offset_y, roi_width, roi_height : Region of interest coordinates for decoding.

```
