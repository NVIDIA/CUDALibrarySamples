# JPEG 2000 Encode example using nvJPEG2000 Library

## Description

This sample demonstrates how to create a jpeg 2000 compressed bitstream using nvJPEG2000 library

## Key Concepts

JPEG 2000 encoding

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
./nvjpeg2k_encode -h

```
Usage: ./build/nvjpeg2k_encode -i images_dir [-b batch_size] [-t total_images] [-I] [-cblk cblk_w,cblk_h]
        [-w warmup_iterations] [-o output_dir] 
        [-img_fmt img_w,img_h,num_comp,precision,chromaformat] (-img_fmt is mandatory for raw yuv files)
        eg: for an 8 bit image of size 1920x1080 with 420 subsamling: -img-dims 1920,1080,3,8,chroma420
Parameters: 
        images_dir      :       Path to single image or directory of images
        batch_size      :       Encode images from input by batches of specified size
        total_images    :       Encode these many images, if there are fewer images 
                                in the input than total images, encoder will loop over the input
                -I      :       Enable irreversible wavelet transform
        cblk_w,cblk_h   :       Code block width and code block height
                                valid values are 32,32 and 64,64 
        warmup_iterations:      Run these many batches first without measuring performance
        output_dir      :       Write compressed jpeg 2000 files to this directory

```


Example:

Sample example on GV100, Ubuntu 18.04, CUDA 11.2

```
$ ./nvjpeg2k_encode -i ../images/ -b 1 -t 1024 -w 4 -o .
```

```
Encoding images in directory: ../images/, total 1024, batchsize 1
Total encode time: 18.1703
Avg encode time per image: 0.0177444
Avg encode speed  (in images per sec): 56.3558
Avg encode time per batch: 0.0177444

```
