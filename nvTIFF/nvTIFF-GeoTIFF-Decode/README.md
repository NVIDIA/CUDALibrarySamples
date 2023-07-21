# TIFF Image decoding / encoding Example using nvTIFF Library

## Description

This code demonstrates GeoTIFF Image decoding using nvTIFF library.

## Key Concepts

GPU accelerated TIFF Image decoding

## Supported SM Architectures

  [SM 6.0 + ](https://developer.nvidia.com/cuda-gpus) 

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64

## APIs involved

[nvTIFF](https://docs.nvidia.com/cuda/nvtiff/index.html)


# Building (make)

# Prerequisites
- A Linux system with recent NVIDIA drivers.
- Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
- CMake (3.17 or later)
- nvTIFF package
- [nvComp] (https://developer.nvidia.com/nvcomp-download)

## Build command on Linux
```
$ mkdir build
$
$ cd build 
$
$ export CUDACXX=nvcc
$
$ cmake ..  -DNVTIFF_PATH=nvTIFF Library location
#
# example  cmake .. -DNVTIFF_PATH==~/project/cudalibrarysamples-mirror/nvTIFF/libnvtiff
#
$ make
```



# Usage
Usage:
nvtiff_geotiff_decode [options] -f|--file <TIFF_FILE>

Usage:
./build/nvtiff_geotiff_decode [options] -f|--file <TIFF_FILE>

```
Usage:
./build/nvtiff_geotiff_decode [options] -f|--file <TIFF_FILE>

General options:

        -d DEVICE_ID
        --device DEVICE_ID
                Specifies the GPU to use for images decoding.
                Default: device 0 is used.

        -v
        --verbose
                Prints some information about the decoded TIFF file.

        -h
        --help
                Prints this help

Decoding options:

        -f TIFF_FILE
        --file TIFF_FILE
                Specifies the TIFF file to decode. The code supports both single and multi-image

        -b BEG_FRM
        --frame-beg BEG_FRM
                Specifies the image id in the input TIFF file to start decoding from.  The image
                id must be a value between 0 and the total number of images in the file minus 1.
                Values less than 0 are clamped to 0.
                Default: 0

        -e END_FRM
        --frame-end END_FRM
                Specifies the image id in the input TIFF file to stop  decoding  at  (included).
                The image id must be a value between 0 and the total number  of  images  in  the
                file minus 1.  Values greater than num_images-1  are  clamped  to  num_images-1.
                Default:  num_images-1.

        --decode-out NUM_OUT
                Enables the writing of selected images from the decoded  input  TIFF  file  into
                separate PNM files for inspection.  If no argument is  passed,  only  the  first
                image is written to disk,  otherwise  the  first  NUM_OUT  images  are  written.
                Output files are named <in_filename>_nvtiff_out_0.(ppm/pgm), 
                <in_filename>_nvtiff_out_1.(ppm/pgm)....
                Default: disabled.

```

Example:

Sample example output on GV100, Ubuntu 22.04, CUDA 12.1

```
$ ./nvTiff_example -E -f ../images/bali_notiles.tif
```

```
Using GPU:
	 0 (Quadro GV100, 80 SMs, 2048 th/SM max, CC 7.0, ECC off)

Decoding 1, RGB 725x489 images [0, 0], from file ../images/bali_notiles.tif... done in 0.000328 secs

Encoding 1, RGB 725x489 images using 1 rows per strip and 2175 bytes per strip... done in 0.002686 secs (compr. ratio: 3.30x)

```
