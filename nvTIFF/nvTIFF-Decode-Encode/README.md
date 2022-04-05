# TIFF Image decoding / encoding Example using nvTIFF Library

## Description

This code demonstrates TIFF Image decoding / encoding using nvTIFF library.

## Key Concepts

Image decoding and encoding from NVTIFF Library

## Supported SM Architectures

  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus) [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved

[NVTIFF](https://docs.nvidia.com/cuda/nvTIFF/index.html)


# Building (make)

# Prerequisites
- A Linux system with recent NVIDIA drivers.
- Install the [CUDA 11.6 toolkit](https://developer.nvidia.com/cuda-downloads).
- CMake (3.13 or later)
- nvTIFF package


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
nvTiff_example [options] -f|--file <TIFF_FILE>

General options:

        -d DEVICE_ID
        --device DEVICE_ID
                Specifies the GPU to use for images decoding/encoding.
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
                tiff files with the following limitations:                                      
                  * color space must be either Grayscale (PhotometricInterp.=1) or RGB (=2)     
                  * image data compressed with LZW (Compression=5) or uncompressed              
                  * pixel components stored in "chunky" format (RGB..., PlanarConfiguration=1)
                    for RGB images                                                              
                  * image data must be organized in Strips, not Tiles                           
                  * pixels of RGB images must be represented with at most 4 components 
                  * each component must be represented exactly with:
                  * 8 bits for LZW compressed images                                        
                  * 8, 16 or 32 bits for uncompressed images                                
                  * all images in the file must have the same properties                        

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

        -m
        --memtype TYPE
                Specifies the type of memory used to hold  the  TIFF  file  content:  pinned  or
                pageable.  Pinned memory is used if 'p' is specified. Pageable memory is used if
                'r' is specified.  In case of pinned memory,  file  content  is  not  copied  to
                device memory before the decoding process (with a resulting performance  impact)
                unless the option -c is also specified (see below).
                Defualt: r (pageable)

        -c
        --copyh2d
                Specifies to copy the file data to device memory in case the -m option specifies
                to use pinned memory.  In case of pageable memory this  option  has  no  effect.
                Default: off.

        --decode-out NUM_OUT
                Enables the writing of selected images from the decoded  input  TIFF  file  into
                separate BMP files for inspection.  If no argument is  passed,  only  the  first
                image is written to disk,  otherwise  the  first  NUM_OUT  images  are  written.
                Output files are named outImage_0.bmp, outImage_1.bmp...
                Defualt: disabled.

Encoding options:

        -E
        --encode
                This option enables the encoding of the raster images obtained by  decoding  the
                input TIFF file.  The images are divided into strips, compressed  with  LZW and,
                optionally, written into an output TIFF file.
                Default: disabled.

        -r
        --rowsxstrip
                Specifies the number of consecutive rows  to  use  to  divide  the  images  into
                strips.  Each image is divided in strips of the same size (except  possibly  the
                last strip) and then the strips are  compressed  as  independent  byte  streams.
                This option is ignored if -E is not specified.
                Default: 1.

        -s
        --stripalloc
                Specifies the initial estimate of the maximum size  of  compressed  strips.   If
                during compression one or more strips require more  space,  the  compression  is
                aborted and restarted automatically with a safe estimate. 
                This option is ignored if -E is not specified.
                Default: the size, in bytes, of a strip in the uncompressed images.

        --encode-out
                Enables the writing of the compressed  images  to  an  output  TIFF  file named
                outFile.tif.
                This option is ignored if -E is not specified.
                Defualt: disabled.


Example:

Sample example output on GV100, Ubuntu 16.04, CUDA 11.6

```
$ ./nvTiff_example -E -f ../images/bali_notiles.tif
```

```
Using GPU:
	 0 (Quadro GV100, 80 SMs, 2048 th/SM max, CC 7.0, ECC off)

Decoding 1, RGB 725x489 images [0, 0], from file ../images/bali_notiles.tif... done in 0.000328 secs

Encoding 1, RGB 725x489 images using 1 rows per strip and 2175 bytes per strip... done in 0.002686 secs (compr. ratio: 3.30x)

```
