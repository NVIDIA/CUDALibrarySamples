# JPEG Image decoding Example using nvJPEG

## Description

This code demonstrates JPEG Image decoding utility using nvJPEG library.

## Key Concepts

Image Decoding from NVJPEG Library

## Supported SM Architectures

[SM 3.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux Windows

## Supported CPU Architecture

x86_64

## CUDA APIs involved
[NPP](https://docs.nvidia.com/cuda/npp/group__image__resize.html)
[NVJPEG](https://docs.nvidia.com/cuda/nvjpeg/index.html)


# Architecture
- JPEG decoding is handled by nvJPEG.

# Building (make)

# Prerequisites
- A Linux/Windows system with recent NVIDIA drivers.
- Install the [CUDA 10.1 toolkit](https://developer.nvidia.com/cuda-downloads).

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
$ Open imageResize.sln project in Visual Studio 15 2017 and build
```




# Usage
./nvjpegDecoder -h

```
Usage: ./nvjpegDecoder -i images_dir [-b batch_size] [-t total_images] [-w warmup_iterations] [-o output_dir] [-pipelined] [-batched] [-fmt output_format]
Parameters: 
	images_dir	:	Path to single image or directory of images
	batch_size	:	Decode images from input by batches of specified size
	total_images	:	Decode this much images, if there are less images 
					in the input than total images, decoder will loop over the input
	warmup_iterations	:	Run this amount of batches first without measuring performance
	output_dir	:	Write decoded images as BMPs to this directory
	pipelined	:	Use decoding in phases
	batched		:	Use batched interface
	output_format	:	nvJPEG output format for decoding. One of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]

```
Example:

Sample example output on GV100, Ubuntu 18.04, CUDA 10.2

```
$  ./nvjpegDecoder -i ../input_images/ -o ~/tmp
```
```
./build/nvjpegDecoder -i input_images/ -o ~/tmp 
Decoding images in directory: input_images/, total 12, batchsize 1
Processing: input_images/cat_baseline.jpg
Image is 3 channels.
Channel #0 size: 64 x 64
Channel #1 size: 64 x 64
Channel #2 size: 64 x 64
YUV 4:4:4 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/cat_baseline.bmp
Processing: input_images/img8.jpg
Image is 3 channels.
Channel #0 size: 480 x 640
Channel #1 size: 240 x 320
Channel #2 size: 240 x 320
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img8.bmp
Processing: input_images/img5.jpg
Image is 3 channels.
Channel #0 size: 640 x 480
Channel #1 size: 320 x 240
Channel #2 size: 320 x 240
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img5.bmp
Processing: input_images/img7.jpg
Image is 3 channels.
Channel #0 size: 480 x 640
Channel #1 size: 240 x 320
Channel #2 size: 240 x 320
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img7.bmp
Processing: input_images/img2.jpg
Image is 3 channels.
Channel #0 size: 480 x 640
Channel #1 size: 240 x 320
Channel #2 size: 240 x 320
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img2.bmp
Processing: input_images/img4.jpg
Image is 3 channels.
Channel #0 size: 640 x 426
Channel #1 size: 320 x 213
Channel #2 size: 320 x 213
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img4.bmp
Processing: input_images/cat.jpg
Image is 3 channels.
Channel #0 size: 64 x 64
Channel #1 size: 64 x 64
Channel #2 size: 64 x 64
YUV 4:4:4 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/cat.bmp
Processing: input_images/cat_grayscale.jpg
Image is 1 channels.
Channel #0 size: 64 x 64
Grayscale JPEG 
Done writing decoded image to file: /home/mahesh/tmp/cat_grayscale.bmp
Processing: input_images/img1.jpg
Image is 3 channels.
Channel #0 size: 480 x 640
Channel #1 size: 240 x 320
Channel #2 size: 240 x 320
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img1.bmp
Processing: input_images/img3.jpg
Image is 3 channels.
Channel #0 size: 640 x 426
Channel #1 size: 320 x 213
Channel #2 size: 320 x 213
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img3.bmp
Processing: input_images/img9.jpg
Image is 3 channels.
Channel #0 size: 640 x 480
Channel #1 size: 320 x 240
Channel #2 size: 320 x 240
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img9.bmp
Processing: input_images/img6.jpg
Image is 3 channels.
Channel #0 size: 640 x 480
Channel #1 size: 320 x 240
Channel #2 size: 320 x 240
YUV 4:2:0 chroma subsampling
Done writing decoded image to file: /home/mahesh/tmp/img6.bmp
Total decoding time: 15.1316
Avg decoding time per image: 1.26097
Avg images per sec: 0.79304
Avg decoding time per batch: 1.26097

```

