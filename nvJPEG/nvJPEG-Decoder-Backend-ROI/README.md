# nvJPEG-Decoder-Backend-ROI

## Description
The sample codes shows the usage of decouple nvJPEG API on decoding JPEG images. This sample help to use different backends and ROI API usage.

## Key Concepts
Image Decoding, different Backend from NVJPEG Library

## Supported SM Architecture 
SM 3.0 SM 3.5 SM 3.7 SM 5.0 SM 5.2 SM 6.0 SM 6.1 SM 7.0 SM 7.2 SM 7.5 8.0

## Supported OS
Linux, Windows

## Supported CPU Architecture
x86_64

## CUDA APIs involved
NVJPEG

# Building
Build command
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

# Usage
```
Usage: ./nvJPEGROIDecode -i images_dir [-roi roi_regions] [-backend backend_enum] [-b batch_size] [-t total_images] [-w warmup_iterations] [-o output_dir] [-pipelined] [-batched] [-fmt output_format]
Parameters: 
        images_dir      :       Path to single image or directory of images
        roi_regions     :       Specify the ROI in the following format [x_offset, y_offset, roi_width, roi_height]
        backend_eum     :       Type of backend for the nvJPEG (0 - NVJPEG_BACKEND_DEFAULT, 1 - NVJPEG_BACKEND_HYBRID,
                                2 - NVJPEG_BACKEND_GPU_HYBRID)
        batch_size      :       Decode images from input by batches of specified size
        total_images    :       Decode this much images, if there are less images 
                                        in the input than total images, decoder will loop over the input
        warmup_iterations       :       Run this amount of batches first without measuring performance
        output_dir      :       Write decoded images as BMPs to this directory
        pipelined       :       Use decoding in phases
        batched         :       Use batched interface
        output_format   :       nvJPEG output format for decoding. One of [rgb, rgbi, bgr, bgri, yuv, y, unchanged]
```




