# nvJPEG2000 Encoder example

## Description

This example encodes images to JPEG 2000 with the nvJPEG2000 library. It reads BMP, PGM/PPM, PFM or raw YUV images,
copies them to device memory, encodes them with `nvjpeg2kEncode`, and writes the resulting bitstreams as .jp2 files.

## Key concepts

- Describing an image and its components with `nvjpeg2kEncodeConfig_t`
- Lossless (reversible wavelet) and lossy (irreversible wavelet) encoding, with quality set by Q-factor, quantization
  step or target PSNR
- High Throughput JPEG 2000 (HTJ2K) encoding
- Encoding floating-point images as HTJ2K with NLT type 3 (`nvjpeg2kEncodeParamsSetNLTInfo`)
- Retrieving the encoded bitstream with `nvjpeg2kEncodeRetrieveBitstream`

## What this example handles

nvJPEG2000 itself supports more; see [what nvJPEG2000 supports](../README.md#what-nvjpeg2000-supports).

- Input, chosen by file extension; a directory is searched recursively, and files of other types are skipped:
  - .bmp: 24-bit, uncompressed, encoded as 3-component RGB
  - .pgm and .ppm: binary (P5 and P6), 1 to 16 bits per component
  - .pfm: 1 or 3 components of floating-point data, encoded as HTJ2K with NLT type 3 and the reversible wavelet, as 16-
    or 32-bit floating point (`-pfm_precision`)
  - .yuv: raw planar data described by `-img_fmt`, with 1 to 4 components of 1 to 16 bits, and 4:4:4, 4:2:2 or 4:2:0
    chroma subsampling
- Code blocks of 32x32, 32x64, 64x32 or 64x64 (the default), and 128x32 for PFM input.
- Quality options need the irreversible wavelet (`-I`), and are not available for PFM input; target PSNR is not available
  with HTJ2K (`-ht`).
- Fixed settings: .jp2 output, LRCP progression order, 6 resolution levels, one tile, and the multi-component transform
  for RGB input.
- Output file names record the settings, for example `TestImage640x480_qt2_qv85.000000_irrevWavelet_HT_blksz64x64.jp2`
  for `-ht -I -q_factor 85`.

## Supported platforms

GPUs of compute capability 5.2 and higher with CUDA 12, and 7.5 and higher with CUDA 13, on Linux (x86_64, arm64 server
and Jetson) and Windows (x86_64). See [supported platforms](../README.md#supported-platforms) for details.

## Prerequisites

- An NVIDIA driver and CUDA Toolkit 12 or 13, the same CUDA major version as the nvJPEG2000 package
- CMake 3.18 or later
- A C++17 compiler: GCC on Linux, Visual Studio on Windows
- [nvJPEG2000](https://developer.nvidia.com/nvjpeg2000/downloads)

## Build

Linux:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DNVJPEG2K_PATH=/opt/libnvjpeg_2k
cmake --build build
```

Windows:
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DNVJPEG2K_PATH=C:\libnvjpeg_2k
cmake --build build --config Release
```

`NVJPEG2K_PATH` is the root of the nvJPEG2000 package, the directory holding `include` and `lib`; leave it out when
nvJPEG2000 is installed in a system location. If CMake cannot find nvcc, add the CUDA Toolkit's `bin` directory to
`PATH` or set `CUDACXX` to the nvcc path. On Windows, the executable needs `nvjpeg2k_0.dll` (in the package's
`lib\<CUDA major>` directory) and the CUDA runtime DLL on `PATH` to run.

To build all the examples together and check that they run, see [building all examples](../README.md#building-all-examples).

## Usage

```
$ ./nvjpeg2k_encode -h
Usage: ./nvjpeg2k_encode -i images_dir [-b batch_size] [-t total_images] [-w warmup_iterations] [-o output_dir]
	[-ht] [-I] [-q_factor value | -quantization value | -psnr value] [-cblk cblk_w,cblk_h]
	[-pfm_precision 16|32] [-img_fmt img_w,img_h,num_comp,precision,chromaformat]
Parameters:
	images_dir	:	Path to a single image or a directory of images (searched recursively):
				.bmp (24-bit, uncompressed), .pgm/.ppm (binary, 1 to 16 bits),
				.pfm (floating point), or raw .yuv (requires -img_fmt)
	batch_size	:	Encode images from input by batches of specified size
	total_images	:	Encode these many images, if there are fewer images
				in the input than total images, encoder will loop over the input
	warmup_iterations:	Run these many batches first without measuring performance
	output_dir	:	Write the encoded .jp2 files to this directory
	-ht		:	Enable High Throughput (HTJ2K) encoding
	-I		:	Enable irreversible wavelet transform. Must be set to use any quality option
	-q_factor	:	Set `value` as Q-Factor (jpeg-like) quality:
				floating point from 1.0 (worst) to 100.0 (best)
	-quantization	:	Set `value` as quantization step (by how much pixel data will be divided):
				floating point, bigger the value, the lower the result quality.
	-psnr		:	Set `value` as target PSNR value: positive floating point.
				Cannot be used with -ht option
	cblk_w,cblk_h	:	Code block width and height: 32x32, 32x64, 64x32 or 64x64 (default),
				or 128x32 for PFM input
	-pfm_precision	:	NLT bit depth for PFM input, 16 or 32 (default). PFM input is always
				encoded as HTJ2K with NLT type 3 and the reversible wavelet
	-img_fmt	:	Layout of raw .yuv input: width, height, number of components (1 to 4),
				precision (1 to 16 bits) and chroma format (chroma444, chroma422 or chroma420),
				e.g. -img_fmt 1920,1080,3,8,chroma420
Exit codes: 0 success, 2 invalid arguments, 3 missing input, 4 empty input directory,
	5 image not supported by this sample, 6 read, encode or write failure, 1 any other error
```

## Exit codes

The example returns the [exit codes shared by all examples](../README.md#exit-codes). Code 5 means an image it cannot
read, such as a compressed BMP.

## Example

On Linux, from the build directory. The timings depend on the GPU. The `-o` directory must already exist (the
example does not create it, and exits with code 6 if it is missing); here it is the build directory itself.

```
$ ./nvjpeg2k_encode -i ../images -o .
Encoding images in directory: ../images, total 1, batchsize 1
Total encode time: 0.033273
Avg encode time per image: 0.033273
Avg encode speed  (in images per sec): 30.0544
Avg encode time per batch: 0.033273
```

This encodes `TestImage640x480.bmp` losslessly to `TestImage640x480_qt0_qv0.000000_revWavelet_legacy_blksz64x64.jp2`.
A lossy HTJ2K encode:

```
$ ./nvjpeg2k_encode -i ../images -ht -I -q_factor 85 -o .
Encoding images in directory: ../images, total 1, batchsize 1
Total encode time: 0.0101492
Avg encode time per image: 0.0101492
Avg encode speed  (in images per sec): 98.5301
Avg encode time per batch: 0.0101492
```

This writes `TestImage640x480_qt2_qv85.000000_irrevWavelet_HT_blksz64x64.jp2`.
