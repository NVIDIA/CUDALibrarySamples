# nvJPEG2000 pipelined decoding example

## Description

This example decodes batches of JPEG 2000 images with the decodes of a batch overlapping. It first parses every image of
a batch into its own `nvjpeg2kStream_t`, then decodes them with `nvjpeg2kDecodeImage`, cycling through 10 decode states,
each with its own CUDA stream. A decode state is reused only after the CUDA event recorded for its previous decode
completes.

## Key concepts

- Parsing a batch of images before decoding them
- Running several decodes at once with one decode state and one CUDA stream per decode
- Reusing a decode state only once its previous decode has finished

## What this example handles

nvJPEG2000 itself supports more; see [what nvJPEG2000 supports](../README.md#what-nvjpeg2000-supports).

- Input: .jp2 files and JPEG 2000 codestreams, Part 1 or HTJ2K. A directory is searched recursively, and every file in it
  is decoded.
- Up to 4 components; images with more exit with code 5.
- Integer images up to 16 bits per component, and NLT type 3 images as 16- or 32-bit floating point.
- 4:2:0 and 4:2:2 images are always converted to RGB.
- Output files (`-o`): .pgm for 1 component and .bmp for 3 or 4 components (the fourth component is dropped and samples
  above 8 bits are reduced to 8 bits; .ppm when the .bmp would exceed 4 GB). NLT type 3 images are written as .pfm. With
  `-o`, 2-component images and signed 3- or 4-component images are decoded but not written, and the example exits
  with code 5.

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
$ ./nvjpeg2k_dec_pipelined -h
Usage: ./nvjpeg2k_dec_pipelined -i images_dir [-b batch_size] [-t total_images] [-w warmup_iterations] [-o output_dir] [-v]
Parameters:
	images_dir	:	Path to a single image or a directory of images (searched recursively),
				as .jp2 files or JPEG 2000 codestreams, Part 1 or HTJ2K
	batch_size	:	Decode images from input by batches of specified size
	total_images	:	Decode these many images, if there are fewer images
				in the input than total images, decoder will loop over the input
	warmup_iterations:	Run these many batches first without measuring performance
	output_dir	:	Write decoded images to this directory: 1-component images as .pgm,
				3- and 4-component images as .bmp (alpha is dropped; .ppm above 4 GB),
				and NLT type 3 images as .pfm;
				4:2:0 and 4:2:2 images are converted to RGB
	-v		:	Log verbose messages to console
Exit codes: 0 success, 2 invalid arguments, 3 missing input, 4 empty input directory,
	5 image not supported by this sample, 6 decode or write failure, 1 any other error
```

## Exit codes

The example returns the [exit codes shared by all examples](../README.md#exit-codes). Code 5 means an image it cannot
handle, as described above.

## Example

On Linux, from the build directory. The timings depend on the GPU. The `-o` directory must already exist (the
example does not create it, and exits with code 6 if it is missing); here it is the build directory itself.

```
$ ./nvjpeg2k_dec_pipelined -i ../../nvJPEG2000-Decoder/images -b 3 -o .
Decoding images in directory: ../../nvJPEG2000-Decoder/images, total 3, batchsize 3
Total decoding time: 0.106382
Avg decoding time per image: 0.0354607
Avg decode speed  (in images per sec): 28.2002
Avg decoding time per batch: 0.106382
```

This decodes the three images of the Decoder example as one batch, and writes `2k_lossless.bmp`, `2k_lossy.bmp` and
`4k_lossy.bmp`.
