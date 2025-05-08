# Watershed Image Segmentation with NPP

## Overview

This project demonstrates grayscale image segmentation using the watershed algorithm implemented through NVIDIA’s NPP library. The technique is based on the GPU-accelerated approach detailed in the paper:

> "Efficient 2D and 3D Watershed on Graphics Processing Unit: Block-Asynchronous Approaches Based on Cellular Automata" by Pablo Quesada-Barriuso et al.

## Key Concepts
- Image Segmentation
- Watershed Transform
- NPP (NVIDIA Performance Primitives)

## Supported Platforms
- **Operating System:** Linux or Windows
- **CPU Architecture:** x86_64
- **GPU Support:** [CUDA-enabled GPUs (SM 7.0, 7.2, 7.5, 8.0, and above)](https://developer.nvidia.com/cuda-gpus)
- **Toolkit:** [CUDA Toolkit 11.5 or later](https://developer.nvidia.com/cuda-downloads)
- **Libraries:** NPP, CMake, and CUDA Runtime

---

## Build Instructions

### Linux
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### Windows
```bash
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
# Open the generated `watershedSegmentation.sln` in Visual Studio 2017 and build the solution
```

---

## Usage
```bash
./watershedSegmentation [-b number-of-batch]
```

### Parameters
- `-b <number>`: Number of images to process in a batch (default is 3).

### Example
```bash
./watershedSegmentation -b 3
```

---

## Output Log
```text
NPP Library Version 11.0.0
CUDA Driver  Version: 11.0
CUDA Runtime Version: 11.0

Input file load succeeded.
Lena_Segments_8Way_512x512_8u succeeded.
Lena_CompressedSegmentLabels_8Way_512x512_32u succeeded.
Lena_SegmentBoundaries_8Way_512x512_8u succeeded.
Lena_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.

Input file load succeeded.
CT_Skull_Segments_8Way_512x512_8u succeeded.
CT_Skull_CompressedSegmentLabels_8Way_512x512_32u succeeded.
CT_Skull_SegmentBoundaries_8Way_512x512_8u succeeded.
CT_Skull_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.

Input file load succeeded.
Rocks_Segments_8Way_512x512_8u succeeded.
Rocks_CompressedSegmentLabels_8Way_512x512_32u succeeded.
Rocks_SegmentBoundaries_8Way_512x512_8u succeeded.
Rocks_SegmentsWithContrastingBoundaries_8Way_512x512_8u succeeded.
```

---

## Credits
- Sample provided by NVIDIA Corporation.
- Based on academic work cited above.
