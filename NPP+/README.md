# Nvidia Performance Primitives (NPP+) Library API Examples

## Overview

This repository demonstrates the usage of the **Nvidia Performance Primitives (NPP+)** library for high-performance image processing tasks. NPP+ provides a wide range of performance-optimized functions for image, video, and signal processing, making it ideal for use cases that require efficient computation on large datasets using Nvidia GPUs.

The examples provided in this repository highlight key NPP+ functions for various image processing tasks, enabling developers to understand how to integrate NPP+ into their own projects.

## Key Concepts

The examples cover the following core concepts using NPP+:

- **Flood Fill Algorithms**: Efficiently filling connected image regions starting from a seed point.
- **Contour Detection**: Detecting the boundaries of objects within an image.
- **Euclidean Distance Transform**: Calculating the Euclidean distance from image elements to the nearest object.
- **Watershed-based Image Segmentation**: Dividing an image into regions based on intensity gradients.
- **Labeling and Compression of Image Regions**: Labeling connected regions and compressing them for optimized processing.

Each example demonstrates the practical application of these high-performance primitives.

## Getting Started with NPP+

To begin using NPP+, download the latest version of the library and install it on your Nvidia GPU-equipped machine.

- [Download NPP+](https://developer.nvidia.com/nppplus-downloads)

### Prerequisites

To run these examples, ensure the following dependencies are installed:

- A Linux or Windows system with an Nvidia GPU.
- The latest [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx).
- [CUDA 12.0 Toolkit](https://developer.nvidia.com/cuda-downloads).
- [CMake](https://cmake.org/) for building the samples.
- [NPP+](https://developer.nvidia.com/nppplus-downloads).

## Examples

Explore the following examples to see how NPP+ APIs are applied in various image processing tasks:

- [**Flood Fill Sample**](floodFill/): Demonstrates the flood fill algorithm using NPP+ to fill connected image regions.
- [**Find Contour Sample**](findContour/): Shows how to find image contours efficiently using NPP+.
- [**Image Euclidean Distance Transform**](distanceTransform/): Applies the Euclidean distance transform to images.
- [**Watershed Segmentation**](watershedSegmentation/): Performs image segmentation using the watershed algorithm.
- [**Batched Label Markers and Compression**](batchedLabelMarkersAndCompression/): Explains how to label connected regions and apply compression to labeled image regions.

Each of these examples provides code snippets, performance benchmarks, and insights into how NPP+ can be leveraged for GPU-accelerated image processing.

## System Requirements

### Supported SM Architectures

The examples in this repository are optimized for the following Nvidia GPU architectures:

- [SM 7.0](https://developer.nvidia.com/cuda-gpus)
- [SM 7.2](https://developer.nvidia.com/cuda-gpus)
- [SM 7.5](https://developer.nvidia.com/cuda-gpus)
- [SM 8.0](https://developer.nvidia.com/cuda-gpus)

### Supported Operating Systems

- Linux
- Windows

### Supported CPU Architecture

- x86_64

## NPP+ CUDA APIs Involved

This repository primarily demonstrates the use of the [NPP+](https://docs.nvidia.com/cuda/nppplus/introduction.html) library.

