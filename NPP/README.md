# NVIDIA Performance Primitives (NPP) Library API Examples

## Overview

This repository showcases example applications that utilize the NVIDIA Performance Primitives (NPP) library for high-performance image processing on NVIDIA GPUs.

## Key Concepts

- GPU-accelerated image processing
- Connected region labeling
- Watershed segmentation
- Contour detection
- Distance transform

## Examples Included

- [Batched Label Markers and Compression](batchedLabelMarkersAndCompression/)  
  Demonstrates how to perform connected region labeling and compress marker labels across a batch of images.

- [Watershed Segmentation](watershedSegmentation/)  
  Applies the watershed segmentation algorithm for image segmentation using NPP utilities.

- [Euclidean Distance Transform](distanceTransform/)  
  Computes the Euclidean distance transform for binary images.

- [Find Contour](findContour/)  
  Extracts contours from labeled image regions using NPP primitives.

## Latest Developments in NPP (12.4.0.0)

### Deprecations

- **Non-CTX APIs** are deprecating and will be removed in **CUDA Toolkit 13.0**.  
- The `nppGetStreamContext()` API will be deprecated starting with CUDA Toolkit 13.0. Developers should migrate to **application-managed stream contexts** as described in the [NPP Documentation – General Conventions](https://docs.nvidia.com/cuda/npp/introduction.html#general-conventions).

### Resolved Issues

- `nppiTranspose` now supports larger dimensions.  
- Fixed a distance calculation bug in `nppiDistanceTransformPBA_8u16u_C1R_Ctx`.  
- Improved performance of `nppiResizeSqrPixel` and `nppiCrossCorrelation`.  
- `nppiYUVToRGB_8u_C3R` now supports block-linear formatted input.  
- Fixed output pitch issue in `nppiFilterGaussAdvanced`.  

## Platform Support

- **Operating Systems**: Linux, Windows  
- **CPU Architectures**: x86_64  
- **Required Toolkit**: [CUDA 12.0 or newer](https://developer.nvidia.com/cuda-downloads)

## CUDA API Reference

- [NPP Library Documentation](https://docs.nvidia.com/cuda/npp/index.html)

## Application-Managed Context and Stream Handling in NPP

The `NppStreamContext` structure was introduced in NPP version 10.1, corresponding to CUDA Toolkit 10.1 (released in early 2019). 
This marked the beginning of support for application-managed stream contexts, allowing developers to explicitly manage CUDA streams for more flexible and concurrent GPU workloads. 

Use of the NppStreamContext structure is strongly recommended in place of the deprecated nppGetStreamContext() API, which is scheduled for removal in CUDA Toolkit 13.0 to support application-managed stream and context control.


### Example: Using Application-Managed Context

```cpp
NppStreamContext nppCtx;
nppCtx.hStream = myCudaStream; // Application-managed stream

// Set other fields as needed based on cudaDeviceProp

// Use in API call
nppiFilterGauss_8u_C1R_Ctx(..., nppCtx); 
```

For the most up-to-date details, refer to the official [NPP Documentation – Application-managed Context](https://docs.nvidia.com/cuda/npp/index.html#application-managed-context).