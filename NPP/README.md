# NVIDIA Performance Primitives (NPP) Library API Examples

## Overview

This repository showcases example applications that utilize the NVIDIA Performance Primitives (NPP) library for high-performance image processing on NVIDIA GPUs.

## Image Processing View

NPP covers the stages of a typical GPU image processing pipeline. The samples are organized below by stage.

```
 decode        pre-process          analyze / detect                 post-process / measure
 (nvJPEG) --> color convert, --> edges, lines, template     --> labeling, contours, distance
              resize, filter       matching, segmentation        transform, statistics
```

| Stage | What it does | NPP functions used in the samples | Sample |
|-------|--------------|-----------------------------------|--------|
| Color conversion and geometry | RGB to gray, resize | `nppiRGBToGray`, `nppiResize` | [opencvBenchmark](opencvBenchmark/) |
| Filtering | Gaussian smoothing | `nppiFilterGaussBorder` | [opencvBenchmark](opencvBenchmark/) |
| Edge detection | Gradient-based edges, color and gray | `nppiFilterCannyBorder` | [nppCanny](nppCanny/), [houghLines](houghLines/) |
| Feature detection | Straight lines | `nppiFilterHoughLine` | [houghLines](houghLines/) |
| Template matching | Locate a pattern, robust to gain and offset | `nppiCrossCorrValid_NormLevel`, `nppiMaxIndx` | [templateMatching](templateMatching/) |
| Segmentation | Region labeling, watershed | `nppiLabelMarkersUF`, `nppiCompressMarkerLabelsUF`, `nppiSegmentWatershed` | [batchedLabelMarkersAndCompression](batchedLabelMarkersAndCompression/), [watershedSegmentation](watershedSegmentation/) |
| Shape analysis | Contours, Euclidean distance transform | `nppiCompressedMarkerLabelsUFInfo`, `nppiDistanceTransformPBA` | [findContour](findContour/), [distanceTransform](distanceTransform/) |
| Pipeline efficiency | Scratch buffer handling across frames, CPU comparison | `nppiWorkspaceCreate`, `_Ctx` calls | [opencvBenchmark](opencvBenchmark/) |

## Examples Included

- [3-Channel Canny Edge Detection](nppCanny/)
  High-performance color edge detection using NPP's unified 3-channel Canny algorithm. 

- [Batched Label Markers and Compression](batchedLabelMarkersAndCompression/)
  Demonstrates how to perform connected region labeling and compress marker labels across a batch of images.

- [Watershed Segmentation](watershedSegmentation/)
  Applies the watershed segmentation algorithm for image segmentation using NPP utilities.

- [Euclidean Distance Transform](distanceTransform/)
  Computes the Euclidean distance transform for binary images.

- [Find Contour](findContour/)
  Extracts contours from labeled image regions using NPP primitives.

- [Template Matching](templateMatching/)
  Locates a template in a large image with normalized cross-correlation and finds the best match on the GPU.

- [Hough Lines](houghLines/)
  Detects straight lines with Canny edge detection followed by the Hough line transform.

- [OpenCV Benchmark](opencvBenchmark/)
  Times Gaussian blur, Canny, resize and RGB to gray on NPP versus single- and multi-threaded OpenCV, kernel-only and end-to-end.

## Latest Developments in NPP (CUDA Toolkit 13.4, NPP 13.2.0.58)

### NPP Workspace

`NppStreamContext` now carries an optional `hWorkspace` handle (the former `nReserved0` field). Attach one with
`nppiWorkspaceCreate()` (declared in `nppworkspace.h`) and pass a `NULL` scratch buffer to the functions that support it:
NPP then reuses a single high-water-mark device buffer instead of the per-call
`GetBufferSize` + `cudaMalloc` + `cudaFree` sequence.

```cpp
NppStreamContext ctx = {};          // hWorkspace must be 0 unless a workspace was created
// ... fill in stream / device fields ...
nppiWorkspaceCreate(&ctx);
nppiLabelMarkersUF_8u32u_C1R_Ctx(src, srcStep, dst, dstStep, roi, nppiNormInf, NULL, ctx);
nppiWorkspaceDestroy(&ctx);
```

Samples using it: [nppCanny](nppCanny/), [findContour](findContour/), [watershedSegmentation](watershedSegmentation/),
[batchedLabelMarkersAndCompression](batchedLabelMarkersAndCompression/). With CUDA 13.4 the
`NULL` scratch buffer is accepted by `nppiLabelMarkersUF_*`, `nppiSegmentWatershed_*` and `nppiFilterCannyBorder_*`;
`nppiCompressMarkerLabelsUF_*` and `nppiDistanceTransformPBA_*` return `NPP_NULL_POINTER_ERROR` for a `NULL` buffer in that release
(later NPP builds accept it). [findContour](findContour/), [batchedLabelMarkersAndCompression](batchedLabelMarkersAndCompression/) and
[distanceTransform](distanceTransform/) therefore try the workspace first and allocate the scratch buffer only if NPP reports that error,
so they work with both.
Applications that do not use a workspace must zero-initialize `NppStreamContext` (as `distanceTransform` now does).
The samples check for `nppworkspace.h` with `__has_include`
and fall back to allocating the scratch buffers themselves, so they also build with CUDA toolkits that predate it.

### Earlier: NPP 13.0.3.9 (CUDA Toolkit 13.0 Update 3)

#### Deprecations

- **Non-CTX APIs** are deprecating and will be removed in **CUDA Toolkit 13.0**.  
- The `nppGetStreamContext()` API will be deprecated starting with CUDA Toolkit 13.0. Developers should migrate to **application-managed stream contexts** as described in the [NPP Documentation – General Conventions](https://docs.nvidia.com/cuda/npp/introduction.html#general-conventions).

#### Resolved Issues

- `nppiTranspose` now supports larger dimensions.  
- Fixed a distance calculation bug in `nppiDistanceTransformPBA_8u16u_C1R_Ctx`.  
- Improved performance of `nppiResizeSqrPixel` and `nppiCrossCorrelation`.  
- `nppiYUVToRGB_8u_C3R` now supports block-linear formatted input.  
- Fixed output pitch issue in `nppiFilterGaussAdvanced`.

## Platform Support

- **Operating Systems**: Linux, Windows  
- **CPU Architectures**: x86_64  
- **Required Toolkit**: [CUDA 13.4 or newer](https://developer.nvidia.com/cuda-downloads)

## CUDA API Reference

- [NPP Library Documentation](https://docs.nvidia.com/cuda/npp/index.html)

## Application-Managed Context and Stream Handling in NPP

The `NppStreamContext` structure was introduced in NPP version 10.1, corresponding to CUDA Toolkit 10.1 (released in early 2019). 
This marked the beginning of support for application-managed stream contexts, allowing developers to explicitly manage CUDA streams for more flexible and concurrent GPU workloads. 

Use of the NppStreamContext structure is strongly recommended in place of the deprecated nppGetStreamContext() API, which is scheduled for removal in CUDA Toolkit 13.0 to support application-managed stream and context control.


### Example: Using Application-Managed Context

```cpp
NppStreamContext nppCtx = {};
nppCtx.hStream = myCudaStream; // Application-managed stream

// Set other fields as needed based on cudaDeviceProp

// Use in API call
nppiFilterGauss_8u_C1R_Ctx(..., nppCtx); 
```

For the most up-to-date details, refer to the official [NPP Documentation – Application-managed Context](https://docs.nvidia.com/cuda/npp/index.html#application-managed-context).