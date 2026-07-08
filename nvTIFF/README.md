# nvTIFF Library API examples

## Description

Samples demonstrating GPU-accelerated TIFF decoding and encoding with the
[nvTIFF library](https://docs.nvidia.com/cuda/nvtiff/index.html) (0.8 API).

## Examples

| Sample | Workload | Key APIs |
|---|---|---|
| [nvTIFF-Decode](nvTIFF-Decode/) | Minimal first-image decode and PNM preview | `nvtiffStreamOpenFromFile`, `nvtiffStreamGetImageInfo`, `nvtiffDecode` |
| [nvTIFF-Decode-Encode](nvTIFF-Decode-Encode/) | Minimal decode → encode roundtrip | `nvtiffStreamOpenFromFile`, `nvtiffDecode`, `nvtiffEncode`, `nvtiffWriteTiffFile` |
| [nvTIFF-Image-Info-Multi-Image](nvTIFF-Image-Info-Multi-Image/) | Multi-page / pyramid navigation, metadata, SubIFDs | `nvtiffStreamGetHeader`, `nvtiffStreamGetNextIFDOffset`, tag APIs |
| [nvTIFF-Batched-Region-Decode](nvTIFF-Batched-Region-Decode/) | Many ROIs in one decode call (patch extraction, tile serving) | `nvtiffDecodeRegion_t`, `nvtiffDecodeParamsSetRegions`, `NVTIFF_STATUS_BATCH_INCOMPATIBLE` |
| [nvTIFF-Encode-Options](nvTIFF-Encode-Options/) | Strip/tile geometry, JPEG options, BigTIFF, custom tags, encode-to-memory | `nvtiffEncodeParamsSetImageGeometry`, `SetJpegOptions`, `SetTiffVariant`, `SetTag`, `nvtiffWriteTiffBuffer` |
| [nvTIFF-GeoTIFF-Decode-Encode](nvTIFF-GeoTIFF-Decode-Encode/) | Georeferencing roundtrip | geo key getters and setters, `NVTIFF_TAG_MODEL_*` |

## Test data

`images/bali_notiles.tif` ships with the samples and works with every decode
sample. `nvTIFF-Encode-Options` generates tiled,
multi-page, JPEG, BigTIFF and custom-tag files that the other samples accept
as input. For real-world large tiled / pyramidal images see the
[OpenSlide test data](https://openslide.cs.cmu.edu/download/openslide-testdata/)
(note the per-file licenses); GeoTIFF inputs can be created with GDAL (see
the GeoTIFF sample README).

## Supported SM Architectures

[SM 7.0 +](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, arm64-sbsa, aarch64-jetson

## Prerequisites

- A Linux or Windows system with recent NVIDIA drivers.
- The [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
- [nvTIFF](https://developer.nvidia.com/nvtiff-downloads) 0.8 or later.
- Optional runtime dependencies, loaded on demand:
  [nvCOMP](https://developer.nvidia.com/nvcomp-download) for Deflate,
  nvJPEG (part of the CUDA toolkit) for JPEG, and
  [nvJPEG2000](https://developer.nvidia.com/nvjpeg2000-downloads) for
  JPEG2000/Aperio streams.

Each sample builds standalone:

```
cd <sample>
cmake -B build -DNVTIFF_PATH=/path/to/nvtiff/cuda13   # archive dir or install prefix
cmake --build build
```
