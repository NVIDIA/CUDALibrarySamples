# nvTIFF Encode-Options Sample

Decodes one image to 8-bit RGB and re-encodes it nine ways using various encode options.

| Variant | Demonstrates |
|---|---|
| LZW, 128-row strips, 2 pages | `nvtiffEncodeParamsSetImageGeometry` (striped), multi-image encode via `nvtiffEncodeParamsSetInputs` |
| LZW, 256×256 tiles | `nvtiffEncodeParamsSetImageGeometry` (tiled; tile width and height must be multiples of 16) |
| JPEG q85, 4:4:4, optimized Huffman | `nvtiffEncodeParamsSetJpegOptions` (needs nvJPEG at runtime) |
| Uncompressed BigTIFF | `nvtiffEncodeParamsSetTiffVariant` |
| LZW + custom tags: ASCII (ImageDescription, Make, Model, Software), RATIONAL and SHORT (XResolution, YResolution, ResolutionUnit) | `nvtiffEncodeParamsSetTag` |
| LZW, encoded to a host buffer | `nvtiffWriteTiffBuffer` |
| LZW, predictor disabled | `nvtiffEncodeParamsSetPredictor` (default is enabled for lossless compression) |
| Deflate, 256×256 tiles | `NVTIFF_COMPRESSION_ADOBE_DEFLATE` (needs [nvCOMP](https://developer.nvidia.com/nvcomp-download) 6.0+ at runtime) |
| Zstandard, 256×256 tiles | `NVTIFF_COMPRESSION_ZSTD` (needs [nvCOMP](https://developer.nvidia.com/nvcomp-download) 6.0+ at runtime) |

The nine outputs are ordinary TIFF/BigTIFF files, and you may use them as inputs to the
other nvTIFF samples (multi-page for `nvTIFF-Image-Info-Multi-Image`, tiled
for `nvTIFF-Batched-Region-Decode`, ...).

If an optional library is not found at runtime ([nvCOMP](https://developer.nvidia.com/nvcomp-download) 6.0 or later for
Deflate and Zstandard, nvJPEG for JPEG), the affected variants are reported as
`skipped (nvCOMP 6 not found)` or `skipped (nvJPEG not found)` and the remaining variants still run.

## Build and run

```
cmake -B build -DNVTIFF_PATH=/path/to/nvtiff/cuda13
cmake --build build
export LD_LIBRARY_PATH=/path/to/nvtiff/cuda13/lib:$LD_LIBRARY_PATH
./build/nvtiff_encode_options ../images/bali_notiles.tif /tmp
```

Expected output:

```
decoded ../images/bali_notiles.tif: 725 x 489 -> RGB8

variant                              file                                bytes    ratio
LZW, 128-row strips, 2 pages         striped_lzw_multipage.tif          580396    3.66x
LZW, 256x256 tiles                   tiled_lzw.tif                      292750    3.63x
JPEG q85 4:4:4 optimized Huffman     jpeg_q85.tif                       271084    3.92x
uncompressed BigTIFF                 bigtiff_uncompressed.tif          1066455    1.00x
LZW, custom tags                     custom_tags.tif                    328204    3.24x
LZW, via host buffer                 lzw_from_buffer.tif                328004    3.24x
LZW, predictor disabled              lzw_no_predictor.tif               275287    3.86x
Deflate, 256x256 tiles               tiled_deflate.tif                  468780    2.27x
Zstandard, 256x256 tiles             tiled_zstd.tif                     312969    3.40x
```
