# nvTIFF Encode-Options Sample

Decodes one image to 8-bit RGB and re-encodes it five ways using various encode options.

| Variant | Demonstrates |
|---|---|
| LZW, 128-row strips, 2 pages | `nvtiffEncodeParamsSetImageGeometry` (striped), multi-image encode via `nvtiffEncodeParamsSetInputs` |
| LZW, 256×256 tiles | `nvtiffEncodeParamsSetImageGeometry` (tiled; dimensions must be multiples of 16) |
| JPEG q85, 4:4:4, optimized Huffman | `nvtiffEncodeParamsSetJpegOptions` (needs nvJPEG at runtime) |
| Uncompressed BigTIFF | `nvtiffEncodeParamsSetTiffVariant` |
| LZW + GeoTIFF model tags (DOUBLE), via host buffer | `nvtiffEncodeParamsSetTag`, `nvtiffWriteTiffBuffer` |

The five outputs are ordinary TIFF/BigTIFF files, and you may use them as inputs to the
other nvTIFF samples (multi-page for `nvTIFF-Image-Info-Multi-Image`, tiled
for `nvTIFF-Batched-Region-Decode`, ...).

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
LZW, custom tags, via buffer         custom_tags_from_buffer.tif        328100    3.24x
```
