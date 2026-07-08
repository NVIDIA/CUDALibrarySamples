# nvTIFF Decode-Encode Sample

This sample demonstrates a minimal GPU TIFF roundtrip:
decode the first image of a TIFF file, write a PNM preview,
re-encode the pixels as an LZW-compressed TIFF.

## Key APIs

| Step | APIs |
|---|---|
| Parse | `nvtiffStreamOpenFromFile`, `nvtiffStreamGetImageInfo` |
| Decode | `nvtiffDecoderCreateSimple`, `nvtiffDecodeParamsSetOutputFormat`, `nvtiffDecode` |
| Encode | `nvtiffEncodeParamsSetImageInfo`, `nvtiffEncodeParamsSetInputs`, `nvtiffEncode`, `nvtiffEncodeFinalize`, `nvtiffGetBitstreamSize`, `nvtiffWriteTiffFile` |

Freshly created decode params already request the first full image, so no
region setup is needed for the common case. The decode output is a single
interleaved plane described by `nvtiffImage_t` (pointer + pitch). Palette
images are converted to RGB16 during decode, and JPEG-compressed YCbCr images
are converted to RGB8; everything else keeps its native layout
(`NVTIFF_OUTPUT_UNCHANGED_I`). Inputs whose decoded layout cannot be represented
by the nvTIFF encoder are rejected before decode.

For custom device/pinned memory allocators, create the decoder with
`nvtiffDecoderCreate` instead of `nvtiffDecoderCreateSimple`.

## Build and run

```
cmake -B build -DNVTIFF_PATH=/path/to/nvtiff/cuda13   # archive dir or install prefix
cmake --build build
export LD_LIBRARY_PATH=/path/to/nvtiff/cuda13/lib:$LD_LIBRARY_PATH
./build/nvtiff_decode_encode ../images/bali_notiles.tif out.tif
```

Expected output:

```
../images/bali_notiles.tif: 725 x 489, 3 channel(s), 24 bits per pixel, RGB, LZW
decoded first image
  wrote preview bali_notiles_decoded.ppm
re-encoded with LZW: 1063575 raw -> 326530 compressed bytes (3.26x)
wrote out.tif
```
