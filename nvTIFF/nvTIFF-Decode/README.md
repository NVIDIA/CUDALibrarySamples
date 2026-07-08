# nvTIFF Decode Sample

Decodes the first image of a TIFF file on the GPU, prints basic image
metadata, and writes a PNM preview.

## Key APIs

| Step | APIs |
|---|---|
| Open | `nvtiffStreamOpenFromFile` |
| Parse | `nvtiffStreamGetImageInfo` |
| Decode | `nvtiffDecoderCreateSimple`, `nvtiffDecodeParamsCreate`, `nvtiffDecodeParamsSetOutputFormat`, `nvtiffDecode` |

## Build and run

```
cmake -B build -DNVTIFF_PATH=/path/to/nvtiff/cuda13
cmake --build build
export LD_LIBRARY_PATH=/path/to/nvtiff/cuda13/lib:$LD_LIBRARY_PATH
./build/nvtiff_decode ../images/bali_notiles.tif
```

Expected output shape:

```
../images/bali_notiles.tif: 725 x 489, 3 channel(s), 24 bits per pixel, RGB, LZW
decoded first image
wrote preview bali_notiles_decoded.ppm
```
