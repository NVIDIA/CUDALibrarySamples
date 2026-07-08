# nvTIFF Image-Info / Multi-Image Sample

Walks every image (IFD) of a TIFF or BigTIFF file, printing metadata,
strip/tile geometry, TIFF tags and SubIFDs.

## Key APIs

| Step | APIs |
|---|---|
| Open | `nvtiffStreamOpenFromFile` does header validation only, IFDs parse on first access |
| Navigate | `nvtiffStreamGetHeader`, `nvtiffStreamGetNextIFDOffset`, `NVTIFF_NO_IMAGE`, SubIFD tag 330 |
| Inspect | `nvtiffStreamGetImageInfo`, `nvtiffStreamGetImageGeometry`, `nvtiffStreamGetNumberOfTags`, `nvtiffStreamGetTagInfo`, `nvtiffStreamGetTagValue` |

Because parsing is lazy, opening a multi-gigabyte whole-slide or pyramidal
TIFF and listing its images can be quite fast. IFD headers are only parsed when an
image's metadata is actually requested. This sample explores the entire IFD tree, but an 
application dealing with extremely large TIFF images can make use of this behavior for best performance.

Large example files can be found at: https://openslide.cs.cmu.edu/download/openslide-testdata/
Multi-page and tiled inputs can also be generated with the nvTIFF-Encode-Options sample.

## Build and run

```
cmake -B build -DNVTIFF_PATH=/path/to/nvtiff/cuda13
cmake --build build
export LD_LIBRARY_PATH=/path/to/nvtiff/cuda13/lib:$LD_LIBRARY_PATH
./build/nvtiff_image_info multipage.tif
```

Expected output shape:

```
multipage.tif: TIFF, 2 image(s)

image 0 (IFD offset 8):
  725 x 489, 3 channel(s), 24 bits per pixel, RGB, LZW
  layout: 4 strip(s) of 725 x 128
  10 tag(s): 256 257 258 259 262 273 277 278 279 284

image 1 (IFD offset 158):
  ...
```
