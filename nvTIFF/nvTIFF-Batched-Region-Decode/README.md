# nvTIFF Batched-Region-Decode Sample

Decodes a reproducible random set of patches from one image in a single `nvtiffDecode` call. 

## Key APIs

| Step | APIs |
|---|---|
| Describe regions | `nvtiffDecodeRegion_t` (IFD offset + ROI), `nvtiffDecodeParamsSetRegions` |
| Validate | `nvtiffDecodeCheckSupported` (full dry-run with output descriptors) |
| Decode | one `nvtiffDecode` call, one `nvtiffImage_t` per region |
| Fallback | `NVTIFF_STATUS_BATCH_INCOMPATIBLE` -> per-region decode loop |

A multi-region request either executes as one batch or is rejected whole
with `NVTIFF_STATUS_BATCH_INCOMPATIBLE`; the library never silently serializes
batches. That status means "individually decodable, but not together" (distinct
from `NVTIFF_STATUS_TIFF_NOT_SUPPORTED`) and the sample shows the
per-region fallback loop for it.

The same pattern extends to volumetric TIFF stacks: if each IFD represents one
Z-slice, a 3D ROI can be assembled as a batch of 2D regions, one per selected
IFD. Each output descriptor can point into the appropriate Z plane of a larger
volume buffer by using its own `plane_data` pointer and `plane_pitch_bytes`.

Tiled inputs show this workload best: generate one with the
nvTIFF-Encode-Options sample, or use real pyramidal slides from
https://openslide.cs.cmu.edu/download/openslide-testdata/.

## Build and run

```
cmake -B build -DNVTIFF_PATH=/path/to/nvtiff/cuda13
cmake --build build
export LD_LIBRARY_PATH=/path/to/nvtiff/cuda13/lib:$LD_LIBRARY_PATH
./build/nvtiff_batched_region_decode tiled.tif
./build/nvtiff_batched_region_decode tiled.tif --patch-width 512 --patch-height 512 --num-patches 4
```

Expected output shape:

```
tiled.tif: 725 x 489, decoding 8 random patch(es) of up to 256 x 256 in one batch
decoded 8 patches in one batch: 9.8 ms (814 patches/s)
  wrote 256 x 256 preview tiled_patch0.ppm
  wrote 256 x 256 preview tiled_patch1.ppm
  wrote 256 x 256 preview tiled_patch2.ppm
  ...
```

By default, the sample decodes eight 256 x 256 patches. It uses a fixed seed,
so the selected patches are reproducible across runs. Requested patch
dimensions are clipped to the source image size and to 4096 pixels per side.
