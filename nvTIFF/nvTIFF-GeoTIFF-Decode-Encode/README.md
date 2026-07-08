# nvTIFF GeoTIFF Decode-Encode Sample

This sample demonstrates a GeoTIFF roundtrip; read the georeferencing (geo keys,
ModelPixelScale / ModelTiePoint / ModelTransformation TIFF tags, and RPC
coefficient tag 50844), decode the raster on the GPU, then encode a new GeoTIFF
with the georeferencing preserved and verify it.

## Key APIs

| Step | APIs |
|---|---|
| Read geo keys | `nvtiffStreamGetNumberOfGeoKeys`, `nvtiffStreamGetGeoKeyInfo`, `nvtiffStreamGetGeoKeySHORT` / `DOUBLE` / `ASCII` |
| Read geo tags | `nvtiffStreamGetTagInfo`, `nvtiffStreamGetTagValue`, `NVTIFF_TAG_MODEL_PIXEL_SCALE` / `MODEL_TIE_POINT` / `MODEL_TRANSFORMATION`, RPC tag 50844 |
| Write geo keys | `nvtiffEncodeParamsSetGeoKeySHORT` / `DOUBLE` / `ASCII`, `nvtiffEncodeParamsSetGeoKey` |
| Write geo tags | `nvtiffEncodeParamsSetTag` |

## Test data

Use a GeoTIFF, or create one from any image with GDAL, e.g.:

```
gdal_translate -of GTiff -a_srs EPSG:4326 -a_ullr 115.0 -8.0 115.7 -8.5 input.tif geo.tif
```

The sample's own output is a GeoTIFF, so it can be fed back to itself.

## Build and run

```
cmake -B build -DNVTIFF_PATH=/path/to/nvtiff/cuda13
cmake --build build
export LD_LIBRARY_PATH=/path/to/nvtiff/cuda13/lib:$LD_LIBRARY_PATH
./build/nvtiff_geotiff_decode_encode geo.tif geo_roundtrip.tif
```

Expected output shape:

```
geo.tif: 4 geo key(s)
  key 1024 (SHORT): 2
  key 1025 (SHORT): 1
  key 1026 (ASCII): WGS 84 sample
  key 2048 (SHORT): 4326
  ModelPixelScale: 0.001 0.001 0
  ModelTiePoint: 0 0 0 115 -8 0
raster: 64 x 64, 3 channel(s), RGB, uncompressed
wrote geo_roundtrip.tif with 4 geo key(s) preserved
roundtrip check: output has 4 geo key(s)
```
