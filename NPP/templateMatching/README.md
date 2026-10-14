# Template Matching with Normalized Cross-Correlation using NPP

## Description

Finds a small template in a large image with NPP's normalized cross-correlation and locates the best match on the GPU.
A template is cut out of a circuit-board image and deliberately degraded (lower contrast, brightness offset, noise), then searched
for again in the full image.

```
nppiCrossCorrValid_NormLevel_8u32f_C1R_Ctx   score for every template position
nppiMaxIndx_32f_C1R_Ctx                      position of the best score, on the GPU
```

The `NormLevel` score is the zero-mean normalized correlation coefficient (Pearson) in [-1, 1]. Both patches have their mean and energy
removed, so the score is unchanged by a gain or offset between template and image. That makes it robust to illumination changes
in inspection and alignment setups. The sample's default template is rendered at 60% contrast plus an offset of 30, yet still scores about 0.9995 at the correct position.

The sample also:
- compares the GPU scores against a CPU computation at several positions,
- reports how clearly the best match stands out from the best score elsewhere in the image (a margin near 0 means the template occurs more than once, so the match is ambiguous; the sample's `Located` line then says so),
- writes `match_result.pgm`, the image with a box around the match,
- times the GPU search.

## Key Concepts
- Normalized cross-correlation template matching (`nppiCrossCorrValid_NormLevel`)
- Finding the maximum and its position on the device (`nppiMaxIndx`)
- Explicit scratch buffers from `nppiValidNormLevelGetBufferHostSize_*` and `nppiMaxIndxGetBufferHostSize_*`

## CUDA APIs involved
[NPP](https://docs.nvidia.com/cuda/npp/index.html)

## Build
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Usage
Run from the `build` directory so the default `../images/` input is found.
```
$ ./templateMatching [-x tpl-x] [-y tpl-y] [-tw tpl-width] [-th tpl-height] [-g gain] [-o offset] [-noise amplitude]
                     [-n timing-iterations] [-i image.raw]
```
The input must be a 2048x1024 8-bit gray RAW image (the included `CircuitBoard_2048x1024_8u.raw` is the same image used by the
[findContour](../findContour/) sample). Pick the template with `-x/-y/-tw/-th`. This image contains repeated structures, so some
locations match in several places; the sample reports this instead of failing silently.
Options are validated: the template must lie inside the image, `-noise` must be 0 to 255, and `-n 0` skips the timing.

Sample output (RTX A6000, CUDA 13.4):
```
Image 2048x1024, template 96x64 cut from (1300,850), degraded with gain 0.60, offset 30.0, noise +/-3
Best match at (1300,850), NCC score 0.99951   [expected (1300,850)]
Best score outside the peak neighbourhood: 0.82065  (margin 0.17886)
Max |GPU - CPU| NCC over 6 checked positions: 3.45e-05  [PASS]
Located the template within 1 pixel: YES
GPU search over 1953 x 961 positions: 17.331 ms per match (average of 50 runs)
```

## Performance notes
`nppiCrossCorrValid_NormLevel` correlates directly, so the cost grows with the template area. On the same GPU and image the search takes about
3 ms for a 32x32 template, 17 ms for 96x64 and 110 ms for 256x192. The scratch buffers are small and allocated once here, so
allocation is not a significant part of the time.

## License
Apache-2.0. See the SPDX headers in the source.
