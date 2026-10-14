# Hough Line Detection using NPP

## Description

Detects straight lines with the Hough transform on the GPU: Canny edge detection followed by `nppiFilterHoughLine`, which returns the
detected lines as `(rho, theta)` pairs in device memory.

```
gray image -> nppiFilterCannyBorder_8u_C1R_Ctx -> binary edges -> nppiFilterHoughLine_8u32f_C1R_Ctx -> lines (rho, theta)
```

By default the sample draws a synthetic image with five lines of known position and orientation plus light noise, runs the
pipeline, and checks that every true line was found. With `-i` it runs on a real 8-bit gray RAW image (for example the
circuit board used by [findContour](../findContour/)) and draws the detected lines without verifying them.

NPP does not provide a Hough circle transform; this sample covers lines only.

## Key Concepts
- Hough line transform on a binary edge image (`nppiFilterHoughLine`, `nppiFilterHoughLineRegion` limits lines to a rho/theta range)
- Edge detection feeding the Hough step (`nppiFilterCannyBorder`)
- Interpreting NPP's polar line output

## CUDA APIs involved
[NPP](https://docs.nvidia.com/cuda/npp/index.html)

## Reading the output lines
Each line is an `NppPointPolar {rho, theta}`. `theta` is in degrees. `rho` is an accumulator index; the signed distance from the image centre
is `rho - accumulatorHeight/2`, where `accumulatorHeight = ceil(sqrt(2) * max(width, height))` for the 1-pixel `nDelta.rho` used here. The line is
`x*cos(theta) + y*sin(theta) = rho_signed` with `x, y` measured from the centre. NPP does not guarantee the order of the returned lines (the sample
sorts them) and does not return vote counts, so strength ordering is not available.

The sample fixes `nDelta = {1.0, 1.0}` (1 pixel, 1 degree), the recommended setting. In testing with CUDA 13.4, other values (for example 0.5 or 2.0)
returned results that did not match the known lines, so the conversion above is only verified for 1.0/1.0.

A 3-pixel-wide bright line is outlined by Canny on both sides, so the synthetic test yields two detections per true line, about 3 pixels apart.
Real images likewise produce many near-duplicate lines for thick structures; merging nearby (rho, theta) pairs is left to the application.

## Build
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Usage
```
$ ./houghLines [-t vote-threshold] [-m max-lines] [-n timing-iterations] [-cl canny-low] [-ch canny-high]
               [-i image.raw -w width -h height]
```
`-t` is the minimum number of edge pixels on a line (default 300), `-m` the capacity of the output list (default 500; the sample warns if it is reached).
The result is written to `hough_result.ppm` (input in gray, detected lines in red). The process exits with a non-zero status if the
synthetic check fails, for example with `-t 2000`. Options are range-checked (for example `-w`/`-h` 16 to 16384, `-m` at least 1,
`-cl` not above `-ch`), and `-n 0` skips the timing.

Sample output (RTX A6000, CUDA 13.4):
```
Synthetic image 1024x768, Canny 60/120, Hough threshold 300
Detected 10 line(s)
  theta   30.0 deg   rho   -152.0 px
  theta   30.0 deg   rho   -149.0 px
  ...
Check against the five true lines (tolerance 3 px, 1 degree):
  true theta  30.0 rho -150.0 : 2 detection(s)
  ...
Detections not matching any true line: 0
All five lines found: [PASS]
GPU time per frame: Canny 0.030 ms, Hough 0.320 ms (average of 50 runs)
```
On the 2048x1024 circuit board (`-t 250 -cl 80 -ch 160`) it detects 139 lines, with Canny 0.07 ms and Hough 0.8 ms per frame.
The timings are GPU kernel time for the two NPP calls only.

## License
Apache-2.0. See the SPDX headers in the source.
