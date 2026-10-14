# NPP vs OpenCV Benchmark

## Description

Times the same four operations with NPP on the GPU and with OpenCV on the CPU, on a 4096x2048 gray image (the 2048x1024 circuit board, tiled):
Gaussian blur 5x5, Canny edge detection, bilinear resize to half size, and RGB to gray.

Three kinds of number are reported, because the right comparison depends on where your data lives:

| Column | What it measures |
|--------|------------------|
| NPP kernel | GPU time of the NPP call alone, with the data already on the device (the case in a GPU pipeline) |
| NPP e2e | host-to-device copy + NPP call + device-to-host copy, using pinned host memory (the case of offloading a single operation) |
| CV 1 thr / CV N thr | OpenCV on the CPU with `cv::setNumThreads(1)` and with all hardware threads |

The outputs are compared (mean absolute difference) as a sanity check only. They are not bit-identical: NPP's Gaussian 5x5 and OpenCV's
`GaussianBlur` with sigma 0 use different kernels, and the Canny implementations differ in detail, so their edge counts differ (the sample prints both).
Resize and RGB-to-gray match exactly on this image.

## Key Concepts
- Comparing GPU and CPU fairly: kernel-only vs end-to-end including PCIe transfers
- Single-thread vs multi-thread CPU baselines
- NPP workspace and `_Ctx` calls used for the timed loop

## CUDA APIs involved
[NPP](https://docs.nvidia.com/cuda/npp/index.html)

## Build
Requires OpenCV with `core` and `imgproc`.
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The CPU numbers are only meaningful if OpenCV was built with a threading backend (pthreads, TBB or OpenMP); the sample warns if it reports a
single thread. If CMake finds a single-threaded OpenCV, point it at another one, for example `cmake .. -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4`.

## Usage
```
$ ./opencvBenchmark [-i CircuitBoard_2048x1024_8u.raw] [-w width] [-h height] [-n iterations]
```
Each timing is the median of five runs of `-n` iterations (default 20) after a warm-up. `-n` must be at least 1 and `-w`/`-h` must be between 16 and 16384.

Sample output (RTX A6000, CUDA 13.4, OpenCV 4.5.4 with 20 CPU threads):
```
operation               NPP kernel     NPP e2e    CV 1 thr    CV N thr |    kernel       e2e       e2e | mean abs
                              (ms)        (ms)        (ms)        (ms) |  vs 1 thr  vs 1 thr  vs N thr | diff
Gaussian 5x5                 0.097       2.131       4.082       0.671 |     42.3x      1.9x      0.3x | 1.70
Canny 3x3 L2                 0.262       2.294      22.111       3.566 |     84.5x      9.6x      1.6x | 6.75
Resize 1/2 bilinear          0.019       1.312       0.545       0.147 |     29.4x      0.4x      0.1x | 0.00
RGB to gray                  0.051       4.135       3.581       0.562 |     70.9x      0.9x      0.1x | 0.00
```
(speedups are CPU time divided by GPU time, so a value below 1x means the CPU is faster)

## Reading the results
- With the data already on the GPU, NPP is 30 to 85 times faster than single-threaded OpenCV for these operations.
- For a single operation on host data, the PCIe transfers dominate: the end-to-end time is about 1.3 to 4 ms regardless of the operation. Cheap
  operations (resize, RGB to gray, Gaussian) are then slower than a multi-threaded CPU. Only the compute-heavy Canny stays ahead (about 1.6x
  over 20 threads).
- The practical conclusion is that NPP pays off when several operations run back to back on the device, or when the data is already there
  (decoded with nvJPEG or nvJPEG2000, produced by a CUDA kernel, or consumed by a neural network). It does not pay off for one cheap operation on a host image.
- Results depend on the GPU, the CPU, the interconnect (PCIe generation) and image size. Run it on your own hardware before quoting numbers.

## License
Apache-2.0. See the SPDX headers in the source.
