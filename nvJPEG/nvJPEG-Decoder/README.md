# JPEG decoding with nvJPEG

This sample decodes JPEG images using the public nvJPEG API. By default it uses
batched hardware decoding for supported inputs and the decoupled decoder for
other inputs. RGB/BGR output is written as BMP. Use `-backend hardware` to require
hardware decoding and fail on unsupported inputs instead of falling back.

## Build

Requirements: CMake 3.17+, a C++17 compiler, CUDA Toolkit 13.0+, and a compatible
NVIDIA driver. Linux x86_64/aarch64 and Windows x64 use the same source.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

On Windows the executable is normally `build/Release/nvjpegDecoder.exe`. Ensure
the CUDA runtime and nvJPEG shared libraries are on the library search path;
CUDA 13.0 on Windows normally uses the toolkit's `bin/x64` directory in `PATH`.

## Run

```sh
# Existing default: automatic backend selection and BMP output.
./build/nvjpegDecoder -i input.jpg -o decoded

# Require hardware; decode 100 images, looping over the input as needed.
./build/nvjpegDecoder -i input.jpg -backend hardware -t 100

# Hardware downscale by two and write planar YUV.
./build/nvjpegDecoder -i input.jpg -backend hardware -fmt yuv -scale 2 -o decoded

# NV12 requires 4:2:0 input; YUY2 requires 4:2:2 input.
./build/nvjpegDecoder -i input_420.jpg -backend hardware -fmt nv12 -o decoded
./build/nvjpegDecoder -i input_422.jpg -backend hardware -fmt yuy2 -o decoded

# Compare every decoded output against approved raw goldens.
./build/nvjpegDecoder -i input.jpg -backend hardware -fmt yuv -scale 2 -t 100 -reference goldens
```

`-i` accepts a file or a directory. Directories are searched recursively for
`.jpg`, `.jpeg`, and `.jpe` files, case-insensitively, in sorted order. Every
selected file must succeed. `-t` is the total image count, not the count per file;
by default each input is decoded once. `-b` sets the batch size (default 1),
and a final partial batch preserves the exact requested count. `-w` sets the
number of warmup batches excluded from performance reporting. `-device` selects
the CUDA device (default 0). Use `-h` for the command syntax.

`-backend hardware` uses the single-image API because the batched hardware path
does not support scaling. It queries support for each input/format/scale
combination before allocating output and synchronizes each decode before state
reuse. This mode is for functional validation and repeated workloads, not
maximum-throughput benchmarking. `-scale` accepts divisors `1`, `2`, `4`, and `8`;
values other than 1 require hardware mode. Output dimensions round up.

## Output and goldens

`-fmt` accepts `rgb` (default), `bgr`, `rgbi`, `bgri`, `yuv`, `y`, `unchanged`,
`nv12`, and `yuy2`. RGB/BGR formats use BMP output unless `-raw` is supplied.
Other formats always use raw output. `-reference` also selects raw output.

Raw files have no headers or row padding; planes are concatenated in nvJPEG
channel order. RGB/BGR planes are full resolution; interleaved RGB/BGR uses
three bytes per pixel. YUV/unchanged preserves native component sizes, and
grayscale/luma uses one plane. NV12 contains Y followed by UV, whose row size is
`2 * ceil(width/2)` and height `ceil(height/2)`. YUY2 uses YUYV pairs with row size
`4 * ceil(width/2)`; the unused second Y in an odd-width row's last pair is set
to zero on readback for reproducible goldens.

Raw output names append `.<format>.s<divisor>.raw` to the complete input name,
for example `input.jpg.yuv.s2.raw`. Directory inputs preserve subdirectories.
`-reference` reads files with the same names and compares every decoded result
byte-for-byte. Comparison occurs before output writing. Missing, unreadable,
or mismatched goldens fail the run. An `-o` run can capture candidate goldens;
approve their correctness independently and qualify them for the intended
backend/GPU/driver/library configuration.

The final log is `RESULT status=PASS code=0 decoded=N` on success. Failures report
`RESULT status=FAIL code=C decoded=N message="..."` on stderr. Capture both output
streams and use the exit code: 0 success, 1 execution/I/O failure, 2 invalid
arguments, 3 unsupported hardware/input/options, 4 golden mismatch. A library
invalid-parameter error can also report an unsupported combination as exit 1.
The reported count excludes warmups. Without goldens, PASS establishes successful
decoding and CUDA completion, not independent pixel correctness.

The public API does not expose private engine controls or arbitrary scaling
ratios. The sample does not demux MJPEG containers; testing extracted JPEG frames
does not reproduce container/streaming tests. Check actual workload inputs,
options, and approved goldens before claiming equivalent test coverage.

## Tests

Optional GPU integration tests require Python 3 with Pillow. They generate their
own images and check hardware formats/scaling, raw packing, golden failures,
invalid inputs, and default BMP output with mixed hardware/fallback batches.
Hardware tests require a GPU with JPEG decoding support and do not skip it.

```sh
cmake -S . -B build -DNVJPEG_DECODER_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```
