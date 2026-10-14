# NPP Workspace in a Per-Frame Loop

## Description

Shows what the NPP workspace (CUDA Toolkit 13.4) changes in a video-style loop, where the same NPP call runs on every frame and frame sizes vary.
The call is `nppiFilterCannyBorder_8u_C1R_Ctx`; four ways of providing its scratch buffer are timed on a stream of 640x480, 1280x720 and 1920x1080 frames:

| Mode | What the loop does per frame |
|------|------------------------------|
| per-frame alloc | `GetBufferSize`, `cudaMalloc`, call, `cudaFree` |
| preallocated | call with one buffer sized once for the largest frame |
| workspace (NULL) | `nppiWorkspaceCreate` once; call with a `NULL` buffer, NPP grows and reuses its own buffer |
| workspace + external | as above, but backed by application-owned memory via `nppiWorkspaceSetExternal` |

The input frames stay on the device, so the numbers isolate NPP call and buffer handling cost. The sample also checks that all four modes
produce identical edge maps and exits non-zero otherwise.

## Key Concepts
- Removing per-frame `cudaMalloc`/`cudaFree` (the `cudaFree` also synchronizes the device and stalls pipelining)
- Workspace as a high-water-mark buffer: no need to know the largest frame size up front
- `nppiWorkspaceSetExternal` for frameworks that must own every allocation
- Zero-initializing `NppStreamContext`

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
```
$ ./workspaceLoop [-n frames]
```
`-n` is the number of frames per timed run (default 2000). Each mode is warmed up over every frame size, then the median of five timed runs is reported.

Sample output (RTX A6000, CUDA 13.4):
```
mode                       us per frame   vs alloc
per-frame alloc                   318.4      1.00x
preallocated                       38.4      8.28x
workspace (NULL)                   38.2      8.34x
workspace + external               38.2      8.33x

Workspace capacity after the loop: 26.7 MB (internal), 17.8 MB (external)
```

## Reading the results
- The gain over per-frame allocation is large because Canny on these sizes runs for tens of microseconds; the allocate/free pair and the synchronization
  in `cudaFree` dominate. For long-running calls the relative gain is small.
- The workspace matches a hand-managed preallocated buffer. Its advantage is that it needs no sizing code and handles sizes you did not plan for.
- The internal workspace ended up larger (26.7 MB) than the largest `GetBufferSize` result (17.8 MB), so it is not an exact fit. Use
  `nppiWorkspaceGetCapacity` to see what it holds, or `SetExternal` when the memory budget must be exact.
- Not every function supports a `NULL` buffer with the 13.4 library (for example `nppiCompressMarkerLabelsUF`, `nppiDistanceTransformPBA`,
  `nppiCrossCorrValid_NormLevel` still need an explicit buffer); see the top-level README.

## License
Apache-2.0. See the SPDX headers in the source.
