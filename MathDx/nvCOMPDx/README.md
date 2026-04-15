# nvCOMPDx Library - API Examples

All examples are shipped within [nvCOMPDx package](https://developer.nvidia.com/nvcompdx-downloads).

## Description

This folder demonstrates nvCOMPDx APIs usage.

* [nvCOMPDx download page](https://developer.nvidia.com/nvcompdx-downloads)
* [nvCOMPDx API documentation](https://docs.nvidia.com/cuda/nvcompdx/index.html)

## Requirements

* [See nvCOMPDx requirements](https://docs.nvidia.com/cuda/nvcompdx/requirements.html)
* CMake 3.26 or newer
* CUDA Toolkit 13.0 or newer
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Turing (SM75) or newer architecture
* External library: `lz4` (required for LZ4 CPU examples and NVRTC example — see Build section)

## Build

* You may specify `NVCOMPDX_CUDA_ARCHITECTURES` to limit CUDA architectures used for compilation (see [CMake:CUDA_ARCHITECTURES](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html#prop_tgt:CUDA_ARCHITECTURES))
* `mathdx_ROOT` - path to MathDx package (XX.Y - version of the package)

The LZ4 CPU and NVRTC examples require the `lz4` library. Install it before building:

```sh
# Ubuntu:
sudo apt-get install liblz4-dev liblz4-1

# Red Hat:
dnf install lz4-devel lz4-libs
```

```
mkdir build && cd build
cmake -DNVCOMPDX_CUDA_ARCHITECTURES=75-real -Dmathdx_ROOT=/opt/nvidia/mathdx/XX.Y ..
make
# Run
ctest
```

## Examples

For the detailed descriptions of the examples please visit Examples section of the [nvCOMPDx documentation](https://docs.nvidia.com/cuda/nvcompdx/index.html).

* [LZ4 GPU compression introduction](01_introduction/lz4_gpu_compression_introduction.cu)

    The introductory sample demonstrates a warp-level GPU LZ4 compression via nvCOMPDx. Note that the example presented here is deliberately simple to demonstrate the basic usage of nvCOMPDx and is **not** optimized for performance.

    ```
    lz4_gpu_compression_introduction -f <input file> -o <output file>
    ```

* [LZ4 GPU compression and decompression](02_lz4_gpu/lz4_gpu_compression_decompression.cu)

    The sample demonstrates warp-level GPU LZ4 compression and decompression into global memory via nvCOMPDx.

    ```
    lz4_gpu_compression_decompression -t {uint8|uint16|uint32} -f <input file(s)>
    ```

* [LZ4 CPU compression](03_lz4_gpu_and_cpu/lz4_cpu_compression_gpu_decompression.cu)

    The sample demonstrates CPU LZ4 compression via `lz4::LZ4_compress_HC`, and subseqent warp-level GPU decompression into global memory via nvCOMPDx.

    ```
    lz4_cpu_compression_gpu_decompression -f <input file(s)>
    ```

* [LZ4 CPU decompression](03_lz4_gpu_and_cpu/lz4_gpu_compression_cpu_decompression.cu)

    The sample demonstrates warp-level GPU LZ4 compression into global memory via nvCOMPDx, and subsequent CPU decompression via `lz4::LZ4_decompress_safe`.

    ```
    lz4_gpu_compression_cpu_decompression -f <input file(s)>
    ```

* [ANS GPU compression and decompression](04_ans_gpu/ans_gpu_compression_decompression.cu)

    The sample demonstrates block-level GPU ANS compression and decompression into global memory via nvCOMPDx.

    ```
    ans_gpu_compression_decompression -t {uint8|float16} -f <input file(s)>
    ```

* [ANS decompress and reduce](04_ans_gpu/ans_gpu_decompression_reduction.cu)

    The sample demonstrates GPU ANS decompression followed by a reduction operation fused within the same GPU kernel.

    ```
    ans_gpu_decompression_and_reduction
    ```

* [LZ4 NVRTC+nvJitLink](05_lz4_cpu_and_nvrtc/lz4_cpu_compression_nvrtc_decompression.cu)

    The sample demonstrates CPU LZ4 compression via `lz4::LZ4_compress_HC`, and subseqent warp-level GPU decompression into global memory via a runtime-compiled and linked kernel by NVRTC and nvJitLink. The non-runtime-compiled variant can be found [here](03_lz4_gpu_and_cpu/lz4_cpu_compression_gpu_decompression.cu).

    ```
    lz4_cpu_compression_nvrtc_decompression -f <input file(s)>
    ```
