# nvCOMPDx Examples

## Description

This folder contains examples demonstrating the usage of the nvCOMPDx C++ API.

## Examples

* [LZ4 GPU compression introduction](lz4_gpu_compression_introduction.cu)

    The introductory sample demonstrates a warp-level GPU LZ4 compression via nvCOMPDx. Note that the example presented here is deliberately simple to demonstrate the basic usage of nvCOMPDx and is **not** optimized for performance.

    ```
    lz4_gpu_compression_introduction -f <input file> -o <output file>
    ```

* [LZ4 GPU compression and decompression](lz4_gpu_compression_decompression.cu)

    The sample demonstrates warp-level GPU LZ4 compression and decompression into global memory via nvCOMPDx.

    ```
    lz4_gpu_compression_decompression -t {uint8|uint16|uint32} -f <input file(s)>
    ```

* [LZ4 CPU compression](lz4_cpu_compression_gpu_decompression.cu)

    The sample demonstrates CPU LZ4 compression via `lz4::LZ4_compress_HC`, and subseqent warp-level GPU decompression into global memory via nvCOMPDx.

    ```
    lz4_cpu_compression_gpu_decompression -f <input file(s)>
    ```

* [LZ4 CPU decompression](lz4_gpu_compression_cpu_decompression.cu)

    The sample demonstrates warp-level GPU LZ4 compression into global memory via nvCOMPDx, and subsequent CPU decompression via `lz4::LZ4_decompress_safe`.

    ```
    lz4_gpu_compression_cpu_decompression -f <input file(s)>
    ```

* [ANS GPU compression and decompression](ans_gpu_compression_decompression.cu)

    The sample demonstrates block-level GPU ANS compression and decompression into global memory via nvCOMPDx.

    ```
    ans_gpu_compression_decompression -t {uint8|float16} -f <input file(s)>
    ```

* [ANS decompress and reduce](ans_gpu_decompression_and_reduction.cu)

    The sample demonstrates GPU ANS decompression followed by a reduction operation fused within the same GPU kernel.

    ```
    ans_gpu_decompression_and_reduction
    ```

* [LZ4 NVRTC+nvJitLink](lz4_cpu_compression_nvrtc_decompression.cu)

    The sample demonstrates CPU LZ4 compression via `lz4::LZ4_compress_HC`, and subseqent warp-level GPU decompression into global memory via a runtime-compiled and linked kernel by NVRTC and nvJitLink. The non-runtime-compiled variant can be found [here](lz4_cpu_compression_nvrtc_decompression.cu).

    ```
    lz4_cpu_compression_nvrtc_decompression -f <input file(s)>
    ```

## Building (x86-64, or aarch64)

The samples require the following external libraries to be installed prior to compilation: `lz4`.

### Linux

The external libraries can be installed via a package manager (both on ARM and on x86):

```sh
# Ubuntu:
# LZ4
sudo apt-get install liblz4-dev
sudo apt-get install liblz4-1

# Red Hat:
# LZ4
dnf install lz4-devel
dnf install lz4-libs
```

Alternatively, they can also be compiled from source.

Afterwards, the example compilation via CMake is relatively simple:

```sh
cd <nvCOMPDx example folder>
mkdir build
cd build

cmake .. -DCMAKE_PREFIX_PATH=<nvCOMPDx sysroot path> \
         -DCMAKE_BUILD_TYPE=Release

cmake --build .
```
