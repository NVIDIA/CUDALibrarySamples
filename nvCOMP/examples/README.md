# nvCOMP Examples

## Description

This folder contains examples demonstrating the usage of the nvCOMP C++ and Python APIs.

## Examples

* [LZ4 CPU compression](lz4_cpu_compression.cu)

    The sample demonstrates CPU compression via lz4::LZ4_compress_HC, and subseqent GPU decompression via nvCOMP.

    ```
    lz4_cpu_compression -f <input file(s)>
    ```

* [LZ4 CPU decompression](lz4_cpu_decompression.cu)

    The sample demonstrates GPU compression via nvCOMP, and subsequent CPU decompression via lz4.

    ```
    lz4_cpu_decompression -f <input file(s)>
    ```

* [nvCOMP with GPUDirect Storage (GDS)](nvcomp_gds.cu)

    The sample demonstrates the usage of GPUDirect Storage with nvCOMP.

    ```
    nvcomp_gds <compressed sample output file>
    ```

* [Deflate CPU compression](deflate_cpu_compression.cu)

    The sample demonstrates CPU compression via libdeflate/zlib::compress2/zlib::deflate, and subsequent GPU decompression via nvCOMP.

    ```
    deflate_cpu_compression -a {0|1|2} -f <input file(s)>
    ```

* [Deflate CPU decompression](deflate_cpu_decompression.cu)

    The sample demonstrates GPU compression via nvCOMP, and subsequent CPU decompression via libdeflate/zlib::inflate.

    ```
    deflate_cpu_decompression -a {0|1} -f <input file(s)>
    ```

* [GDeflate CPU compression](gdeflate_cpu_compression.cu)

    The sample demonstrates CPU compression via gdeflate, and subsequent GPU decompression via nvCOMP.

    ```
    gdeflate_cpu_compression -f <input file(s)>
    ```

* [GZIP GPU decompression](gzpip_gpu_decompression.cu)

    The sample demonstrates CPU compression via zlib::deflate, and subsequent GPU decompression via nvCOMP.

    ```
    gzip_gpu_decompression -f <input file(s)>
    ```

* [High-level quickstart example](high_level_quickstart_example.cpp)

    The sample demonstrates the usage of the high-level nvCOMP API.

    ```
    high_level_quickstart_example
    ```

* [Low-level quickstart example](low_level_quickstart_example.cpp)

    The sample demonstrates the usage of the low-level nvCOMP API.

    ```
    low_level_quickstart_example
    ```

* [Python API usage example](python/nvcomp_basic.ipynb)

    The sample demonstrates the usage of the nvCOMP Python API.

## Building (x86-64, or aarch64)

The samples require the following external libraries to be installed prior to compilation: `libdeflate`, `zlib`, and `lz4`.

### Linux

The external libraries can be installed via a package manager (both on ARM and on x86):

```sh
# LZ4
sudo apt-get install liblz4-dev
sudo apt-get install liblz4-1
# ZLib
sudo apt-get install zlib1g-dev
sudo apt-get install zlib1g
# Libdeflate
sudo apt-get install libdeflate-dev
sudo apt-get install libdeflate0
```

Alternatively, they can also be compiled from source.

Afterwards, the example compilation via CMake is relatively simple:

```sh
cd <nvCOMP example folder>
mkdir build
cd build

cmake .. -DCMAKE_PREFIX_PATH=<nvCOMP sysroot path> \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_GDS_EXAMPLE=ON

cmake --build .
```

### Windows

To the best of our knowledge, one needs to compile the dependencies from source before compiling the nvCOMP examples. This guide will help in preparing the necessary external libraries and header files using the [MSVC compiler](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#release-dates-and-build-numbers).

#### Preparing a sysroot folder

```sh
mkdir sysroot && cd sysroot
mkdir include
mkdir lib
mkdir bin
```

#### ZLib

```sh
# ZLib v1.3.1 (released on January 22, 2024)
# Website with latest source: https://zlib.net/
# Note: version available in Ubuntu's apt as of now:
#       - zlib1g-dev/jammy-updates,jammy-security,now 1:1.2.11.dfsg-2ubuntu9.2 amd64
#       - zlib1g/jammy-updates,jammy-security,now 1:1.2.11.dfsg-2ubuntu9.2 amd64
#
curl -LO https://zlib.net/zlib131.zip
tar -xf zlib131.zip
cd zlib-1.3.1
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<path to planned sysroot folder>
cmake --build . --config Release --target install --parallel 14
```

#### LZ4

```sh
# LZ4 v1.9.4 (released on August 16, 2022)
# Website with latest source: https://github.com/lz4/lz4
# Note: version available in Ubuntu's apt as of now:
#       - liblz4-1/jammy,now 1.9.3-2build2 amd64
#       - liblz4-dev/jammy,now 1.9.3-2build2 amd64
#
curl -LO https://github.com/lz4/lz4/archive/refs/tags/v1.9.4.zip
tar -xf v1.9.4.zip
cd lz4-1.9.4/build/cmake
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<path to planned sysroot folder>
cmake --build . --config Release --target install --parallel 14
```

#### Libdeflate

```sh
# Libdeflate v1.10
# Website with latest source: https://github.com/ebiggers/libdeflate
# Note: version available in Ubuntu's apt as of now:
#       - libdeflate-dev/jammy,now 1.10-2 amd64
#       - libdeflate0/jammy,now 1.10-2 amd64
#
curl -LO https://github.com/ebiggers/libdeflate/releases/download/v1.10/libdeflate-1.10-windows-x86_64-bin.zip
mkdir libdeflate-1.10
tar -xf libdeflate-1.10-windows-x86_64-bin.zip -C libdeflate-1.10
cd libdeflate-1.10
mv libdeflate.h <sysroot path>/include/.
mv libdeflate.dll <sysroot path>/bin/.
mv libdeflate.def <sysroot path>/lib/deflate.def
mv libdeflate.lib <sysroot path>/lib/deflate.lib
mv libdeflatestatic.lib <sysroot path>/lib/deflatestatic.lib
```

The resulting sysroot tree should look as follows:

```sh
sysroot
├───bin
│       libdeflate.dll
│       lz4.dll
│       lz4.exe
│       lz4c.exe
│       zlib.dll
│
├───include
│       libdeflate.h
│       lz4.h
│       lz4frame.h
│       lz4hc.h
│       zconf.h
│       zlib.h
│
├───lib
│   │   deflate.def
│   │   deflate.lib
│   │   deflatestatic.lib
│   │   lz4.lib
│   │   zlib.lib
│   │   zlibstatic.lib
│   │
│   ├───cmake
│   │   └───lz4
│   │           lz4Config.cmake
│   │           lz4ConfigVersion.cmake
│   │           lz4Targets-release.cmake
│   │           lz4Targets.cmake
│   │
│   └───pkgconfig
│           liblz4.pc
│
└───share
    ├───man
    │   ├───man1
    │   │       lz4.1
    │   │
    │   └───man3
    │           zlib.3
    │
    └───pkgconfig
            zlib.pc
```

Afterwards, the example compilation via CMake is relatively simple:

```sh
cd <nvCOMP example folder>
mkdir build
cd build

# Run CMake configuration
cmake .. -DCMAKE_PREFIX_PATH=<nvCOMP sysroot path> \
         -DCMAKE_BUILD_TYPE=Release

# Run the actual build
cmake --build . --parallel 14
```