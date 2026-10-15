# nvCOMP Benchmarks

This folder contains benchmarks demonstrating the performance (and usage) of the nvCOMP C and C++ APIs. They test compressing and decompressing input data, reporting the compression throughput, decompression throughput, and compression ratio. You can run the benchmark executables on your own files, or test standard benchmark datasets, as described below.

The chunked (low-level API) benchmarks use [NVBench](https://github.com/NVIDIA/nvbench) as their benchmarking framework. NVBench is fetched automatically via [CPM](https://github.com/cpm-cmake/CPM.cmake) during the CMake configuration step -- no manual installation is required.

## Benchmarks

### NVBench-based benchmarks (chunked / low-level API)

The following benchmarks are built on NVBench. Each executable registers separate compression and decompression benchmarks. In addition to the nvCOMP-specific options listed below, all NVBench command-line flags are supported. Run any benchmark with `--help` to see the full list.

- [ANS benchmark](benchmark_ans_chunked.cu)

  The sample demonstrates the ANS compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_ans_chunked {-f|--input_file} <input_file>
                        [{-t|--type} {char|uchar|float16|float32|float8_e4m3}]
                        [{-s|--max-sub-chunk-count} <power-of-2 in [4, 64], or 0 for default>]
                        [{-n|--states-per-lane} {0|1|2}]
                        [{-S|--histogram-reduction-log2} {0..8}]
                        [{-k|--skip-validate} {0|1}]
  ```

- [Bitcomp benchmark](benchmark_bitcomp_chunked.cu)

  The sample demonstrates the Bitcomp compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_bitcomp_chunked {-f|--input_file} <input_file>
                            [{-t|--type} {char|uchar|short|ushort|int|uint|longlong|ulonglong|float16|float32|float64}]
                            [{--alg|--algorithm} {0|1}]
                            [{--mode} {lossless|lossy-signed|lossy-unsigned}]
                            [{--delta} <quantization error-bound, e.g., 0.5>]
  ```

  Lossless mode accepts integral and floating-point types. Lossy modes require a floating-point type and an explicit
  `--delta`.

- [Cascaded benchmark](benchmark_cascaded_chunked.cu)

  The sample demonstrates the Cascaded compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_cascaded_chunked {-f|--input_file} <input_file>
                             [{-t|--type} {char|uchar|short|ushort|int|uint|longlong|ulonglong}]
                             [{--cm|--mode} {asymmetric|symmetric}]
                             [{--cl|--level} <compression_level>]
                             [{--cf|--fine_grained_encodings} {none|delta|rle|for|...}]
  ```

  Asymmetric mode selects useful Delta and RLE stages independently per internal
  chunk and prioritizes compression ratio; it is the default compression mode.
  Symmetric mode balances compression and
  decompression throughput and supports only `int`, `uint`, `longlong`, and
  `ulonglong`.

  Fine-grained encodings are supplied as a comma-separated list. Asymmetric mode
  currently requires Frame Of Reference to be enabled. Frame Of Reference is
  optional in symmetric mode. Symmetric mode currently ignores the compression
  level and considers every encoding in the list. Low-level compression and
  specified-mode decompression options must use the same mode. Cascaded
  bitpacking is currently required as the terminal codec.

- [Deflate benchmark](benchmark_deflate_chunked.cu)

  The sample demonstrates the Deflate compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_deflate_chunked {-f|--input_file} <input_file> [{--alg|--algorithm} {0|1|2|3|4|5}]
  ```

- [GDeflate benchmark](benchmark_gdeflate_chunked.cu)

  The sample demonstrates the GDeflate compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_gdeflate_chunked {-f|--input_file} <input_file> [{--alg|--algorithm} {0|1|2|3|4|5}]
  ```

- [Gzip benchmark](benchmark_gzip_chunked.cu)

  The sample demonstrates the gzip compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_gzip_chunked {-f|--input_file} <input_file>
                         [{--alg|--algorithm} {0|1|2|3|4|5}]
                         [{--da|--decompress-algorithm} {0|1}]
  ```

- [LZ4 benchmark](benchmark_lz4_chunked.cu)

  The sample demonstrates the LZ4 compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_lz4_chunked {-f|--input_file} <input_file>
                        [{-t|--type} {bits|char|uchar|short|ushort|int|uint}]
                        [{--bs|--bitshuffle} {0|1|2}]
  ```

- [Snappy benchmark](benchmark_snappy_chunked.cu)

  The sample demonstrates the Snappy compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_snappy_chunked {-f|--input_file} <input_file>
  ```

- [Zstandard benchmark](benchmark_zstd_chunked.cu)

  The sample demonstrates the Zstandard compression and decompression usage and performance via nvCOMP.

  ```
  benchmark_zstd_chunked {-f|--input_file} <input_file>
  ```

- [Gzip streaming benchmark](benchmark_gzip_streaming.cpp)

  Benchmarks the gzip streaming API (`nvcompGzipStreamingCompress` /
  `nvcompGzipStreamingDecompress`), which compresses or decompresses a single
  contiguous byte stream rather than an array of independent chunks. The
  benchmark supports two I/O modes:

  - **memory** – the input is loaded into memory and streamed via `std::istringstream`
    / `std::ostringstream`. This minimises I/O overhead and measures pure
    compression/decompression throughput.
  - **disk** – the input is read directly from the filesystem via `std::ifstream`
    and the output is written to a temporary file. This mode measures end-to-end
    throughput including I/O, which is useful for evaluating performance with OS I/O overheads.

  On Windows, only the compression benchmark is available; decompression requires
  concurrent managed memory access which is not supported on Windows.

  ```
  benchmark_gzip_streaming  {-f|--input_file} <input_file>
                            [{-m|--mode} {memory|disk}]
                            [{--alg|--algorithm} {0|1|2|3|4|5}]
                            [{-x|--duplicate_data} <num_copies>]
  ```

### Other benchmarks

- [CRC32 benchmark](benchmark_crc32.cpp)

  The sample demonstrates the nvCOMP CRC32 usage and performance via nvCOMP.

  ```
  benchmark_crc32 [{-o|--output} <output_file>]
                  [{-c|--complete} # boolean flag, enabling a fine-grained sweep over the buffer range]
                  [{-s|--storage} <selected storage size in bytes, must be a power of 2>]
  ```

- [Bitcomp CPU benchmark](benchmark_bitcomp_cpu.cu)

  The sample demonstrates the Bitcomp CPU usage (Native API) and performance via nvCOMP.

  ```
  benchmark_bitcomp_cpu [{-f} <input_file>]
                        [{-t} {uchar|char|ushort|short|uint|int|ulonglong|longlong}]
                        [{--alg} {0|1}]
                        [{--lossy} # boolean flag, activating the lossy mode of bitcomp]
                        [{--fp} {16|32|64}]
                        [{--delta} <quantization accuracy, e.g., 0.5>]
  ```

- [GDeflate CPU benchmark](benchmark_gdeflate_cpu.cpp)

  The sample demonstrates the GDeflate CPU (Native API) usage and performance via nvCOMP.

  ```
  benchmark_gdeflate_cpu [{-f} <input_file>]
                         [{-l} <compression level between 0 and 12 (both inclusive), e.g., 6>]
                         [{-p} <chunk size in bytes, e.g., 65536>]
  ```

- [Python (HLIF) benchmark](benchmark_python.py)

  The sample demonstrates the high-level Python API usage and performance via nvCOMP.
  Requires `cupy`, `numpy`, and the `nvidia.nvcomp` package.

  Each algorithm is benchmarked with its default options. Algorithm-specific
  settings can be overridden via `--codec-opts`. For all supported codec options see the nvCOMP Python documentation.

  ```
  benchmark_python.py {-f|--input_file} <input_file>
                      {-a|--algorithm} {ANS|Bitcomp|Cascaded|Deflate|GDeflate|Gzip|LZ4|Snappy|Zstd}
                      [{-b|--bitstream_kind} {NVCOMP_NATIVE|RAW|WITH_UNCOMPRESSED_SIZE}]
                      [{-p|--chunk_size} <chunk size, default 64 KiB>]
                      [{-x|--duplicate_data} <num_copies>]
                      [{--seconds} <per-operation timing budget>]
                      [{--min-samples} <min samples per operation>]
                      [{--csv} <output_file>]
                      [{--data_type} <typestr, e.g. <i4>]
                      [{--device_id} <CUDA device id>]
                      [{--codec-opts} <key=val ...>  # forwarded to nvcomp.Codec]
  ```

  `--seconds` is the per-operation timing budget: compression and decompression are
  each run repeatedly, accumulating per-run GPU timings until they sum to this many
  seconds, and the reported GB/s is the average over those samples (default 2 s).
  `--min-samples` sets a floor on the sample count (default 10), collected even after
  the `--seconds` budget is spent, so slow operations that fit only a few runs in the
  budget still get a stable average.

Most of the benchmark executables also support:

```
{--db|--decompress_backend} {0|1|2}                  Decompression backend to use: best available (0), hardware (1), or
                                                     CUDA (2). Default is best available.
{-m|--multiple_of} <num_bytes>                       The number of bytes the input is padded to, such that its overall length
                                                     becomes a multiple of the given argument (in bytes). Only applicable to
                                                     data without page sizes.
{-x|--duplicate_data} <num_copies>                   The number of copies to make of the input data before compressing
{-p|--chunk_size} <num_bytes>                        Chunk size when splitting uncompressed data
{--oc|--output_compressed_file} <output_file>        Path for the resulting compressed data chunks
{-o|--output_decompressed_file} <output_file>        Path for the resulting decompressed data (chunks)
{--single|--single_output_buffer} {false|true}       Use a single buffer for decompressing data
{--compressed|--compressed_inputs} {false|true}      The input dataset is compressed
{-h|--help}                                          Show help text for the benchmark
```

For compressors that accept a data type option (`--type`), input data for which all of the input matches that type will usually compress better than arbitrary data. The sizes of the types are 1 byte for `char/uchar/bits`, 2 bytes for `short/ushort`, 4 bytes for `int/uint`, and 8 bytes for `longlong/ulonglong`. Input files whose sizes aren't multiples of the data type size are unsupported. However, one can apply a padding to the input data via `--multiple_of` to make sure that the overall input size becomes an integer multiple of the data type size.

## Round-trip example

You can perform a round-trip compression and decompression while saving the nvCOMP output via the following commands:

```sh
# Perform the compression
# It will save the compressed chunks separately
./benchmark_zstd_chunked -f <input data> \
                         --output_compressed_file out.dump

# Perform the decompression
# Note: If we want to recover the original file, it is important to feed the
#       executable with chunks in the correct order.
./benchmark_zstd_chunked -f $(ls -v out.dump*) \
                         --compressed_inputs true \
                         --single_output_buffer true \
                         --output_decompressed_file replicated.dump

# Perform a verification against the original input
diff <input data> replicated.dump
```

## Benchmark datasets

If you would like to use standard benchmark datasets, there are two described here, "TPC-H" and "Mortgage", both of which are in the form of text tables that will first need to have a column extracted and converted to binary data, using the `text_to_binary.py` script.

To obtain TPC-H data tables, randomly generating a simulated database table of purchases:

- Clone and compile https://github.com/electrum/tpch-dbgen
- Run `./dbgen -s <scale factor>` to generate data in the file `lineitem.tbl`. A larger scale factor will result in a larger generated table.

To obtain [Fannie Mae's Single-Family Loan Performance Data](http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html)

- Download any of the archives from [Data Dynamics](https://datadynamics.fanniemae.com/data-dynamics/#/reportMenu;category=HP) (a quick registration is required)
- Navigate to "Download Data", then to "Single-Family Loan Performance Data"
- Download any of the quarterly results, e.g. 2000/Q1 Records
- Unpack `2000Q1.csv` from the downloaded `2000Q1.zip`

`text_to_binary.py` is provided to read a text file (e.g. csv) containing a table of data and output a specified column of data into a binary file. Both the TPC-H and Mortgage datasets use the vertical pipe character `|` as a column separator (delimiter) and store one row per text line. Usage:

```
python benchmarks/text_to_binary.py <input_text_file> <column_number, 0-based indexing> {int|long|float|double|string} <output_binary_file> [<column_separator>]
```

For example, to extract column 10 (in zero-based indexing; or the 11th column in one-based indexing) from `lineitem.tbl`, where columns are separated by `|`, and write it to binary file `shipdate_column.bin` as a sequence of 8-byte integers, run:

```
python text_to_binary.py lineitem.tbl 10 long shipdate_column.bin '|'
```

The default delimiter, if not specified, is a comma character, and the `string` data type converts the text to UTF-16 and concatenates all of the text in the output file. `float` is single-precision floating-point (4 bytes), and `double` is double-precision floating-point (8 bytes).

## Building (x86-64, or aarch64)

The compilation of benchmarks does not require additional external libraries, apart from NVBench which is fetched during compilation:

```sh
cd <nvCOMP benchmark folder>
mkdir build
cd build

# Run CMake configuration
cmake .. -DCMAKE_PREFIX_PATH=<nvCOMP sysroot path> \
         -DCMAKE_BUILD_TYPE=Release

# Run the actual build
cmake --build . --config Release --parallel 14
```
