# nvCOMP Benchmarks

This folder contains benchmarks demonstrating the performance (and usage) of the nvCOMP C++ API. They test compressing and decompressing input data, reporting the compression throughput, decompression throughput, and compression ratio. You can run the benchmark executables on your own files, or test standard benchmark datasets, as described below.

## Benchmarks

* [ANS benchmark](benchmark_ans_chunked.cu)

    The sample demonstrates the ANS compression and decompression usage and performance via nvCOMP.

    ```
    benchmark_ans_chunked {-f|--input_file} <input_file>
                          [{-t|--type} {uint8|float16}]
    ```

* [Bitcomp benchmark](benchmark_bitcomp_chunked.cu)

    The sample demonstrates the Bitcomp compression and decompression usage and performance via nvCOMP.

    ```
    benchmark_bitcomp_chunked {-f|--input_file} <input_file>
                              [{-t|--type} {char|uchar|short|ushort|int|uint|longlong|ulonglong}]
                              [{-a|--algorithm} {0|1}]
    ```

* [Cascaded benchmark](benchmark_cascaded_chunked.cu)

    The sample demonstrates the Cascaded compression and decompression usage and performance via nvCOMP.

    ```
    benchmark_cascaded_chunked {-f|--input_file} <input_file>
                               [{-t|--type} {char|uchar|short|ushort|int|uint|longlong|ulonglong}]
                               [{-r|--num_rles} <num_RLE_passes>]
                               [{-d|--num_deltas} <num_delta_passes>]
                               [{-b|--num_bps} <do_bitpack_0_or_1>]
    ```

* [Deflate benchmark](benchmark_deflate_chunked.cu)

    The sample demonstrates the Deflate compression and decompression usage and performance via nvCOMP.

    ```
    benchmark_deflate_chunked {-f|--input_file} <input_file>
                              [{-a|--algorithm} {1|2|3|4|5}]
    ```

* [GDeflate benchmark](benchmark_gdeflate_chunked.cu)

    The sample demonstrates the GDeflate compression and decompression usage and performance via nvCOMP.

    ```
    benchmark_gdeflate_chunked {-f|--input_file} <input_file>
                               [{-a|--algorithm} {0|1|2|3|4|5}]
    ```

* [LZ4 benchmark](benchmark_lz4_chunked.cu)

    The sample demonstrates the LZ4 compression and decompression usage and performance via nvCOMP.

    ```
    benchmark_lz4_chunked {-f|--input_file} <input_file>
                          [{-t|--type} {bits|char|uchar|short|ushort|int|uint}]
    ```

* [Snappy benchmark](benchmark_snappy_chunked.cu)

    The sample demonstrates the Snappy compression and decompression usage and performance via nvCOMP.

    ```
    benchmark_snappy_chunked {-f|--input_file} <input_file>
    ```

* [Zstandard benchmark](benchmark_zstd_chunked.cu)

    The sample demonstrates the Zstandard compression and decompression usage and performance via nvCOMP.

    ```
    benchmark_zstd_chunked {-f|--input_file} <input_file>
    ```

* [High-level interface benchmark](benchmark_hlif.cpp)

    The sample demonstrates the high-level interface usage and performance with the selected algorithm via nvCOMP.

    ```
    benchmark_hlif {ans|bitcomp|cascaded|gdeflate|deflate|lz4|snappy|zstd}
                {-f|--input_file} <input_file>
                [{-c|--chunk_size} <chunk size, e.g. 64kB>]
                [{-g|--gpu} <device ordinal, e.g. 0>]
                [{-n|--num_iters} <number of iterations used for performance benchmarking, e.g. 1>]
                [{-t|--type} {char|short|int|longlong|float16}]
                [{-m|--memory}]
    ```

Most of the benchmark executables also support:

```
{-g|--gpu} <gpu_num>                                 GPU device ordinal to use for benchmarking
{-w|--warmup_count} <num_iterations>                 The number of warmup (unrecorded) iterations to perform
{-i|--iteration_count} <num_iterations>              The number of recorded iterations to perform
{-m|--multiple_of} <num_bytes>                       The number of bytes the input is padded to, such that its overall length
                                                     becomes a multiple of the given argument (in bytes). Only applicable to
                                                     data without page sizes.
{-x|--duplicate_data} <num_copies>                   The number of copies to make of the input data before compressing
{-c|--csv_output} {false|true}                       When true, the output is in comma-separated values (CSV) format
{-e|--tab_separator} {false|true}                    When true and --csv_output is true, tabs are used to separate values,
                                                     instead of commas
{-p|--chunk_size} <num_bytes>                        Chunk size when splitting uncompressed data
{-oc|--output_compressed_file} <output_file>         Path for the resulting compressed data chunks
{-o|--output_decompressed_file} <output_file>        Path for the resulting decompressed data (chunks)
{-single|--single_output_buffer} {false|true}        Use a single buffer for decompressing data
{-compressed|--compressed_inputs} {false|true}       The input dataset is compressed
{-?|--help}                                          Show help text for the benchmark
```

For compressors that accept a data type option (`--type`), input data for which all of the input matches that type will usually compress better than arbitrary data. The sizes of the types are 1 byte for `char/uchar/bits`, 2 bytes for `short/ushort`, 4 bytes for `int/uint`, and 8 bytes for `longlong/ulonglong`. Input files whose sizes aren't multiples of the data type size are unsupported. However, one can apply a padding to the input data via `--multiple_of` to make sure that the overall input size becomes an integer multiple of the data type size.

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
                         -compressed true \
                         -single true \
                         --output_decompressed_file replicated.dump

# Perform a verification against the original input
diff <input data> replicated.dump
```

If you would like to use standard benchmark datasets, there are two described here, "TPC-H" and "Mortgage", both of which are in the form of text tables that will first need to have a column extracted and converted to binary data, using the `text_to_binary.py` script.

To obtain TPC-H data tables, randomly generating a simulated database table of purchases:
- Clone and compile https://github.com/electrum/tpch-dbgen
- Run `./dbgen -s <scale factor>` to generate data in the file `lineitem.tbl`.  A larger scale factor will result in a larger generated table.

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

Below are some example benchmark results running the LZ4 compressor via the high-level interface (hlif) and the low-level interface (chunked) on a A100 GPU for the Mortgage 2009Q2 column 0:

```
./benchmark_hlif lz4 -f /data/nvcomp/benchmark/mortgage-2009Q2-col0-long.bin
----------
uncompressed (B): 329055928
comp_size: 8582564, compressed ratio: 38.34
compression throughput (GB/s): 90.48
decompression throughput (GB/s): 312.81
```

```
./benchmark_lz4_chunked -f /data/nvcomp/benchmark/mortgage-2009Q2-col0-long.bin
----------
uncompressed (B): 329055928
comp_size: 8461988, compressed ratio: 38.89
compression throughput (GB/s): 95.87
decompression throughput (GB/s): 320.70
```

## Building (x86-64, or aarch64)

The compilation of benchmarks does not require additional external libraries:

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