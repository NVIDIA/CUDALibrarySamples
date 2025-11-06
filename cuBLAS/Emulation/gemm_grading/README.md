# Matrix Multiplication Algorithm Detection and Accuracy Grading

## Description

This repository contains two programs, which are based on the methodology described in [1,2].

- `test_blas3`: Tests that detect the type of matrix multiplication algorithm being used based on numerical properties of the results. The tests are designed to distinguish between:
  - Conventional O(n³) floating-point [D,Z]GEMM
  - Fixed-point O(n³) [D,Z]GEMM
  - Conventional floating-point Strassen [D,Z]GEMM
  - Fixed-point Strassen [D,Z]GEMM

- `grade_blas3`: Evaluates the error bound for Grade A compliance, the most stringent criterion for assessing the accuracy of an implementation of matrix multiplication.


We gratefully acknowledge Prof. James W. Demmel for his generous support, for sharing the manuscript [2], and for his review and feedback on this work prior to publication.

## Usage

Two executables are built:

- `test_blas3`: Detects the type of matrix multiplication algorithm based on numerical properties of the results.
- `grade_blas3`: Evaluates error bounds for Grade A compliance.

```bash
./test_blas3 [options] [matrix_size]
./grade_blas3 [options] [matrix_size]
```

Both executables accept the same command-line options and defaults.

### Command-line Options

- `-s, --host-strassen-native`              : Use double arithmetic floating-point Strassen's algorithm
- `-s -z, --host-strassen-native --complex` : Use complex arithmetic floating-point Strassen's algorithm
- `-d, --host-gemm-native`                  : Use floating-point DGEMM provided by the linked BLAS library
- `-d -z, --host-gemm-native --complex`     : Use floating-point ZGEMM provided by the linked BLAS library
- `-f, --host-strassen-fixed`               : Use double arithmetic fixed-point Strassen's algorithm
- `-f -z, --host-strassen-fixed --complex`  : Use complex arithmetic fixed-point Strassen's algorithm
- `-c, --cuda-gemm-native`                  : Use floating-point CUDA DGEMM implementation
- `-c -z, --cuda-gemm-native --complex`     : Use floating-point CUDA ZGEMM implementation
- `-e, --cuda-gemm-emu`                     : Use emulated CUDA DGEMM based on Ozaki's scheme (default)
- `-e -z, --cuda-gemm-emu --complex`        : Use emulated CUDA ZGEMM based on Ozaki's scheme
- `-h, --help`                              : Show help message

### Matrix Size

You can specify one or more matrix dimensions to test. If no size is provided, the program will run tests on default sizes: 16, 32, 64, 128, 256, 512, and 1024.

### Examples

```bash

# Test CUDA DGEMM implementation
./test_blas3 -c 512

# Test emulated CUDA DGEMM based on Ozaki's scheme on the default matrix sizes
./test_blas3 -e

# Test native ZGEMM on a 128x128 matrix
./test_blas3 --complex -d 128

# Test double precision fixed-point Strassen's algorithm on multiple sizes
./test_blas3 -f 64 128 256

# Show help message
./test_blas3 --help
```

## Building

### Prerequisites

- A Linux/Windows system with recent NVIDIA drivers
- CMake version 3.18 minimum
- C++ compiler with C++17 support
- BLAS library (the specific BLAS used is whichever implementation is linked against)
- CUDA Toolkit 13.0u2

### Building with CMake (Linux)

```bash
mkdir -p build
cd build
cmake ..
make -j
```


## Test Methodology

The program `test_blas3` uses three main tests to detect the type of matrix multiplication algorithm:

1. Test 2: Distinguishes between Strassen-like and conventional O(n³) GEMM
2. Test 4: Distinguishes between fixed-point and conventional floating-point GEMM
3. Test 6: Distinguishes between conventional floating-point and fixed-point Strassen GEMM

Each test uses carefully constructed matrices to reveal numerical properties that are characteristic of each algorithm type. The summary of the tests indicates what algorithm
was detected. Further, the status field marks a run as safe if the implementation being tested could not be distinguished from conventional O(n³) GEMM and as unsafe otherwise.

### Examples

(1) Test emulated CUDA DGEMM based on Ozaki's scheme on 128x128 and 256x256 matrices
```
$  ./test_blas3 -e 128 256
```
Sample output
```
Analyzing emulated CUDA GEMM (double)

      Size | Algorithm Detected                                 |     Status
-----------+----------------------------------------------------+-----------
       128 | Conventional O(n³) floating-point GEMM            | Safe
       256 | Conventional O(n³) floating-point GEMM            | Safe
```

(2) Test standard ZGEMM on a 128x128 matrix with complex numbers
```
./test_blas3 --complex -d 128
```
Sample output
```
Analyzing standard GEMM provided by linked BLAS library (complex double)

      Size | Algorithm Detected                                 |     Status
-----------+----------------------------------------------------+-----------
       128 | Conventional O(n³) floating-point GEMM            | Safe
```

(3) Test fixed-point Strassen on multiple sizes
```
./test_blas3 -f 64 128 256
```
Sample output
```
Analyzing fixed-point Strassen's algorithm (double)

      Size | Algorithm Detected                                 |     Status
-----------+----------------------------------------------------+-----------
        64 | Fixed-point Strassen GEMM                          | Unsafe
       128 | Fixed-point Strassen GEMM                          | Unsafe
       256 | Fixed-point Strassen GEMM                          | Unsafe
```

## Grading Methodology

The program `grade_blas3` concerns the componentwise relative error bound
\[ |fl((A*B)(i,j)) - (A*B)(i,j)| \le f(n)\,\epsilon\,(|A|\,|B|)(i,j) \]
where, for Grade A compliance [1], \(f(n)\) must not exceed linear growth.

The program returns the maximum and average componentwise relative error
\[ \frac{|fl((A*B)(i,j)) - (A*B)(i,j)|}{\epsilon\,(|A|\,|B|)(i,j)} \]
for square matrix multiplication \(C = AB\). The results can be analyzed to assess error growth. If the maximum error does not grow more than linearly, the implementation meets Grade A.

### Examples

(1) Evaluate CUDA DGEMM based on Ozaki's scheme on the default matrix sizes
```
./grade_blas3 -e
```
Sample output
```
n = 16 max error = 1.81489 avg error = 0.4135
n = 32 max error = 2.9519 avg error = 0.556189
n = 64 max error = 3.52298 avg error = 0.764285
n = 128 max error = 5.59914 avg error = 1.04993
n = 256 max error = 8.62172 avg error = 1.47098
n = 512 max error = 11.9877 avg error = 2.04442
n = 1024 max error = 17.6456 avg error = 2.86258
For grade A compliance, errors must not exceed linear growth.
Under the assumption that the error grows linearly over the tested
range, the fitted model in log-log scale, log(error) = a + b * log(n), has slope 0.542104.
For grade A compliance, the slope in log-log scale must be <= 1.
We highlight that this is not a proof; a formal claim requires proper statistical testing.
```
(2) Evaluate double complex Strassen on selected matrix sizes
```
./grade_blas3 -s --complex 128 256 512 1024
```
Sample output
```
n = 128 max error = 19.1117 avg error = 2.13447
n = 256 max error = 56.5692 avg error = 3.29812
n = 512 max error = 112.218 avg error = 5.23016
n = 1024 max error = 248.041 avg error = 8.45242
For grade A compliance, errors must not exceed linear growth.
Under the assumption that the error grows linearly over the tested
range, the fitted model in log-log scale, log(error) = a + b * log(n), has slope 1.20823.
For grade A compliance, the slope in log-log scale must be <= 1.
We highlight that this is not a proof; a formal claim requires proper statistical testing.
```

## References

[1] Jim Demmel, Xiaoye Li, Julien Langou, Weslley Pereira, Mark Gates, Cindy Rubio Gonzalez (2024), "How to grade the accuracy of an implementation of the BLAS," slides available at: https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Grading_BLAS.pdf

[2] Jim Demmel et al. (2025), "More aggressive (sparse) BLAS testing, to identify aggressive optimizations."
    Private communication. Unpublished manuscript, referenced with author approval.
