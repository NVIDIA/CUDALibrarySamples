# MathDx Tile - API Examples

The C++ examples are shipped under `examples/cpp` in the MathDx Tile package. The Python examples are not shipped in the package; they are in the `python` folder of the [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples/tree/main/MathDx/Tile) repository.

## Description

This folder demonstrates the MathDx Tile APIs: routines from cuFFTDx, cuSolverDx, cuBLASDx (`triangular_solve`), and nvCOMPDx, called from CUDA Tile kernels. The C++ examples use CUDA Tile C++ and the Python examples use cuTile Python through `nvmath.tile`. The two suites mostly mirror each other, with the same file names, so examples can be read side by side. Each example launches one or more Tile kernels and checks its numerical result.

* [MathDx Tile download page](https://developer.nvidia.com/mathdx-tile-downloads)
* [MathDx Tile API documentation](https://docs.nvidia.com/cuda/mathdx_tile/index.html)

## Requirements

* [See MathDx Tile requirements](https://docs.nvidia.com/cuda/mathdx_tile/getting_started.html#requirements)
* Linux (x86_64, aarch64) or Windows (x86_64, arm64) system with installed NVIDIA drivers (r580 or newer)
* NVIDIA GPU of Ampere (SM80) or newer architecture
* C++ examples:
  * [MathDx Tile package](https://developer.nvidia.com/mathdx-tile-downloads)
  * CUDA Toolkit 13.5
  * CMake 3.27 or newer
* Python examples:
  * A Python version supported by [cuTile Python](https://docs.nvidia.com/cuda/cutile-python/quickstart.html)
  * `nvmath-python[cu13-dx]` 1.0.1 or newer
  * CuPy, for the FFT examples

**MathDx Tile 0.5.0 requires CUDA 13.5 exactly.** Other CUDA releases, including later 13.x, are not supported.

See the [compatibility table](https://docs.nvidia.com/cuda/mathdx_tile/release_notes.html#compatibility) for later releases and the [Getting Started](https://docs.nvidia.com/cuda/mathdx_tile/getting_started.html) guide for installation.

## Build and run

Run the commands below from this directory. With the extracted MathDx Tile package, run them from its `examples` directory, with the package as `<mathdx_tile_path>`.

### C++

The C++ sources are in `cpp`. `error_checks.hpp`, `error_metrics.hpp`, and `device_buffer.hpp` hold the CUDA-library checks, relative-error norm, and device allocation used by every example family, and `sm_arch.hpp` gives `suggested_tile_config` the SM architecture. `common.hpp` and `common_perf_measure.hpp` hold the solver matrix generator and the timing loop and FLOP counts of the advanced solver examples.

* The examples use `find_package(mathdx_tile CONFIG REQUIRED)`, which consumes the `mathdx_tile-config.cmake` shipped under `lib/cmake/mathdx_tile/` in the MathDx Tile package. Point CMake at the package with `-DCMAKE_PREFIX_PATH=<mathdx_tile_path>` and at the Tile compiler with `-DCMAKE_CUDA_COMPILER`.
* The kernels are compiled with `nvcc --enable-tile --tile-preview=simt`; the `mathdx_tile::` CMake targets add these flags.
* `CMAKE_CUDA_ARCHITECTURES` defaults to `native`.
* The reference checks link cuBLAS, cuSolver, and cuFFT from the full toolkit at `CUDAToolkit_ROOT`, whose `nvcc` also compiles the cuFFT LTO callback.

#### Linux

```bash
cmake -S cpp -B build \
  -DCMAKE_CUDA_COMPILER=<cuda_root_directory>/bin/nvcc \
  -DCUDAToolkit_ROOT=<cuda_root_directory> \
  -DCMAKE_PREFIX_PATH=<mathdx_tile_path>
cmake --build build
# Run
ctest --test-dir build --output-on-failure
```

#### Windows

Use a Visual Studio Native Tools command prompt with CMake and Ninja on `PATH`. The example uses the Ninja generator. The Visual Studio generator takes the CUDA toolkit registered in Visual Studio, not `CMAKE_CUDA_COMPILER` or `CUDAToolkit_ROOT`.

```bat
REM tileiras loads mathdx64_0.dll from the package while compiling (bin\arm64 on Windows on Arm)
set PATH=<mathdx_tile_path>\bin\x64;%PATH%
cmake -S cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_COMPILER=<cuda_root_directory>\bin\nvcc.exe ^
  -DCUDAToolkit_ROOT=<cuda_root_directory> ^
  -DCMAKE_PREFIX_PATH=<mathdx_tile_path>
cmake --build build
REM Run
ctest --test-dir build --output-on-failure
```

#### Building without CMake

An example can also be built without CMake: the `mathdx_tile::cusolverdx` target only adds the two Tile flags, the tilelib, and the MathDx headers:

```bash
CTK=<cuda_root_directory>
MATHDX_TILE=<mathdx_tile_path>

# reg_least_squares uses the cuSolverDx tilelib
"$CTK/bin/nvcc" -std=c++20 \
  -arch=native \
  --enable-tile --tile-preview=simt \
  -Xtileiras=-l="$MATHDX_TILE/lib/cusolverdx.tilelib" \
  -I cpp \
  -I "$MATHDX_TILE/include" -I "$CTK/include" \
  cpp/cusolverdx/advanced/reg_least_squares.cu -o reg_least_squares \
  -L"$CTK/lib64" -lcudart -lcublas -lcusolver

# blocked_potrf also uses triangular_solve, so it needs the cuBLASDx tilelib too
"$CTK/bin/nvcc" -std=c++20 \
  -arch=native \
  --enable-tile --tile-preview=simt \
  -Xtileiras=-l="$MATHDX_TILE/lib/cusolverdx.tilelib" \
  -Xtileiras=-l="$MATHDX_TILE/lib/cublasdx.tilelib" \
  -I cpp \
  -I "$MATHDX_TILE/include" -I "$CTK/include" \
  cpp/cusolverdx/advanced/blocked_potrf.cu -o blocked_potrf \
  -L"$CTK/lib64" -lcudart -lcublas -lcusolver
```

`-I cpp` is what finds `common.hpp` and `common_perf_measure.hpp`. The toolkit include is there for `cublas_v2.h` and `cusolverDn.h`. An example from the `cusolverdx/simple` group builds the same way without `-lcublas -lcusolver` and without the toolkit include. Keep each tilelib next to `libmathdx.so` in the package `lib` directory, which is where the compiler loads `libmathdx` from. To run an example, add `$CTK/lib64` to `LD_LIBRARY_PATH`.

### Python

Install the packages listed under [Requirements](#requirements), then run any script from the directory that contains `python`:

```bash
python -m pip install "nvmath-python[cu13-dx]>=1.0.1" cupy-cuda13x
python python/linalg/simple/cholesky_factor.py
```

* Linear algebra and compression examples use NumPy arrays, with `cuda.core` helpers in `python/common.py`.
* FFT examples use CuPy.
* With the `cu13-dx` extra, no separate CUDA Toolkit is needed, so the advanced solver examples run on a machine without the full toolkit that the C++ build needs for its host references.
* `filtered_backprojection_3d.py --plot` also needs Matplotlib (`python -m pip install matplotlib`).

To run the whole suite, both groups:

```bash
status=0
for f in python/linalg/{simple,advanced}/*.py python/fft/*.py python/compress/*.py; do
  python "$f" || { echo "FAILED: $f"; status=1; }
done
test "$status" -eq 0
```

See [Getting Started](https://docs.nvidia.com/cuda/mathdx_tile/getting_started.html) for more details.

## Examples

See also the [Examples](https://docs.nvidia.com/cuda/mathdx_tile/examples.html) section of the MathDx Tile documentation.

The examples are grouped by family. The C++ and Python columns show which suites have each example; `fft_autotuning` and `sglang_ipm` are Python only.

Tile extents must be compile-time powers of two, as the current tilelibs require. Only the tile extents are restricted, not the tensor a tile is loaded from.

### FFT

The C++ examples are in `cpp/cufftdx` and the Python examples in `python/fft`. The C++ variants validate against cuFFT, except `filtered_backprojection_3d`, which checks that its output is finite. The Python variants use direct or CuPy references.

| Example                    | C++ | Python | What it does                                                                                                                     | Routines                       |
| -------------------------- | :-: | :----: | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| fft_convolution_1d         |  ✓  |   ✓    | Real 1D linear convolution in one fused kernel                                                                                   | `rfft`, `irfft`                |
| fft_convolution_3d         |  ✓  |   ✓    | 8x8x8 complex convolution, all three axes in one kernel                                                                          | `fft`, `ifft`                  |
| fft_convolution_3d_real    |  ✓  |   ✓    | 3D real convolution with `rfft` and `irfft`                                                                                      | `fft`, `ifft`, `rfft`, `irfft` |
| fft_convolution_3d_split   |  ✓  |   ✓    | 128^3 convolution split into three kernels, staged through global memory                                                         | `fft`, `ifft`                  |
| fft_convolution_5d         |  ✓  |   ✓    | 8^5 convolution, staged across five axes                                                                                         | `fft`, `ifft`                  |
| mixed_precision_fft_1d     |  ✓  |   ✓    | Float input and output around a double-precision transform                                                                       | `fft`                          |
| filtered_backprojection_3d |  ✓  |   ✓    | 3D ramp-filtered backprojection in one fused Tile kernel, checked for finite output                                              | `rfft`, `irfft`                |
| fft_autotuning             |     |   ✓    | Sweeps the batch extent, the CTA size, and the `occupancy` hint for a batched 1D FFT, validating each candidate before timing it | `fft`, `suggested_tile_config` |

### Linear algebra

The C++ examples are in `cpp/cusolverdx` and the Python examples in `python/linalg`. The `simple` examples show one routine each, solving a single system. The `advanced` examples are larger and combine several routines. `reg_least_squares` and `blocked_potrf`, which exist in both suites, are batched with one block per problem, report throughput, and match the examples of the same name under `example/cusolverdx/10_Advanced` in the cuSolverDx package. The C++ advanced examples additionally check themselves against the equivalent batched cuBLAS and cuSolver host APIs, and report GFLOP/s for the reference. The Python versions check the problem's optimality or reconstruction condition.

`sglang_ipm.py` instead solves one application-sized problem. It is adapted from SGLang's LPLB solver and the DeepSeek LPLB `minilp` implementation, keeps the application dimensions and algorithm, and checks the result against the same algorithm in float64 NumPy.

| Group    | Example                   | C++ | Python | What it does                                                                                                                                                                                                  | Routines                                                                                                 |
| -------- | ------------------------- | :-: | :----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| simple   | cholesky_factor           |  ✓  |   ✓    | Factors a positive-definite matrix                                                                                                                                                                            | `cholesky_factor`                                                                                        |
|          | cholesky_solve            |  ✓  |   ✓    | Factors and solves a positive-definite system                                                                                                                                                                 | `cholesky_solve`                                                                                         |
|          | lu_solve                  |  ✓  |   ✓    | Solves a general system with partial-pivot LU                                                                                                                                                                 | `lu_solve`                                                                                               |
|          | factor_reuse              |  ✓  |   ✓    | Factors once, then reuses the Cholesky and LU factors in separate kernels                                                                                                                                     | `cholesky_factor`, `cholesky_solve_using_factor`, `lu_factor`, `lu_solve_using_factor`                   |
|          | modified_lu_factor        |  ✓  |   ✓    | Computes a modified LU factorization                                                                                                                                                                          | `modified_lu_factor`                                                                                     |
|          | qr                        |  ✓  |   ✓    | Factors a rectangular matrix, forms its thin Q, and applies the implicit Q to another matrix                                                                                                                  | `qr_factor`, `qr_generate_q`, `qr_multiply_by_q`                                                         |
|          | lq                        |  ✓  |   ✓    | Factors a wide matrix, generates Q explicitly, and applies the implicit Q to another matrix                                                                                                                   | `lq_factor`, `lq_generate_q`, `lq_multiply_by_q`                                                         |
|          | least_squares_solve       |  ✓  |   ✓    | Solves one overdetermined least-squares problem twice, against a tall `A` and against the transpose of the equivalent wide matrix                                                                             | `least_squares_solve`                                                                                    |
|          | tridiagonal_solve         |  ✓  |   ✓    | Solves a diagonally dominant tridiagonal system                                                                                                                                                               | `tridiagonal_solve`                                                                                      |
|          | eig_hermitian             |  ✓  |   ✓    | Computes eigenvalues and eigenvectors of a symmetric matrix                                                                                                                                                   | `eig_hermitian`                                                                                          |
|          | eig_hermitian_tridiagonal |  ✓  |   ✓    | Computes eigenvalues of a symmetric tridiagonal matrix                                                                                                                                                        | `eig_hermitian_tridiagonal`                                                                              |
|          | generalized_eigh          |  ✓  |   ✓    | Computes generalized symmetric-definite eigenvalues in one call, and again by reducing to a standard eigenproblem by hand                                                                                     | `cholesky_factor`, `generalized_eig_hermitian`, `generalized_to_standard_eig_hermitian`, `eig_hermitian` |
|          | svd                       |  ✓  |   ✓    | Computes the thin SVD of a rectangular matrix, both with and without singular vectors                                                                                                                         | `svd`                                                                                                    |
|          | svd_bidiagonal            |  ✓  |   ✓    | Computes the singular values of a bidiagonal matrix                                                                                                                                                           | `svd_bidiagonal`                                                                                         |
| advanced | reg_least_squares         |  ✓  |   ✓    | Solves 10000 regularized 100x8 least-squares problems two ways: a fused kernel that builds the normal equations with tile matmuls and factors them in registers, and a Householder QR on the augmented system | `cholesky_solve`, `least_squares_solve`                                                                  |
|          | blocked_potrf             |  ✓  |   ✓    | Factors 400 512x512 matrices that are far too large for one solver call, by walking 32x32 tiles in a left-looking order                                                                                       | `cholesky_factor`, `triangular_solve`                                                                    |
|          | sglang_ipm                |     |   ✓    | Solves one 24x42 SGLang MoE load-balancing problem with an affine-scaling interior-point method and a 16+8 blocked Cholesky factorization                                                                     | `cholesky_factor`, `triangular_solve`                                                                    |

Every routine in the table is available in both APIs, and every one has an example. In C++, `triangular_solve`, `tril`, and `triu` are in `cublasdx_tile.hpp` under `cublasdx::tiles`, and the other linear algebra routines are in `cusolverdx_tile.hpp` under `cusolverdx::tiles`. Python exposes all of them from `nvmath.tile.linalg`. `triangular_solve` is served by the cuBLASDx tilelib, so kernels that combine it with cuSolverDx routines, such as `blocked_potrf`, also link `mathdx_tile::cublasdx`.

A system whose size is not a power of two can still be solved by padding the tile with zero rows, as `reg_least_squares` does to keep the 100x8 problem size of the cuSolverDx example it mirrors. The C++ examples use real `float` tiles except `blocked_potrf`, which uses `double`.

### Compression

The C++ examples are in `cpp/nvcompdx` and the Python examples in `python/compress`. Each example compresses four chunks per kernel with nvCOMPDx, one chunk per block, using `compress` and `decompress`, and checks an exact round trip. The tile examples keep the chunk and its payload in tiles. The global-memory examples read and write global-memory buffers in place, and only the byte counts are tiles.

| Example  | C++ | Python | Algorithm | Uncompressed data | Data and payload |
| -------- | :-: | :----: | --------- | ----------------- | ---------------- |
| lz4_tile |  ✓  |   ✓    | LZ4       | `uint8`           | Tiles            |
| lz4_gmem |  ✓  |   ✓    | LZ4       | `uint8`           | Global memory    |
| ans_tile |  ✓  |   ✓    | ANS       | `float16`         | Tiles            |
| ans_gmem |  ✓  |   ✓    | ANS       | `float16`         | Global memory    |
