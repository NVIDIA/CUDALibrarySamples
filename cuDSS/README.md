# cuDSS Library

## Description

This folder demonstrates main cuDSS APIs usage.

Note: samples are updated to work with the latest version of cuDSS available (currently: 1.0.0) and might be incompatible with older versions.
For further details, please check the release notes in the official documentation.

[cuDSS Documentation](https://docs.nvidia.com/cuda/cudss/index.html)

## cuDSS Samples

* [Simple workflow for solving a real-valued sparse linear system](simple/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system without any advanced features

* [Simple workflow for solving a real-valued sparse linear system from a matrix market file](simple_matrix_market/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system read from a `.mtx` file without any advanced features

* [Simple workflow for solving a complex-valued sparse linear system](simple_complex/)

    The sample demonstrates how to use cuDSS for solving a complex sparse linear system without any advanced features

* [Hybrid host/device memory mode](simple_hybrid_memory_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the hybrid host/device memory mode

* [Multi-gpu multi-node (MGMN) mode](simple_mgmn_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multi-GPU multi-node mode (available since cudss 0.3.0)

    Note: building MGMN mode samples require extra options for the communication backend at build and runtime, and must be run with mpirun

* [Getter/setter APIs for cudssConfig, cudssData and cudssHandle objects](simple_get_set/)

    The sample extends the previous code to demonstrate how extra settings can be applied to solving sparse linear
    systems and how to retrieve extra information from the solver

* [Device memory handler APIs for cudssDeviceMemHandler_t defined device memory pools or allocators](simple_memory_handler/)

    The sample demonstrates how to use cuDSS device memory handler APIs (available since cudss 0.2.0) for providing user-defined device memory pools
    or allocators

* [Non-uniform batch API](simple_batch/)

    The sample demonstrates how to use cuDSS batch APIs (available since cudss 0.4.0) for solving a non-uniform batch of linear systems where matrices and righthand sides
    can all be different

* [Hybrid execution mode](simple_hybrid_execute_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the hybrid execute mode (available since cudss 0.5.0)

* [Multi-threaded (MT) mode](simple_multithreaded_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multithreading (MT) mode (available since cudss 0.5.0)

* [Sparse matrix interactions](simple_sparse_matrix_helpers/)

    The sample demonstrates how to interact with sparse matrices in cuDSS

* [Dense matrix interactions](simple_dense_matrix_helpers/)

    The sample demonstrates how to interact with dense matrices in cuDSS

* [Batch matrix interactions](simple_batch_sparse_matrix_helpers/)

    The sample demonstrates how to interact with batched systems in cuDSS (available since cudss 0.5.0)

* [Batch system solving](simple_uniform_batch/)

    The sample demonstrates how to use cuDSS for solving a uniform batched system (available since cudss 0.6.0)

* [Extracting reordering information](simple_reordering_phase/)

    The sample demonstrates how to use cuDSS to extract the reordering information it uses internally immediately after it is generated (available since cudss 0.6.0)

* [Multi-gpu multi-node (MGMN) mode with a distributed matrix](simple_mgmn_distributed_matrix/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multi-GPU multi-node mode with a distributed system matrix (available since cudss 0.6.0)

    Note: building MGMN mode samples require extra options for the communication backend at build and runtime, and must be run with mpirun

* [Multi-gpu single-node (MG) mode](simple_mg_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multi-GPU mode (without any distributed communication backend) (available since cudss 0.7.0)

* [Schur complement computation](simple_schur_complement/)

    The sample demonstrates how to use cuDSS for computing the Schur complement matrix (and how it can be used for solving the full system) (available since cudss 0.7.0)

* [Residual computation](simple_residual/)

    The sample demonstrates how one can compute the relative residual norm to estimate accuracy of cuDSS (available since cudss 0.7.0)

* [Helper for testing the communication layer library](test_communication_layer/)

    The sample demonstrates how one can check functional health of a distributed communication backend and the associated communication layer library of cuDSS (available since cudss 0.7.0)

* [Helper for testing the threading layer library](test_threading_layer/)

    The sample demonstrates how one can check functional health of a threading backend and the associated threading layer library of cuDSS (available since cudss 0.7.0)

* [Higher than double-precision solve](simple_high_precision/)

    The sample demonstrates how to use cuDSS for solving with higher than double-precision (available since cudss 0.8.0)

    Note: This requires a GPU with compute capability 9.0 or higher.

* [Pinned host memory handler](simple_pinned_memory_handler/)

    The sample demonstrates how to provide custom pinned host memory allocation and deallocation callbacks (available since cudss 1.0.0).

## Fortran Samples

Fortran interfaces and examples are available since cudss 1.0.0. The examples provide the same cuDSS workflows in two flavors:

- **Standard Fortran** (`.f90`) under `fortran/<name>/` — build with any Fortran compiler that supports C interoperability (set `FC`, e.g. `gfortran`, `ifx`, or `nvfortran`).
- **CUDA Fortran** (`.cuf`) under `nvfortran/<name>/` — requires the NVIDIA HPC compiler `nvfortran`.

Each sample directory contains the example source, the support modules it uses, and both a `CMakeLists.txt` and a `Makefile`. The cuDSS Fortran interface itself is compiled from the installed cuDSS (`include/fortran/`).

The following `<name>` samples are available under both `fortran/` and `nvfortran/`:

* [`simple`](fortran/simple/) — real-valued sparse linear system
* [`simple_complex`](fortran/simple_complex/) — complex-valued sparse linear system
* [`get_set`](fortran/get_set/) — getter/setter APIs for config, data, and handle objects
* [`memory_handler`](fortran/memory_handler/) — user-defined device memory pools/allocators
* [`batch`](fortran/batch/) — non-uniform batch API
* [`uniform_batch`](fortran/uniform_batch/) — uniform batch solve
* [`hybrid_memory_mode`](fortran/hybrid_memory_mode/) — hybrid host/device memory mode
* [`hybrid_execute_mode`](fortran/hybrid_execute_mode/) — hybrid host/device execution mode
* [`multithreaded_mode`](fortran/multithreaded_mode/) — multi-threaded (MT) mode
* [`mg_mode`](fortran/mg_mode/) — multi-GPU single-node (MG) mode
* [`reordering_phase`](fortran/reordering_phase/) — extract internal reordering information
* [`schur_complement`](fortran/schur_complement/) — Schur complement computation
* [`residual`](fortran/residual/) — relative residual norm computation
* [`high_precision`](fortran/high_precision/) — higher-than-double-precision solve (needs compute capability 9.0+)
* [`sparse_matrix_helpers`](fortran/sparse_matrix_helpers/) — sparse matrix interactions
* [`dense_matrix_helpers`](fortran/dense_matrix_helpers/) — dense matrix interactions
* [`batch_sparse_matrix_helpers`](fortran/batch_sparse_matrix_helpers/) — batched sparse matrix interactions
* [`mgmn_mode`](fortran/mgmn_mode/) — multi-GPU multi-node (MGMN) mode

    Note: MGMN samples need a communication backend at build and run time and must be launched with mpirun.

## Building the Fortran samples

Use cuDSS 1.0.0 with its installed `include/fortran/` sources and a compatible CUDA Toolkit.
Each project builds independently; no precompiled Fortran modules are required.

For standard Fortran, for example:

```sh
cmake -S fortran/simple -B build-simple -DCMAKE_Fortran_COMPILER=gfortran -Dcudss_DIR=/path/to/cudss/lib/cmake/cudss
cmake --build build-simple
./build-simple/simple_example
```

For [CUDA Fortran](nvfortran/simple/), use `nvfortran` as the Fortran compiler and
`nvfortran/simple` as the source directory. The [standard Fortran](fortran/simple/)
and CUDA Fortran trees provide the same workflows listed above.

Alternatively, run `make` in an individual sample directory, setting `FC`,
`CUDSS_INC`, `CUDSS_LIB`, and, for standard Fortran, `CUDA_INC` and `CUDA_LIB`.
Run the resulting executable directly (for example, `./simple`).
MPI Fortran module files are compiler-specific; use an MPI installation built
with your Fortran compiler. The MGMN sample requires MPI C and Fortran compilers;
set `MPICC` and `MPIF90` for Makefile builds, then launch with
`mpirun -np 2 ./mgmn_mode /path/to/libcudss_commlayer_openmpi.so`.
For CMake MGMN builds with multiple MPI installations, select matching wrappers
with `-DMPI_Fortran_COMPILER=/path/to/mpifort` and `-DMPI_C_COMPILER=/path/to/mpicc`.
The multithreaded sample needs the threading-layer library as its first argument
or through `CUDSS_THREADING_LIB`. Multi-GPU samples require at least two GPUs,
and high-precision samples require compute capability 9.0 or higher.

## License and Contributions

The examples are licensed under Apache-2.0; see the per-file license notices and
[repository license](../LICENSE.md). The Fortran bindings used from the installed
cuDSS package are licensed under BSD-3-Clause; see that package's license file.
Contributions follow the [repository contribution guidelines](../CONTRIBUTING.md).
