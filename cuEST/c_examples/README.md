# cuEST C API Examples

This directory contains standalone C examples demonstrating the cuEST C API.

For full API documentation, see https://docs.nvidia.com/cuda/cuest/.

## Prerequisites

- CUDA Toolkit 12 or later
- CMake 3.20 or later
- A C compiler

## Building

```bash
mkdir build && cd build
cmake .. -DCUEST_INSTALL_DIR=/path/to/cuest/install
make -j
```

`CUEST_INSTALL_DIR` must point to the cuEST installation prefix, which is
expected to contain:

```
<CUEST_INSTALL_DIR>/
  include/          # cuEST headers
  lib/<cuda_major>/ # libcuest.so  (e.g. lib/12/libcuest.so)
```

### Optional CMake variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUEST_INSTALL_DIR` | *(required)* | Path to cuEST installation |
| `CMAKE_CUDA_ARCHITECTURES` | `80 90` | Target GPU architectures |

Example with custom GPU architectures:

```bash
cmake .. -DCUEST_INSTALL_DIR=/opt/cuest -DCMAKE_CUDA_ARCHITECTURES="80;90;100"
```

## Directory Structure

```
c_examples/
  common/           # Shared helper headers (parsers, workspace utilities, etc.)
  examples/
    0_context/                   # Handle creation and multi-GPU/stream usage
    1_basic_data_structures/     # AO basis, shells, XC grid setup
    2_one_electron_integrals/    # Overlap, kinetic, potential integrals
    3_density_fitting/           # DF-J and DF-K integrals and gradients
    4_exchange_correlation/      # Local and nonlocal XC potential and gradients
    5_effective_core_potentials/ # ECP integrals and gradients
```

## Running an Example

After building, executables are placed in the `build/` directory tree mirroring
the source layout. For example:

```bash
./build/basic_usage
./build/one_electron_integrals
```
