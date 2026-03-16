# cuEST Examples

This directory contains standalone examples for the cuEST library, covering
the C API, Python API, and the cuEST SCF Python library.

For full API documentation, see https://docs.nvidia.com/cuda/cuest/.

## Prerequisites

- CUDA Toolkit 12 or later
- Python 3.10 or later
- An NVIDIA GPU (Ampere or later recommended)

## Quick Start

### 1. Install Python packages

From this directory:

```bash
pip install ".[cu12]"  # or ".[cu13]" for CUDA 13
```

This installs:
- `helpers` — shared utilities used by the Python API examples
- `cuest_scf` — the cuEST SCF Python library
- `nvidia-cuest-cu12` (or `nvidia-cuest-cu13`)

### 2. Run the examples

**C API examples** — see [`c_examples/README.md`](c_examples/README.md) for build instructions, then:

```bash
cd c_examples
./run_all.sh
```

**Python API examples:**

```bash
cd python_examples
./run_all.sh
```

**cuEST SCF examples and tests:**

```bash
cd cuest_scf_examples
./run_all.sh
```

## Third-Party Notices

See [`THIRD_PARTY_NOTICES.txt`](THIRD_PARTY_NOTICES.txt) for licenses
and attributions for bundled third-party data and software dependencies.

## Input Data

Shared geometry and basis set files used by all examples are in `data/`:

```
data/
  geometry/
    h2o.xyz               # Water molecule geometry
    ch2i2.xyz             # Diiodomethane geometry (ECP examples)
  basis_set/
    def2-svp.gbs          # def2-SVP basis set
    def2-universal-jkfit.gbs  # def2-universal-JKFIT auxiliary basis set
    def2-svp-ecp.gbs      # def2-SVP ECP basis set
```
