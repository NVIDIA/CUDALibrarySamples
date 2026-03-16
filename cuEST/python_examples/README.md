# cuEST Python API Examples

This directory contains standalone Python examples demonstrating the cuEST Python API.

For full API documentation, see https://docs.nvidia.com/cuda/cuest/.

## Prerequisites

- CUDA Toolkit 12 or later
- Python 3.10 or later
- NumPy

## Installation

From the `cuda_samples/` directory:

```bash
pip install "..[cu12]"  # or "..[cu13]" for CUDA 13
```

This installs:
- `helpers` — shared utilities used by the examples (parsers, CUDA memory helpers, etc.)
- `cuest_scf` — the cuEST SCF Python library
- `nvidia-cuest-cu12` (or `nvidia-cuest-cu13`)

## Running Examples

Each example has a `run.py` entry point. Run from the example directory:

```bash
cd 2_one_electron_integrals/one_electron_integrals
python run.py /path/to/h2o.xyz /path/to/def2-svp.gbs
```

Or use the provided script to run all examples at once:

```bash
./run_all.sh
```

Data files (geometries and basis sets) are located in `../data/`.

## Directory Structure

```
python_examples/
  helpers/                        # Shared utility package
    cuda_utils.py                 # CUDA memory allocation helpers
    parsers.py                    # XYZ and GBS file parsers
    utilities.py                  # Shell normalization and other utilities
    grid_utils.py                 # Grid utility helpers
  0_context/                      # Handle creation and multi-stream usage
  1_basic_data_structures/        # AO basis and shell setup
  2_one_electron_integrals/       # Overlap, kinetic, potential integrals and gradients
  3_density_fitting/              # DF-J and DF-K integrals and gradients
  4_exchange_correlation/         # Local and nonlocal XC potential and gradients
```
