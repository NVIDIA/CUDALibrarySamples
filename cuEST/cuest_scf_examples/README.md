# cuEST SCF Examples

This directory contains the cuEST SCF Python library (`cuest_scf`),
integration tests, and standalone SCF examples.

For full API documentation, see https://docs.nvidia.com/cuda/cuest/.

## Prerequisites

- CUDA Toolkit 12 or later
- Python 3.10 or later
- pytest

## Installation

From the `cuda_samples/` directory:

```bash
pip install "..[cu12]"  # or "..[cu13]" for CUDA 13
```

This installs `cuest_scf` and `nvidia-cuest-cu12` (or `nvidia-cuest-cu13`).

## Running Examples and Tests

```bash
./run_all.sh
```

Or run individual tests with pytest:

```bash
pytest test/rhf_1/test.py -v
```

Or run a standalone example directly:

```bash
python examples/rhf-1/test.py
```

## Directory Structure

```
cuest_scf_examples/
  cuest_scf/          # The cuEST SCF Python library
  examples/
    rhf-1/            # Standalone RHF example
  test/
    rhf_1/            # RHF integration test
    rhf_grad_1/       # RHF gradient integration test
    rhf_pcm/          # RHF with PCM solvation
    b3lyp1_grad_1/    # B3LYP gradient integration test
    b97mv_grad_1/     # B97M-V gradient integration test
    blyp_grad_grids_1/ # BLYP gradient with custom grids
```
