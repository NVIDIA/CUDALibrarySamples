# cuEST SCF Examples

This directory contains the cuEST SCF Python library (`cuest_scf`),
integration tests, and standalone SCF examples.

For full API documentation, see https://docs.nvidia.com/cuda/cuest/.

## Prerequisites

- CUDA Toolkit 12 or later
- Python 3.12 or later
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
  data/
    gbs/              # Gaussian basis set files (G94/Psi4 GBS format)
  examples/
    rhf-1/            # Standalone RHF example
    uhf-1/            # Standalone UHF example
    mp2-1/            # Standalone MP2 example (uses nvmath-python)
    cphf-1/           # Standalone CPHF example (uses nvmath-python)
  test/
    rhf_1/            # RHF integration test
    rhf_grad_1/       # RHF gradient integration test
    rhf_pcm/          # RHF with PCM solvation
    rhf_polarizability_1/ # RHF polarizability via CPHF (uses nvmath-python)
    uhf_1/            # UHF energies, gradients, and PCM tests
    b3lyp1_grad_1/    # B3LYP gradient integration test
    b97mv_grad_1/     # B97M-V gradient integration test
    blyp_grad_grids_1/ # BLYP gradient with custom grids
    dft_energies/     # DFT energy and gradient sweeps across functionals
    ecp_1/            # RHF/UHF with effective core potentials (+PCM)
    mp2_1/            # DF-MP2 test (uses nvmath-python)
  run_all.sh          # Run all tests and examples
```
