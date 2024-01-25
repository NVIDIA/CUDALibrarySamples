# Auxiliary functions for samples

This folder contains a few auxiliary functions for various samples.

## error_checks.hpp
- error_check: Compute the global L2 norm between reference and test values by using `BoxIterator`.
- assess_error: Assess the error based on some tolerance (default: `tolerance = 1e-6`). This also produces a print statement on the MPI rank 0.

## generate_random.hpp
- Two generate_random functions that generate real or complex values in a `std::vector`

## scaling.cuh
- scaling_kernel: Normalize entries in the box with a constant scaling factor using `BoxIterator`. By default, entries corresponding to the first 10 threads are printed for illustration. This kernel serves as an example of intermediate operations that can be done between two Fourier transforms.