# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Float I/O with a double-precision cuTile FFT and CuPy validation."""

import cuda.tile as ct
import cupy as cp
import nvmath.tile.fft as mfft

N = 16
COMPLEX_SHAPE = (N, 2)


@ct.kernel(num_worker_warps=1)
def mixed_precision_fft_kernel(input_: ct.Array, output: ct.Array):
    """Load float values, execute the FFT in double, and store float values."""
    input_tile = ct.load(input_, (0, 0), shape=COMPLEX_SHAPE)
    double_spectrum = mfft.fft(ct.astype(input_tile, ct.float64), axis=0)
    ct.store(output, (0, 0), ct.astype(double_spectrum, ct.float32))


def relative_l2_error(actual: cp.ndarray, expected: cp.ndarray) -> float:
    """Return ||actual - expected||_2 / ||expected||_2."""
    return float(
        cp.linalg.norm((actual - expected).reshape(-1))
        / cp.linalg.norm(expected.reshape(-1))
    )


def main() -> None:
    input_ = cp.zeros(COMPLEX_SHAPE, dtype=cp.float32)
    output = cp.zeros_like(input_)
    index = cp.arange(N, dtype=cp.float32)
    input_[:, 0] = 1.0 + index / 8.0
    input_[:, 1] = 0.125 * (cp.remainder(index, 5.0) - 2.0)

    ct.launch(
        cp.cuda.get_current_stream(),
        (1, 1, 1),
        mixed_precision_fft_kernel,
        (input_, output),
    )
    cp.cuda.runtime.deviceSynchronize()

    # The C++ check uses double cuFFT. CuPy's complex128 FFT is the equivalent
    # reference, while the Tile output remains float by design.
    expected = cp.fft.fft(
        cp.ascontiguousarray(input_).view(cp.complex64)[..., 0].astype(cp.complex128)
    )
    actual = (
        cp.ascontiguousarray(output).view(cp.complex64)[..., 0].astype(cp.complex128)
    )
    cp.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-5)

    error = relative_l2_error(actual, expected)
    print(f"float I/O, double FFT; relative L2 error vs. double CuPy FFT: {error:.6g}")


if __name__ == "__main__":
    main()
