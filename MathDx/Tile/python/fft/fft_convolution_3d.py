# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D 8x8x8 linear convolution with cuTile Python and CuPy validation."""

import cuda.tile as ct
import cupy as cp
import nvmath.tile.fft as mfft

# These match the C++ example exactly. Padding a 4^3 signal and 5^3 filter
# to 8^3 makes their linear convolution fit without circular wraparound.
N = 8
SIGNAL_EXTENT = 4
FILTER_EXTENT = 5
COMPLEX_SHAPE = (N, N, N, 2)


@ct.kernel(num_worker_warps=1)
def fft_convolution_3d_kernel(signal: ct.Array, filter_: ct.Array, output: ct.Array):
    signal_spectrum = mfft.fft(
        ct.load(signal, (0, 0, 0, 0), shape=COMPLEX_SHAPE), axis=0
    )
    signal_spectrum = mfft.fft(signal_spectrum, axis=1)
    signal_spectrum = mfft.fft(signal_spectrum, axis=2)
    filter_spectrum = mfft.fft(
        ct.load(filter_, (0, 0, 0, 0), shape=COMPLEX_SHAPE), axis=0
    )
    filter_spectrum = mfft.fft(filter_spectrum, axis=1)
    filter_spectrum = mfft.fft(filter_spectrum, axis=2)

    # The trailing dimension holds each complex value as (real, imaginary).
    scalar_shape = (N, N, N, 1)
    signal_real = ct.extract(signal_spectrum, (0, 0, 0, 0), shape=scalar_shape)
    signal_imag = ct.extract(signal_spectrum, (0, 0, 0, 1), shape=scalar_shape)
    filter_real = ct.extract(filter_spectrum, (0, 0, 0, 0), shape=scalar_shape)
    filter_imag = ct.extract(filter_spectrum, (0, 0, 0, 1), shape=scalar_shape)
    product = ct.cat(
        (
            signal_real * filter_real - signal_imag * filter_imag,
            signal_real * filter_imag + signal_imag * filter_real,
        ),
        3,
    )

    result = mfft.ifft(product, axis=0)
    result = mfft.ifft(result, axis=1)
    result = mfft.ifft(result, axis=2)
    ct.store(output, (0, 0, 0, 0), result * (1.0 / (N * N * N)))


def main() -> None:
    signal = cp.zeros(COMPLEX_SHAPE, dtype=cp.float32)
    filter_ = cp.zeros_like(signal)
    output = cp.zeros_like(signal)

    # Use the same deterministic real-valued signal and filter as the C++
    # example. Imaginary components remain zero.
    i, j, k = cp.meshgrid(
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        indexing="ij",
    )
    signal[:SIGNAL_EXTENT, :SIGNAL_EXTENT, :SIGNAL_EXTENT, 0] = (i + 1.0) * (
        j + 2.0
    ) - 0.5 * k

    i, j, k = cp.meshgrid(
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        indexing="ij",
    )
    filter_[:FILTER_EXTENT, :FILTER_EXTENT, :FILTER_EXTENT, 0] = (
        0.25 * (i + 1.0) - 0.125 * j + 0.0625 * k
    )

    ct.launch(
        cp.cuda.get_current_stream(),
        (1, 1, 1),
        fft_convolution_3d_kernel,
        (signal, filter_, output),
    )
    cp.cuda.runtime.deviceSynchronize()

    # CuPy is the sole numerical reference, as in the C++ example's cuFFT
    # check. Both use the same padded 8^3 complex storage.
    signal_complex = cp.ascontiguousarray(signal).view(cp.complex64)[..., 0]
    filter_complex = cp.ascontiguousarray(filter_).view(cp.complex64)[..., 0]
    expected = cp.fft.ifftn(cp.fft.fftn(signal_complex) * cp.fft.fftn(filter_complex))
    actual = cp.ascontiguousarray(output).view(cp.complex64)[..., 0]
    cp.testing.assert_allclose(actual, expected, rtol=1e-3, atol=3e-3)

    max_error = float(cp.abs(actual - expected).max())
    print(f"8x8x8 3D linear convolution passed; maximum CuPy error: {max_error:.6g}")


if __name__ == "__main__":
    main()
