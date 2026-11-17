# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real 3D convolution using folded rfft/irfft and CuPy validation."""

import cuda.tile as ct
import cupy as cp
import nvmath.tile.fft as mfft

# The same padded 4^3-signal, 5^3-filter, and 8^3-output arrangement used by
# the C++ real convolution example.
N = 8
SIGNAL_EXTENT = 4
FILTER_EXTENT = 5
REAL_SHAPE = (N, N, N)
COMPLEX_SHAPE = (N, N, N, 2)


@ct.kernel(num_worker_warps=1)
def fft_convolution_3d_real_kernel(
    signal: ct.Array, filter_: ct.Array, output: ct.Array
):
    # "full" retains a simple N-element complex spectrum. The tilelib packs
    # it into cuFFTDx's folded real layout internally before the raw R2C call.
    signal_spectrum = mfft.rfft(
        ct.load(signal, (0, 0, 0), shape=REAL_SHAPE), axis=0, complex_layout="full"
    )
    signal_spectrum = mfft.fft(signal_spectrum, axis=1)
    signal_spectrum = mfft.fft(signal_spectrum, axis=2)
    filter_spectrum = mfft.rfft(
        ct.load(filter_, (0, 0, 0), shape=REAL_SHAPE), axis=0, complex_layout="full"
    )
    filter_spectrum = mfft.fft(filter_spectrum, axis=1)
    filter_spectrum = mfft.fft(filter_spectrum, axis=2)

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

    product = mfft.ifft(product, axis=2)
    product = mfft.ifft(product, axis=1)
    ct.store(
        output,
        (0, 0, 0),
        mfft.irfft(product, axis=0, complex_layout="full") * (1.0 / (N * N * N)),
    )


def main() -> None:
    signal = cp.zeros(REAL_SHAPE, dtype=cp.float32)
    filter_ = cp.zeros_like(signal)
    output = cp.zeros_like(signal)

    i, j, k = cp.meshgrid(
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        indexing="ij",
    )
    signal[:SIGNAL_EXTENT, :SIGNAL_EXTENT, :SIGNAL_EXTENT] = (i + 1.0) * (
        j + 2.0
    ) - 0.5 * k

    i, j, k = cp.meshgrid(
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        indexing="ij",
    )
    filter_[:FILTER_EXTENT, :FILTER_EXTENT, :FILTER_EXTENT] = (
        0.25 * (i + 1.0) - 0.125 * j + 0.0625 * k
    )

    ct.launch(
        cp.cuda.get_current_stream(),
        (1, 1, 1),
        fft_convolution_3d_real_kernel,
        (signal, filter_, output),
    )
    cp.cuda.runtime.deviceSynchronize()

    # Mirror the exact transform sequence in CuPy. The cuTile tilelib's
    # folded representation is internal; its public result is the full RFFT.
    signal_spectrum = cp.fft.rfft(signal, axis=0)
    signal_spectrum = cp.fft.fft(signal_spectrum, axis=1)
    signal_spectrum = cp.fft.fft(signal_spectrum, axis=2)
    filter_spectrum = cp.fft.rfft(filter_, axis=0)
    filter_spectrum = cp.fft.fft(filter_spectrum, axis=1)
    filter_spectrum = cp.fft.fft(filter_spectrum, axis=2)
    expected = cp.fft.ifft(signal_spectrum * filter_spectrum, axis=2)
    expected = cp.fft.ifft(expected, axis=1)
    expected = cp.fft.irfft(expected, n=N, axis=0)

    cp.testing.assert_allclose(output, expected, rtol=1e-3, atol=3e-3)
    max_error = float(cp.abs(output - expected).max())
    print(f"8x8x8 real 3D convolution passed; maximum CuPy error: {max_error:.6g}")


if __name__ == "__main__":
    main()
