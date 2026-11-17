# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""128^3 three-kernel cuTile convolution with CuPy validation."""

import math

import cuda.tile as ct
import cupy as cp
import nvmath.tile.fft as mfft

# These extents exactly match the C++ split-kernel example. A 64^3 signal and
# 65^3 filter fit their full linear convolution in the 128^3 FFT domain.
N = 128
SIGNAL_EXTENT = 64
FILTER_EXTENT = 65
COMPLEX_SHAPE = (N, N, N, 2)
PLANE_SHAPE = (N, N, 1, 2)
LINE_SHAPE = (1, 1, N, 2)


@ct.kernel(num_worker_warps=1)
def forward_axes_01_kernel(
    signal: ct.Array, filter_: ct.Array, signal_stage: ct.Array, filter_stage: ct.Array
):
    """Transform the first two axes for one fixed axis-2 coordinate."""
    k = ct.bid(0)
    signal_plane = ct.reshape(
        ct.load(signal, (0, 0, k, 0), shape=PLANE_SHAPE), (N, N, 2)
    )
    filter_plane = ct.reshape(
        ct.load(filter_, (0, 0, k, 0), shape=PLANE_SHAPE), (N, N, 2)
    )

    signal_plane = mfft.fft(mfft.fft(signal_plane, axis=0), axis=1)
    filter_plane = mfft.fft(mfft.fft(filter_plane, axis=0), axis=1)
    ct.store(signal_stage, (0, 0, k, 0), ct.reshape(signal_plane, PLANE_SHAPE))
    ct.store(filter_stage, (0, 0, k, 0), ct.reshape(filter_plane, PLANE_SHAPE))


@ct.kernel(num_worker_warps=1)
def convolve_axis_2_kernel(
    signal_stage: ct.Array, filter_stage: ct.Array, result_stage: ct.Array
):
    """Transform, convolve, and invert the contiguous third axis."""
    i = ct.bid(0)
    j = ct.bid(1)
    signal_line = ct.reshape(
        ct.load(signal_stage, (i, j, 0, 0), shape=LINE_SHAPE), (N, 2)
    )
    filter_line = ct.reshape(
        ct.load(filter_stage, (i, j, 0, 0), shape=LINE_SHAPE), (N, 2)
    )

    signal_spectrum = mfft.fft(signal_line, axis=0)
    filter_spectrum = mfft.fft(filter_line, axis=0)
    signal_real = ct.extract(signal_spectrum, (0, 0), shape=(N, 1))
    signal_imag = ct.extract(signal_spectrum, (0, 1), shape=(N, 1))
    filter_real = ct.extract(filter_spectrum, (0, 0), shape=(N, 1))
    filter_imag = ct.extract(filter_spectrum, (0, 1), shape=(N, 1))
    product = ct.cat(
        (
            signal_real * filter_real - signal_imag * filter_imag,
            signal_real * filter_imag + signal_imag * filter_real,
        ),
        1,
    )
    ct.store(
        result_stage, (i, j, 0, 0), ct.reshape(mfft.ifft(product, axis=0), LINE_SHAPE)
    )


@ct.kernel(num_worker_warps=1)
def inverse_axes_10_kernel(result_stage: ct.Array, output: ct.Array):
    """Finish the two inverse plane transforms and apply 1 / N^3 scaling."""
    k = ct.bid(0)
    result_plane = ct.reshape(
        ct.load(result_stage, (0, 0, k, 0), shape=PLANE_SHAPE), (N, N, 2)
    )
    result_plane = mfft.ifft(mfft.ifft(result_plane, axis=1), axis=0)
    ct.store(output, (0, 0, k, 0), ct.reshape(result_plane / (N * N * N), PLANE_SHAPE))


def relative_l2_error(actual: cp.ndarray, expected: cp.ndarray) -> float:
    """Return ||actual - expected||_2 / ||expected||_2."""
    return float(
        cp.linalg.norm((actual - expected).reshape(-1))
        / cp.linalg.norm(expected.reshape(-1))
    )


def main() -> None:
    signal = cp.zeros(COMPLEX_SHAPE, dtype=cp.float32)
    filter_ = cp.zeros_like(signal)
    signal_stage = cp.zeros_like(signal)
    filter_stage = cp.zeros_like(signal)
    result_stage = cp.zeros_like(signal)
    output = cp.zeros_like(signal)

    i, j, k = cp.meshgrid(
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        cp.arange(SIGNAL_EXTENT, dtype=cp.float32),
        indexing="ij",
    )
    signal[:SIGNAL_EXTENT, :SIGNAL_EXTENT, :SIGNAL_EXTENT, 0] = (
        0.03125 * (i + 1.0) * (j + 2.0) - 0.015625 * k
    )

    i, j, k = cp.meshgrid(
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        cp.arange(FILTER_EXTENT, dtype=cp.float32),
        indexing="ij",
    )
    filter_[:FILTER_EXTENT, :FILTER_EXTENT, :FILTER_EXTENT, 0] = (
        0.0078125 * (i + 1.0) - 0.00390625 * j + 0.001953125 * k
    )

    stream = cp.cuda.get_current_stream()
    ct.launch(
        stream,
        (N, 1, 1),
        forward_axes_01_kernel,
        (signal, filter_, signal_stage, filter_stage),
    )
    ct.launch(
        stream,
        (N, N, 1),
        convolve_axis_2_kernel,
        (signal_stage, filter_stage, result_stage),
    )
    ct.launch(stream, (N, 1, 1), inverse_axes_10_kernel, (result_stage, output))
    cp.cuda.runtime.deviceSynchronize()

    signal_complex = cp.ascontiguousarray(signal).view(cp.complex64)[..., 0]
    filter_complex = cp.ascontiguousarray(filter_).view(cp.complex64)[..., 0]
    expected = cp.fft.ifftn(cp.fft.fftn(signal_complex) * cp.fft.fftn(filter_complex))
    actual = cp.ascontiguousarray(output).view(cp.complex64)[..., 0]
    error = relative_l2_error(actual, expected)
    if not math.isfinite(error) or error >= 1e-5:
        raise RuntimeError(
            f"split 3D convolution relative L2 error is too large: {error:.6g}"
        )
    print(f"128x128x128 split 3D convolution; relative L2 error vs. CuPy: {error:.6g}")


if __name__ == "__main__":
    main()
