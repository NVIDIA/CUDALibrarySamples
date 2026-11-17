# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""5D 8^5 linear convolution with cuTile Python and CuPy validation."""

import math

import cuda.tile as ct
import cupy as cp
import nvmath.tile.fft as mfft

# Padding a 4^5 signal and 5^5 filter to 8^5 makes their full linear
# convolution fit without circular wraparound. The logical tensor is
# [i,j,k,l,m,(real,imaginary)]: m is the fastest spatial dimension and the
# final dimension stores each complex value as two adjacent floats.
#
# A cuTile FFT can transform at most three axes. We therefore factor the 5D
# DFT into 64 strided 3D (i,j,k) transforms, then 512 contiguous 2D (l,m)
# transforms. The inverse performs those groups in reverse order; reshape only
# changes the tile view and does not copy global-memory data.
N = 8
SIGNAL_EXTENT = 4
FILTER_EXTENT = 5
COMPLEX_SHAPE = (N, N, N, N, N, 2)
AXES_012_SLICE_SHAPE = (N, N, N, 1, 1, 2)
AXES_012_FFT_SHAPE = (N, N, N, 2)
AXES_34_SLICE_SHAPE = (1, 1, 1, N, N, 2)
AXES_34_FFT_SHAPE = (N, N, 2)


@ct.kernel(num_worker_warps=1)
def forward_axes_012_kernel(
    signal: ct.Array, filter_: ct.Array, signal_stage: ct.Array, filter_stage: ct.Array
):
    axis_l = ct.bid(0)
    axis_m = ct.bid(1)

    # This block owns one fixed (l,m) pair and loads its strided (i,j,k) slice.
    # Reshape exposes it as the 3D tile used for axes 0, 1, and 2.
    signal_tile = ct.reshape(
        ct.load(signal, (0, 0, 0, axis_l, axis_m, 0), shape=AXES_012_SLICE_SHAPE),
        AXES_012_FFT_SHAPE,
    )
    filter_tile = ct.reshape(
        ct.load(filter_, (0, 0, 0, axis_l, axis_m, 0), shape=AXES_012_SLICE_SHAPE),
        AXES_012_FFT_SHAPE,
    )
    signal_tile = mfft.fft(mfft.fft(mfft.fft(signal_tile, axis=0), axis=1), axis=2)
    filter_tile = mfft.fft(mfft.fft(mfft.fft(filter_tile, axis=0), axis=1), axis=2)
    ct.store(
        signal_stage,
        (0, 0, 0, axis_l, axis_m, 0),
        ct.reshape(signal_tile, AXES_012_SLICE_SHAPE),
    )
    ct.store(
        filter_stage,
        (0, 0, 0, axis_l, axis_m, 0),
        ct.reshape(filter_tile, AXES_012_SLICE_SHAPE),
    )


@ct.kernel(num_worker_warps=1)
def convolve_axes_34_kernel(
    signal_stage: ct.Array, filter_stage: ct.Array, result_stage: ct.Array
):
    i = ct.bid(0)
    j = ct.bid(1)
    k = ct.bid(2)

    # This block owns one fixed (i,j,k) triple and loads its contiguous (l,m)
    # plane. Together with the first kernel's strided slices, this covers all
    # five axes of the original tensor.
    signal_tile = ct.reshape(
        ct.load(signal_stage, (i, j, k, 0, 0, 0), shape=AXES_34_SLICE_SHAPE),
        AXES_34_FFT_SHAPE,
    )
    filter_tile = ct.reshape(
        ct.load(filter_stage, (i, j, k, 0, 0, 0), shape=AXES_34_SLICE_SHAPE),
        AXES_34_FFT_SHAPE,
    )
    signal_tile = mfft.fft(mfft.fft(signal_tile, axis=0), axis=1)
    filter_tile = mfft.fft(mfft.fft(filter_tile, axis=0), axis=1)

    scalar_shape = (N, N, 1)
    signal_real = ct.extract(signal_tile, (0, 0, 0), shape=scalar_shape)
    signal_imag = ct.extract(signal_tile, (0, 0, 1), shape=scalar_shape)
    filter_real = ct.extract(filter_tile, (0, 0, 0), shape=scalar_shape)
    filter_imag = ct.extract(filter_tile, (0, 0, 1), shape=scalar_shape)
    result = ct.cat(
        (
            signal_real * filter_real - signal_imag * filter_imag,
            signal_real * filter_imag + signal_imag * filter_real,
        ),
        2,
    )
    result = mfft.ifft(mfft.ifft(result, axis=1), axis=0)
    ct.store(result_stage, (i, j, k, 0, 0, 0), ct.reshape(result, AXES_34_SLICE_SHAPE))


@ct.kernel(num_worker_warps=1)
def inverse_axes_210_kernel(result_stage: ct.Array, output: ct.Array):
    axis_l = ct.bid(0)
    axis_m = ct.bid(1)
    # Return to the fixed-(l,m), strided (i,j,k) geometry. N**5 normalizes
    # all five unnormalized inverse FFTs.
    result = ct.reshape(
        ct.load(result_stage, (0, 0, 0, axis_l, axis_m, 0), shape=AXES_012_SLICE_SHAPE),
        AXES_012_FFT_SHAPE,
    )
    result = mfft.ifft(mfft.ifft(mfft.ifft(result, axis=2), axis=1), axis=0)
    ct.store(
        output,
        (0, 0, 0, axis_l, axis_m, 0),
        ct.reshape(result / (N**5), AXES_012_SLICE_SHAPE),
    )


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

    signal_axes = cp.meshgrid(
        *(cp.arange(SIGNAL_EXTENT, dtype=cp.float32) for _ in range(5)), indexing="ij"
    )
    i, j, k, axis_l, axis_m = signal_axes
    signal[
        :SIGNAL_EXTENT,
        :SIGNAL_EXTENT,
        :SIGNAL_EXTENT,
        :SIGNAL_EXTENT,
        :SIGNAL_EXTENT,
        0,
    ] = i + 1.0 + 0.5 * j - 0.25 * k + 0.125 * axis_l - 0.0625 * axis_m

    filter_axes = cp.meshgrid(
        *(cp.arange(FILTER_EXTENT, dtype=cp.float32) for _ in range(5)), indexing="ij"
    )
    i, j, k, axis_l, axis_m = filter_axes
    filter_[
        :FILTER_EXTENT,
        :FILTER_EXTENT,
        :FILTER_EXTENT,
        :FILTER_EXTENT,
        :FILTER_EXTENT,
        0,
    ] = 0.25 * (i + 1.0) - 0.125 * j + 0.0625 * k - 0.03125 * axis_l + 0.015625 * axis_m

    stream = cp.cuda.get_current_stream()
    ct.launch(
        stream,
        (N, N, 1),
        forward_axes_012_kernel,
        (signal, filter_, signal_stage, filter_stage),
    )
    ct.launch(
        stream,
        (N, N, N),
        convolve_axes_34_kernel,
        (signal_stage, filter_stage, result_stage),
    )
    ct.launch(stream, (N, N, 1), inverse_axes_210_kernel, (result_stage, output))
    cp.cuda.runtime.deviceSynchronize()

    signal_complex = cp.ascontiguousarray(signal).view(cp.complex64)[..., 0]
    filter_complex = cp.ascontiguousarray(filter_).view(cp.complex64)[..., 0]
    expected = cp.fft.ifftn(cp.fft.fftn(signal_complex) * cp.fft.fftn(filter_complex))
    actual = cp.ascontiguousarray(output).view(cp.complex64)[..., 0]
    error = relative_l2_error(actual, expected)
    if not math.isfinite(error) or error >= 1e-5:
        raise RuntimeError(
            f"5D convolution relative L2 error is too large: {error:.6g}"
        )
    print(f"8^5 convolution; relative L2 error vs. CuPy: {error:.6g}")


if __name__ == "__main__":
    main()
