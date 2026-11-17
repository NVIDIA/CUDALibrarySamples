# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched real 1D linear convolution with cuTile Python and the cuFFTDx tilelib.

Each CTA convolves a batch of signal rows with the matching filter rows. The batch per CTA
and the CTA size come from nvmath.tile.fft.suggested_tile_config for the device.
"""

import math

import cuda.tile as ct
import cupy as cp
import nvmath.tile.fft as mfft

# N is larger than SIGNAL_SIZE + FILTER_SIZE - 1, so zero padding makes the
# FFT convolution linear rather than circular.
N = 16
SIGNAL_SIZE = 8
FILTER_SIZE = 5
OUTPUT_SIZE = SIGNAL_SIZE + FILTER_SIZE - 1
# Rows of signals and filters; a multiple of every suggested batch per CTA.
BATCH = 1024


def make_fft_convolution_kernel(batches_per_block, num_warps):
    """Build the kernel for one (batches_per_block, N) tile per CTA."""

    @ct.kernel(num_worker_warps=num_warps)
    def fft_convolution_kernel(signals: ct.Array, filters: ct.Array, outputs: ct.Array):
        """Compute irfft(rfft(signal) * rfft(filter)) / N for this CTA's rows."""
        rows = (ct.bid(0), 0)
        signal_tile = ct.load(signals, rows, shape=(batches_per_block, N))
        filter_tile = ct.load(filters, rows, shape=(batches_per_block, N))

        # The public full layout is packed into the tilelib's folded real layout.
        signal_spectrum = mfft.rfft(signal_tile, axis=1, complex_layout="full")
        filter_spectrum = mfft.rfft(filter_tile, axis=1, complex_layout="full")

        # The final dimension stores each complex value as (real, imaginary).
        scalar_shape = (batches_per_block, N, 1)
        signal_real = ct.extract(signal_spectrum, (0, 0, 0), shape=scalar_shape)
        signal_imag = ct.extract(signal_spectrum, (0, 0, 1), shape=scalar_shape)
        filter_real = ct.extract(filter_spectrum, (0, 0, 0), shape=scalar_shape)
        filter_imag = ct.extract(filter_spectrum, (0, 0, 1), shape=scalar_shape)
        product_real = signal_real * filter_real - signal_imag * filter_imag
        product_imag = signal_real * filter_imag + signal_imag * filter_real
        product = ct.cat((product_real, product_imag), 2)

        ct.store(
            outputs,
            rows,
            mfft.irfft(product, axis=1, complex_layout="full") * (1.0 / N),
        )

    return fft_convolution_kernel


def direct_convolution(signal_row, filter_row):
    """CPU reference implementation for the unpadded linear convolution."""
    result = [0.0] * (len(signal_row) + len(filter_row) - 1)
    for i, signal_value in enumerate(signal_row):
        for j, filter_value in enumerate(filter_row):
            result[i + j] += signal_value * filter_value
    return result


def main() -> None:
    signal_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    filter_values = [0.5, -1.0, 0.25, 2.0, -0.5]
    expected = direct_convolution(signal_values, filter_values)

    # Row r holds the signal scaled by r + 1 and the same filter, so its convolution is the
    # reference scaled by r + 1.
    scale = cp.arange(1, BATCH + 1, dtype=cp.float32)[:, None]
    signals = cp.zeros((BATCH, N), dtype=cp.float32)
    filters = cp.zeros((BATCH, N), dtype=cp.float32)
    outputs = cp.zeros_like(signals)
    signals[:, :SIGNAL_SIZE] = scale * cp.asarray(signal_values, dtype=cp.float32)
    filters[:, :FILTER_SIZE] = cp.asarray(filter_values, dtype=cp.float32)

    # The three transforms share one (batch, N) tile, so one entry configures the kernel:
    # the inverse transform's, measured closest to the fused kernel's optimum. A device
    # without an entry runs one batch per CTA on one warp.
    arch = f"sm_{cp.cuda.Device().compute_capability}"
    config = mfft.suggested_tile_config("irfft", 1, 0, N, ct.float32, arch)
    batches_per_block, num_warps = config.batches_per_block, config.num_worker_warps
    if config.found:
        print(f"{arch}: {batches_per_block} batches per CTA, {num_warps} warp(s)")
    else:
        print(f"no suggested tile config for N = {N} on {arch}; using 1 batch per CTA")

    ct.launch(
        cp.cuda.get_current_stream(),
        (BATCH // batches_per_block, 1, 1),
        make_fft_convolution_kernel(batches_per_block, num_warps),
        (signals, filters, outputs),
    )
    cp.cuda.runtime.deviceSynchronize()

    outputs_host = outputs.get()
    expected_rows = (
        cp.asarray(expected + [0.0] * (N - OUTPUT_SIZE), dtype=cp.float32)[None, :]
        * scale
    ).get()
    max_error = 0.0
    for r in range(BATCH):
        for i in range(N):
            real_error = abs(float(outputs_host[r, i]) - float(expected_rows[r, i]))
            if not math.isfinite(real_error):
                raise SystemExit(
                    f"cuTile convolution produced a non-finite value at row {r}, index {i}"
                )
            max_error = max(max_error, real_error / float(r + 1))

    print(f"1D linear convolution, row 0 (first {OUTPUT_SIZE} values):")
    for i, expected_real in enumerate(expected):
        print(f"y[{i}] = {outputs_host[0, i]:8.5f}, expected {expected_real:8.5f}")
    print(f"maximum absolute error over {BATCH} rows (per unit scale): {max_error:.6g}")
    if not math.isfinite(max_error) or max_error >= 2e-3:
        raise SystemExit("cuTile convolution does not match the CPU reference")


if __name__ == "__main__":
    main()
