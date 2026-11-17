#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autotune a batched 1D FFT over the batch extent, the CTA size and occupancy.

The three knobs interact, so the fastest combination is not generally the one
that is fastest for any of them alone. Only the first changes the kernel:

* ``batches_per_block`` is the tile's leading extent. It is part of the tile
  shape, so it is a compile-time constant and each value needs its own kernel.
* ``num_worker_warps`` and ``occupancy`` are compiler hints, so
  ``cuda.tile.tune`` varies them per candidate through ``hints_fn`` without
  rebuilding anything.

``cuda.tile.tune.exhaustive_search`` compiles and times every configuration.
It reports configurations that fail to resolve and continues the search. It returns the fastest configuration.
The total transform count is fixed. The grid shrinks as the batch extent grows.
Every configuration performs the same work, so the times are comparable.

The example checks the winner against ``cupy.fft`` because the tuner ranks by time without checking correctness.
"""

import math
from itertools import product
from typing import Annotated

import cuda.tile as ct
import cupy as cp
import nvmath.tile.fft as mfft
from cuda.tile import tune

N = 1024

# Transform a 512 MiB buffer, the operand the shipped configurations were tuned
# against. A smaller buffer does not saturate memory bandwidth on a large GPU,
# and then measures a different regime from the one the suggestion targets.
BUFFER_BYTES = 512 * 1024 * 1024
TOTAL_TRANSFORMS = BUFFER_BYTES // (N * 2 * 4)
TOLERANCE = 2e-5

# Two values of each knob keeps the sweep quick. Widen them for a real tuning
# run; every batch extent compiles its own kernel, so that is the costly axis.
BATCH_EXTENTS = (1, 4)
WORKER_WARPS = (1, 4)
OCCUPANCIES = (None, 2)


# Declaring every extent and stride static lets the tilelib see a fully static
# demand, so the loads and stores lower to fixed offsets instead of computing
# them at run time. See the Static Shapes section of the performance guide.
StaticArray = Annotated[
    ct.Array,
    ct.ArrayAnnotation(static_shape_dims=(0, 1, 2), static_stride_dims=(0, 1, 2)),
]


def make_kernel(batches_per_block):
    """One kernel per batch extent, which is the only knob in the tile shape."""
    tile_shape = (batches_per_block, N, 2)

    @ct.kernel()
    def fft_kernel(source: StaticArray, destination: StaticArray):
        # The index is in tile space, so it is not scaled by the tile extent here.
        block = ct.bid(0)
        data = ct.load(source, (block, 0, 0), shape=tile_shape)
        ct.store(destination, (block, 0, 0), mfft.fft(data, axis=1))

    return fft_kernel


def main():
    stream = cp.cuda.get_current_stream()
    source = cp.random.randn(TOTAL_TRANSFORMS, N, 2).astype(cp.float32)
    destination = cp.zeros_like(source)

    # Keep the reference in the tiles' own interleaved (real, imaginary) layout,
    # so the comparison needs no conversion.
    spectrum = cp.fft.fft(source[..., 0] + 1j * source[..., 1], axis=1)
    reference = cp.stack((spectrum.real, spectrum.imag), axis=-1).astype(cp.float32)

    arch = f"sm_{cp.cuda.Device().compute_capability}"
    suggested = mfft.suggested_tile_config(
        "fft", rank=2, axis=1, size=N, precision=ct.float32, arch=arch
    )
    print(f"{arch}: N={N}, {TOTAL_TRANSFORMS} transforms")
    if suggested.found:
        print(
            f"suggested: {suggested.batches_per_block} batches per block, "
            f"{suggested.num_worker_warps} warp(s)\n"
        )
    else:
        print("suggested: none for this size on this architecture\n")

    # One kernel per batch extent, reused across every width and occupancy.
    kernels = {
        batches_per_block: make_kernel(batches_per_block)
        for batches_per_block in BATCH_EXTENTS
        if TOTAL_TRANSFORMS % batches_per_block == 0
    }
    search_space = list(product(kernels, WORKER_WARPS, OCCUPANCIES))

    def hints_fn(config):
        _, num_worker_warps, occupancy = config
        hints = {"num_worker_warps": num_worker_warps}
        if occupancy is not None:
            hints["occupancy"] = occupancy
        return hints

    result = tune.exhaustive_search(
        search_space,
        stream,
        grid_fn=lambda config: (TOTAL_TRANSFORMS // config[0], 1, 1),
        kernel=lambda config: kernels[config[0]],
        args_fn=lambda config: (source, destination),
        hints_fn=hints_fn,
        quiet=True,
    )
    print(result.summary())

    best = result.best
    batches_per_block, num_worker_warps, occupancy = best.config
    print(
        f"\nfastest: batches_per_block={batches_per_block} "
        f"num_worker_warps={num_worker_warps} "
        f"occupancy={'auto' if occupancy is None else occupancy} "
        f"at {best.mean_us:.1f} us"
    )

    # A tuner ranks by time, so confirm the winner actually computes the FFT.
    # The hints have to be reapplied: the tuner passed them through hints_fn, so
    # the kernel alone is a different kernel from the one that won.
    destination.fill(0)
    winner = kernels[batches_per_block].replace_hints(**hints_fn(best.config))
    ct.launch(stream, (TOTAL_TRANSFORMS // batches_per_block, 1, 1), winner, (source, destination))
    cp.cuda.runtime.deviceSynchronize()

    error = float(cp.linalg.norm(reference - destination) / cp.linalg.norm(reference))
    print(f"relative error of the fastest configuration: {error:.2e}")
    if not math.isfinite(error) or error >= TOLERANCE:
        raise SystemExit(
            f"the fastest configuration does not match cupy.fft: "
            f"relative error {error:.2e}"
        )


if __name__ == "__main__":
    main()
