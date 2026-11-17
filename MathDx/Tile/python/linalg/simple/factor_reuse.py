# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse Cholesky and pivoted-LU factors in separate CUDA Tile kernels."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import (
    FillMode,
    cholesky_factor,
    cholesky_solve_using_factor,
    lu_factor,
    lu_solve_using_factor,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime, relative_error

N, NRHS = 32, 4


@ct.kernel
def cholesky_factor_kernel(a_data: ct.Array, info_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    factors, info = cholesky_factor(a, FillMode.LOWER)
    ct.store(a_data, (0, 0), factors)
    ct.store(info_data, (0,), info)


@ct.kernel
def cholesky_solve_kernel(a_data: ct.Array, b_data: ct.Array, x_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    b = ct.load(b_data, (0, 0), shape=(N, NRHS))
    x = cholesky_solve_using_factor(a, b, FillMode.LOWER)
    ct.store(x_data, (0, 0), x)


@ct.kernel
def lu_factor_kernel(a_data: ct.Array, pivots_data: ct.Array, info_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    factors, pivots, info = lu_factor(a)
    ct.store(a_data, (0, 0), factors)
    ct.store(pivots_data, (0,), pivots)
    ct.store(info_data, (0,), info)


@ct.kernel
def lu_solve_kernel(
    a_data: ct.Array, pivots_data: ct.Array, b_data: ct.Array, x_data: ct.Array
):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    b = ct.load(b_data, (0, 0), shape=(N, NRHS))
    pivots = ct.load(pivots_data, (0,), shape=(N,))
    x = lu_solve_using_factor(a, b, pivots)
    ct.store(x_data, (0, 0), x)


def main() -> None:
    rt = Runtime()

    # One dense SPD matrix drives both paths: Cholesky needs the definiteness, LU
    # just needs it to be nonsingular. Solve against a known X.
    expected = np.arange(N * NRHS, dtype=np.float32).reshape(N, NRHS) / 17
    rows, cols = np.arange(N)[:, None], np.arange(N)[None, :]
    lower = np.tril(
        np.where(rows == cols, 2 + rows / N, ((rows + 2 * cols) % 5) / 8 - 0.25)
    ).astype(np.float32)
    original = lower @ lower.T
    info = rt.full((1,), -1, np.int32)
    b = original @ expected
    x = rt.zeros(expected.shape, np.float32)
    a = rt.to_device(original)
    rt.launch(cholesky_factor_kernel, (a, info))
    rt.sync()
    assert int(rt.to_host(info)[0]) == 0
    rt.launch(cholesky_solve_kernel, (a, rt.to_device(b), x))
    rt.sync()
    cholesky_error = relative_error(expected, rt.to_host(x))
    pivots = rt.zeros(N, np.int32)
    rt.copy_from_host(info, np.full((1,), -1, dtype=np.int32))
    rt.copy_from_host(x, np.zeros_like(expected))
    a = rt.to_device(original)
    rt.launch(lu_factor_kernel, (a, pivots, info))
    rt.sync()
    assert int(rt.to_host(info)[0]) == 0
    rt.launch(lu_solve_kernel, (a, pivots, rt.to_device(b), x))
    rt.sync()
    lu_error = relative_error(expected, rt.to_host(x))
    print(f"factor reuse: Cholesky error={cholesky_error:g}, LU error={lu_error:g}")
    assert cholesky_error < 2e-4 and lu_error < 2e-4


if __name__ == "__main__":
    main()
