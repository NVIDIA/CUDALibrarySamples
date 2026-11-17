# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Positive-definite solve with nvmath.tile.linalg."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import FillMode, cholesky_solve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime, relative_error

N, NRHS = 32, 4


@ct.kernel
def cholesky_solve_kernel(
    a_data: ct.Array, b_data: ct.Array, x_data: ct.Array, info_data: ct.Array
):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    b = ct.load(b_data, (0, 0), shape=(N, NRHS))
    x, info = cholesky_solve(a, b, FillMode.LOWER)
    ct.store(x_data, (0, 0), x)
    ct.store(info_data, (0,), info)


def main() -> None:
    rt = Runtime()

    # Dense symmetric positive-definite A, built as L L^T from a dense
    # lower-triangular factor whose diagonal dominates its off-diagonal entries.
    # Solving against a known X makes the error below measure the solver, not the data.
    rows, cols = np.arange(N)[:, None], np.arange(N)[None, :]
    lower = np.tril(
        np.where(rows == cols, 2 + rows / N, ((rows + 2 * cols) % 5) / 8 - 0.25)
    ).astype(np.float32)
    a = lower @ lower.T
    expected = np.arange(N * NRHS, dtype=np.float32).reshape(N, NRHS) / 11 + 0.1
    b = a @ expected
    x = rt.zeros(expected.shape, np.float32)
    info = rt.full((1,), -1, np.int32)
    rt.launch(cholesky_solve_kernel, (rt.to_device(a), rt.to_device(b), x, info))
    rt.sync()
    error = relative_error(expected, rt.to_host(x))
    info = int(rt.to_host(info)[0])
    print(f"cholesky_solve: info={info}, relative solution error={error:g}")
    assert info == 0 and error < 1e-4


if __name__ == "__main__":
    main()
