# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cholesky factorization with nvmath.tile.linalg."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import FillMode, cholesky_factor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime

N = 32


@ct.kernel
def cholesky_factor_kernel(a_data: ct.Array, factors_data: ct.Array, info_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    factors, info = cholesky_factor(a, FillMode.LOWER)
    ct.store(factors_data, (0, 0), factors)
    ct.store(info_data, (0,), info)


def main() -> None:
    rt = Runtime()

    # Dense symmetric positive-definite A, built as L L^T from a dense
    # lower-triangular factor whose diagonal dominates its off-diagonal entries.
    rows, cols = np.arange(N)[:, None], np.arange(N)[None, :]
    lower = np.tril(
        np.where(rows == cols, 2 + rows / N, ((rows + 2 * cols) % 5) / 8 - 0.25)
    ).astype(np.float32)
    a = lower @ lower.T
    factors = rt.zeros(a.shape, np.float32)
    info = rt.full((1,), -1, np.int32)
    rt.launch(cholesky_factor_kernel, (rt.to_device(a), factors, info))
    rt.sync()
    factors = rt.to_host(factors)
    info = int(rt.to_host(info)[0])
    reconstructed = np.tril(factors) @ np.tril(factors).T
    error = float(np.linalg.norm(a - reconstructed) / np.linalg.norm(a))
    print(f"cholesky_factor: info={info}, relative reconstruction error={error:g}")
    assert info == 0 and error < 1e-4


if __name__ == "__main__":
    main()
