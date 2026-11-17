# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modified LU factorization with nvmath.tile.linalg: A - diag(S) = L U."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import modified_lu_factor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime

M, N = 16, 8


@ct.kernel
def modified_lu_factor_kernel(
    a_data: ct.Array, factors_data: ct.Array, s_data: ct.Array
):
    a = ct.load(a_data, (0, 0), shape=(M, N))
    factors, shifts = modified_lu_factor(a)
    ct.store(factors_data, (0, 0), factors)
    ct.store(s_data, (0,), shifts)


def main() -> None:
    rt = Runtime()

    # Dense tall A whose leading N x N block is diagonally dominant, since this
    # factorization does not pivot.
    rows, cols = np.arange(M)[:, None], np.arange(N)[None, :]
    a = (4 * (rows == cols) + ((rows + 2 * cols) % 5) / 8 - 0.25).astype(np.float32)
    factors = rt.zeros(a.shape, np.float32)
    s = rt.zeros(N, np.float32)
    rt.launch(modified_lu_factor_kernel, (rt.to_device(a), factors, s))
    rt.sync()
    factors = rt.to_host(factors)
    s = rt.to_host(s)

    # L is unit lower trapezoidal and U upper triangular, both packed into factors.
    lower = np.tril(factors, -1) + np.eye(M, N, dtype=np.float32)
    upper = np.triu(factors[:N, :])
    expected = a.copy()
    expected[np.arange(N), np.arange(N)] -= s
    error = float(np.linalg.norm(expected - lower @ upper) / np.linalg.norm(expected))
    print(f"modified_lu_factor: relative reconstruction error={error:g}")
    assert error < 1e-4


if __name__ == "__main__":
    main()
