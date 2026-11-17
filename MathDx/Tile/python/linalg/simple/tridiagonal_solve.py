# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-pivot tridiagonal solve with nvmath.tile.linalg."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import tridiagonal_solve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime, relative_error

N, NRHS = 32, 4


@ct.kernel
def tridiagonal_solve_kernel(
    dl_data: ct.Array,
    diag_data: ct.Array,
    du_data: ct.Array,
    b_data: ct.Array,
    x_data: ct.Array,
    info_data: ct.Array,
):
    dl = ct.load(dl_data, (0,), shape=(N,))
    diag = ct.load(diag_data, (0,), shape=(N,))
    du = ct.load(du_data, (0,), shape=(N,))
    b = ct.load(b_data, (0, 0), shape=(N, NRHS))
    x, info = tridiagonal_solve(dl, diag, du, b)
    ct.store(x_data, (0, 0), x)
    ct.store(info_data, (0,), info)


def main() -> None:
    rt = Runtime()

    # Diagonally dominant tridiagonal system, given as its three diagonals rather
    # than a dense matrix. Solve against a known X.
    dl = np.full(N, 0.25, dtype=np.float32)
    diag = np.full(N, 4, dtype=np.float32)
    du = np.full(N, -0.125, dtype=np.float32)
    expected = np.arange(N * NRHS, dtype=np.float32).reshape(N, NRHS) / 19
    a = np.diag(diag) + np.diag(dl[:-1], -1) + np.diag(du[:-1], 1)
    b = a @ expected
    x = rt.zeros(expected.shape, np.float32)
    info = rt.full((1,), -1, np.int32)
    rt.launch(
        tridiagonal_solve_kernel,
        (
            rt.to_device(dl),
            rt.to_device(diag),
            rt.to_device(du),
            rt.to_device(b),
            x,
            info,
        ),
    )
    rt.sync()
    error = relative_error(expected, rt.to_host(x))
    info = int(rt.to_host(info)[0])
    print(f"tridiagonal_solve: info={info}, relative solution error={error:g}")
    assert info == 0 and error < 2e-4


if __name__ == "__main__":
    main()
