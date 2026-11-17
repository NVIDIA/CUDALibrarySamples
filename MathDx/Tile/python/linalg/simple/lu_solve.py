# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General dense solve with partial-pivot LU through nvmath.tile.linalg."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import lu_solve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime, relative_error

N, NRHS = 32, 4


@ct.kernel
def lu_solve_kernel(
    a_data: ct.Array, b_data: ct.Array, x_data: ct.Array, info_data: ct.Array
):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    b = ct.load(b_data, (0, 0), shape=(N, NRHS))
    x, info = lu_solve(a, b)
    ct.store(x_data, (0, 0), x)
    ct.store(info_data, (0,), info)


def main() -> None:
    rt = Runtime()

    # Dense nonsingular A: a heavy diagonal plus small off-diagonal entries.
    # Solve against a known X.
    rows = np.arange(N, dtype=np.float32)[:, None]
    cols = np.arange(N, dtype=np.float32)[None, :]
    a = 5 * np.eye(N, dtype=np.float32) + ((3 * rows + 5 * cols) % 7 - 3) / 32
    expected = (
        (np.arange(N * NRHS, dtype=np.float32).reshape(N, NRHS) * 7) % 13 - 6
    ) / 13
    b = a @ expected
    x = rt.zeros(expected.shape, np.float32)
    info = rt.full((1,), -1, np.int32)
    rt.launch(lu_solve_kernel, (rt.to_device(a), rt.to_device(b), x, info))
    rt.sync()
    error = relative_error(expected, rt.to_host(x))
    info = int(rt.to_host(info)[0])
    print(f"lu_solve: info={info}, relative solution error={error:g}")
    assert info == 0 and error < 1e-4


if __name__ == "__main__":
    main()
