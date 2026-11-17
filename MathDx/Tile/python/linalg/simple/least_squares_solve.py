# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Overdetermined least-squares solve with nvmath.tile.linalg, expressed both ways."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import TransposeMode, least_squares_solve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime, relative_error

M, N, NRHS = 32, 16, 4


@ct.kernel
def least_squares_solve_kernel(a_data: ct.Array, b_data: ct.Array, x_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(M, N))
    b = ct.load(b_data, (0, 0), shape=(M, NRHS))
    _, _, x = least_squares_solve(a, b)
    ct.store(x_data, (0, 0), x)


# The same problem given the wide matrix instead: op(A) is again M x N, so this
# consumes the same B and produces the same X.
@ct.kernel
def least_squares_solve_transpose_kernel(
    at_data: ct.Array, b_data: ct.Array, x_data: ct.Array
):
    at = ct.load(at_data, (0, 0), shape=(N, M))
    b = ct.load(b_data, (0, 0), shape=(M, NRHS))
    _, _, x = least_squares_solve(at, b, TransposeMode.TRANS)
    ct.store(x_data, (0, 0), x)


def main() -> None:
    rt = Runtime()

    # Dense full-rank tall A, plus its explicit transpose. Building B from a known X
    # makes the overdetermined system consistent, so the least-squares solution is
    # that X.
    rows, cols = np.arange(M)[:, None], np.arange(N)[None, :]
    a = (rows == cols).astype(np.float32) + ((rows + 2 * cols) % 7).astype(
        np.float32
    ) / 64
    at = np.ascontiguousarray(a.T)
    expected = (np.arange(N * NRHS, dtype=np.float32).reshape(N, NRHS) % 9) / 9
    b = a @ expected
    x = rt.zeros(expected.shape, np.float32)
    transposed_x = rt.zeros(expected.shape, np.float32)
    d_b = rt.to_device(b)

    rt.launch(least_squares_solve_kernel, (rt.to_device(a), d_b, x))
    rt.launch(
        least_squares_solve_transpose_kernel,
        (rt.to_device(at), d_b, transposed_x),
    )
    rt.sync()

    # Both calls describe the same problem, so they must agree on the solution.
    x = rt.to_host(x)
    transposed_x = rt.to_host(transposed_x)
    error = relative_error(expected, x)
    transposed_error = relative_error(expected, transposed_x)
    agreement = relative_error(x, transposed_x)
    print(
        f"least_squares_solve: relative solution error={error:g}, transposed={transposed_error:g}, "
        f"agreement={agreement:g}"
    )
    assert error < 2e-4 and transposed_error < 2e-4 and agreement < 2e-4


if __name__ == "__main__":
    main()
