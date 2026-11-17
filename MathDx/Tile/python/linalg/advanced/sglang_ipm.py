# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang load-balancing interior-point solver with nvmath.tile.linalg.

This is a blocked cuTile formulation, adapted from SGLang's LPLB solver and the
DeepSeek LPLB ``minilp`` implementation. The logical 24x42 problem is split
into power-of-two tiles: 24 constraint rows are 16 + 8, while 42 variables use
32 + 16 columns with six inert padded entries.

Each affine-scaling iteration forms a 2x2 blocked normal matrix, factors it
with two Cholesky calls, and solves it with triangular solves. One deterministic
problem is checked against the same algorithm in float64 NumPy.
"""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import (
    DiagMode,
    FillMode,
    SideMode,
    TransposeMode,
    cholesky_factor,
    triangular_solve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime

GROUP_SIZE = 8
DUPLICATES_PER_RANK = 2
NUM_CONSTRAINTS = GROUP_SIZE + GROUP_SIZE * DUPLICATES_PER_RANK
NUM_VARIABLES = GROUP_SIZE * DUPLICATES_PER_RANK * 2 + GROUP_SIZE + 2
NUM_ITERATIONS = 5

ROW_BLOCK_1, ROW_BLOCK_2 = 16, 8
COLUMN_BLOCK_1, COLUMN_BLOCK_2 = 32, 16
PADDED_VARIABLES = COLUMN_BLOCK_1 + COLUMN_BLOCK_2


def make_lplb_instance(generator: np.random.Generator):
    """Build one Cube8P2E load-balancing problem in SGLang's layout."""
    group_size = GROUP_SIZE
    duplicates = DUPLICATES_PER_RANK
    duplicate_workload = generator.uniform(0.05, 1.0, (group_size, duplicates))
    fixed_workload = generator.uniform(0.05, 1.0, group_size)
    rank_to_origin = np.array(
        [
            [
                (current_rank + duplicate + 1) % group_size
                for duplicate in range(duplicates)
            ]
            for current_rank in range(group_size)
        ]
    )

    a = np.zeros((NUM_CONSTRAINTS, NUM_VARIABLES), dtype=np.float64)
    b = np.zeros(NUM_CONSTRAINTS, dtype=np.float64)
    b[:group_size] = -fixed_workload
    b[group_size:] = 1.0

    for constraint in range(NUM_CONSTRAINTS):
        for variable in range(NUM_VARIABLES - 1):
            if constraint < group_size:
                if variable < group_size * duplicates:
                    variable_rank, duplicate = divmod(variable, duplicates)
                    if constraint == variable_rank:
                        a[constraint, variable] = duplicate_workload[
                            variable_rank, duplicate
                        ]
                elif variable < group_size * duplicates * 2:
                    variable_rank, duplicate = divmod(variable, duplicates)
                    variable_rank -= group_size
                    if variable_rank == rank_to_origin[constraint, duplicate]:
                        a[constraint, variable] = duplicate_workload[
                            variable_rank, duplicate
                        ]
                elif variable < group_size * duplicates * 2 + group_size:
                    if variable - group_size * duplicates * 2 == constraint:
                        a[constraint, variable] = 1.0
                elif variable == group_size * duplicates * 2 + group_size:
                    a[constraint, variable] = -1.0
            elif (
                variable == constraint - group_size
                or variable - group_size * duplicates == constraint - group_size
            ):
                a[constraint, variable] = 1.0

        # The Big-M column makes x = 1 a feasible starting point.
        a[constraint, NUM_VARIABLES - 1] = (
            b[constraint] - a[constraint, : NUM_VARIABLES - 1].sum()
        )

    c = np.zeros(NUM_VARIABLES, dtype=np.float64)
    c[NUM_VARIABLES - 2] = 1.0
    c[NUM_VARIABLES - 1] = 1000.0
    return a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)


def reference_solve(a_input, b_input, c_input):
    """Run the affine-scaling iterations in float64 NumPy."""
    a = a_input.astype(np.float64)
    b = b_input.astype(np.float64)
    c = c_input.astype(np.float64)
    x = np.ones(NUM_VARIABLES)
    max_direction = 0.0

    for _ in range(NUM_ITERATIONS):
        ax_squared = a * (x * x)[None, :]
        normal_matrix = ax_squared @ a.T
        rhs = ax_squared @ c
        dual = np.linalg.solve(normal_matrix, rhs)
        direction = x * (c - dual @ a)
        max_direction = direction.max()
        step = 0.999 / max_direction if max_direction > 1e-9 else 1.0
        x *= 1.0 - step * direction

    max_residual = np.abs(a @ x - b).max()
    converged = bool(
        max_direction < 0.1
        and 0.0 <= x[NUM_VARIABLES - 1] < 1e-4
        and max_residual < 0.05
    )
    return x, converged


@ct.kernel
def ipm_kernel(
    a_data: ct.Array,
    b_data: ct.Array,
    c_data: ct.Array,
    x_out: ct.Array,
    status_out: ct.Array,
):
    a11 = ct.load(a_data, (0, 0), shape=(ROW_BLOCK_1, COLUMN_BLOCK_1))
    a12 = ct.load(a_data, (0, 2), shape=(ROW_BLOCK_1, COLUMN_BLOCK_2))
    a21 = ct.load(a_data, (2, 0), shape=(ROW_BLOCK_2, COLUMN_BLOCK_1))
    a22 = ct.load(a_data, (2, 2), shape=(ROW_BLOCK_2, COLUMN_BLOCK_2))
    b1 = ct.load(b_data, (0,), shape=(ROW_BLOCK_1,))
    b2 = ct.load(b_data, (2,), shape=(ROW_BLOCK_2,))
    c1 = ct.load(c_data, (0,), shape=(COLUMN_BLOCK_1,))
    c2 = ct.load(c_data, (2,), shape=(COLUMN_BLOCK_2,))

    x1 = ct.full((COLUMN_BLOCK_1,), 1.0, ct.float32)
    x2 = ct.full((COLUMN_BLOCK_2,), 1.0, ct.float32)
    max_direction = ct.full((), 0.0, ct.float32)
    factor_status = ct.zeros((1,), ct.int32)

    for _ in range(NUM_ITERATIONS):
        ax2_11 = a11 * ct.reshape(x1 * x1, (1, COLUMN_BLOCK_1))
        ax2_12 = a12 * ct.reshape(x2 * x2, (1, COLUMN_BLOCK_2))
        ax2_21 = a21 * ct.reshape(x1 * x1, (1, COLUMN_BLOCK_1))
        ax2_22 = a22 * ct.reshape(x2 * x2, (1, COLUMN_BLOCK_2))

        m11 = ct.matmul(ax2_11, ct.transpose(a11)) + ct.matmul(
            ax2_12, ct.transpose(a12)
        )
        m21 = ct.matmul(ax2_21, ct.transpose(a11)) + ct.matmul(
            ax2_22, ct.transpose(a12)
        )
        m22 = ct.matmul(ax2_21, ct.transpose(a21)) + ct.matmul(
            ax2_22, ct.transpose(a22)
        )
        rhs1 = ct.matmul(ax2_11, ct.reshape(c1, (COLUMN_BLOCK_1, 1))) + ct.matmul(
            ax2_12, ct.reshape(c2, (COLUMN_BLOCK_2, 1))
        )
        rhs2 = ct.matmul(ax2_21, ct.reshape(c1, (COLUMN_BLOCK_1, 1))) + ct.matmul(
            ax2_22, ct.reshape(c2, (COLUMN_BLOCK_2, 1))
        )

        # Block Cholesky of [[m11, m21.T], [m21, m22]].
        l11, info1 = cholesky_factor(m11, FillMode.LOWER)
        l21 = triangular_solve(
            l11,
            m21,
            FillMode.LOWER,
            DiagMode.NON_UNIT,
            SideMode.RIGHT,
            TransposeMode.TRANS,
        )
        l22, info2 = cholesky_factor(
            m22 - ct.matmul(l21, ct.transpose(l21)), FillMode.LOWER
        )
        factor_status = factor_status + info1 + info2

        # Forward and backward substitution with the two Cholesky blocks.
        z1 = triangular_solve(
            l11,
            rhs1,
            FillMode.LOWER,
            DiagMode.NON_UNIT,
            SideMode.LEFT,
            TransposeMode.NO_TRANS,
        )
        z2 = triangular_solve(
            l22,
            rhs2 - ct.matmul(l21, z1),
            FillMode.LOWER,
            DiagMode.NON_UNIT,
            SideMode.LEFT,
            TransposeMode.NO_TRANS,
        )
        y2 = triangular_solve(
            l22,
            z2,
            FillMode.LOWER,
            DiagMode.NON_UNIT,
            SideMode.LEFT,
            TransposeMode.TRANS,
        )
        y1 = triangular_solve(
            l11,
            z1 - ct.matmul(ct.transpose(l21), y2),
            FillMode.LOWER,
            DiagMode.NON_UNIT,
            SideMode.LEFT,
            TransposeMode.TRANS,
        )

        dual1 = ct.reshape(
            ct.matmul(ct.transpose(a11), y1) + ct.matmul(ct.transpose(a21), y2),
            (COLUMN_BLOCK_1,),
        )
        dual2 = ct.reshape(
            ct.matmul(ct.transpose(a12), y1) + ct.matmul(ct.transpose(a22), y2),
            (COLUMN_BLOCK_2,),
        )
        direction1 = x1 * (c1 - dual1)
        direction2 = x2 * (c2 - dual2)
        max_direction = ct.maximum(ct.max(direction1), ct.max(direction2))
        step = ct.where(
            max_direction > 1e-9,
            0.999 / ct.maximum(max_direction, 1e-9),
            1.0,
        )
        x1 = x1 * (1.0 - step * direction1)
        x2 = x2 * (1.0 - step * direction2)

    residual1 = (
        ct.reshape(
            ct.matmul(a11, ct.reshape(x1, (COLUMN_BLOCK_1, 1)))
            + ct.matmul(a12, ct.reshape(x2, (COLUMN_BLOCK_2, 1))),
            (ROW_BLOCK_1,),
        )
        - b1
    )
    residual2 = (
        ct.reshape(
            ct.matmul(a21, ct.reshape(x1, (COLUMN_BLOCK_1, 1)))
            + ct.matmul(a22, ct.reshape(x2, (COLUMN_BLOCK_2, 1))),
            (ROW_BLOCK_2,),
        )
        - b2
    )
    max_residual = ct.maximum(ct.max(ct.abs(residual1)), ct.max(ct.abs(residual2)))
    big_m = ct.extract(x2, (NUM_VARIABLES - 1 - COLUMN_BLOCK_1,), shape=(1,))
    converged = (
        (max_direction < 0.1) & (big_m >= 0.0) & (big_m < 1e-4) & (max_residual < 0.05)
    )

    ct.store(x_out, (0,), ct.where(converged, x1, 0.5))
    ct.store(x_out, (2,), ct.where(converged, x2, 0.5))
    ct.store(status_out, (0,), ct.astype(converged, ct.int32))
    ct.store(status_out, (1,), factor_status)


def main() -> None:
    generator = np.random.default_rng(42)
    a, b, c = make_lplb_instance(generator)
    a_padded = np.zeros((NUM_CONSTRAINTS, PADDED_VARIABLES), dtype=np.float32)
    c_padded = np.zeros(PADDED_VARIABLES, dtype=np.float32)
    a_padded[:, :NUM_VARIABLES] = a
    c_padded[:NUM_VARIABLES] = c

    rt = Runtime()
    d_a = rt.to_device(a_padded)
    d_b = rt.to_device(b)
    d_c = rt.to_device(c_padded)
    d_x = rt.zeros((PADDED_VARIABLES,), np.float32)
    d_status = rt.full((2,), -1, np.int32)
    rt.launch(ipm_kernel, (d_a, d_b, d_c, d_x, d_status))
    rt.sync()

    x = rt.to_host(d_x)
    converged, factor_status = (int(value) for value in rt.to_host(d_status))
    x_reference, reference_converged = reference_solve(a, b, c)

    assert factor_status == 0
    assert bool(converged) == reference_converged
    assert reference_converged
    np.testing.assert_allclose(x[:NUM_VARIABLES], x_reference, rtol=1e-3, atol=1e-4)
    np.testing.assert_array_equal(
        x[NUM_VARIABLES:], np.ones(PADDED_VARIABLES - NUM_VARIABLES)
    )

    max_error = np.max(np.abs(x[:NUM_VARIABLES] - x_reference))
    print(f"Solved one SGLang {NUM_CONSTRAINTS}x{NUM_VARIABLES} load-balancing problem")
    print(f"Maximum absolute error against the float64 reference: {max_error:g}")


if __name__ == "__main__":
    main()
