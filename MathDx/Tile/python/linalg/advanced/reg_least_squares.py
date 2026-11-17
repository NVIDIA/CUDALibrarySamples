# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched regularized least squares with nvmath.tile.linalg.

    minimize ||b - A x||^2 + lambda ||x||^2

This mirrors the C++ example of the same name, with the same problem sizes, and
compares the same two tile approaches:

  1. A fused kernel forms the normal equations (A^T A + lambda I) x = A^T b with tile matmuls.
     It factors them with Cholesky. The Gram matrix never reaches global memory.
  2. A kernel runs Householder QR on the augmented system [A; sqrt(lambda) I] x ~ [b; 0].
     This costs more but does not square the condition number.

The C++ example additionally checks both against the batched cuBLAS and
cuSolver host APIs.
This example checks the two kernels against each other and against the optimality condition.

One block solves one problem: each kernel indexes its batch through the
partition coordinates of a load that spans the whole batch.
"""

import math
import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import FillMode, cholesky_solve, least_squares_solve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime, relative_error

# Sizes taken from the example this mirrors.
M, N, NRHS, BATCHES = 100, 8, 1, 10000
# Tile extents must be powers of two, so the M x N system is loaded as a
# 128 x 8 tile whose trailing rows are zero. Zero rows contribute nothing to
# A^T A or A^T b, and they leave room for the regularization block that the QR
# path writes at rows M .. M + N. Only the tensor extent spanning the batch is
# free of that restriction, which is what lets the batch stay at 10000.
M_TILE = 128
LAMBDA = 0.1
# Augmenting with sqrt(lambda) I is what makes ||[A; SCALE I] x - [b; 0]||^2
# equal the regularized objective.
SCALE = math.sqrt(LAMBDA)


# Fused solve: [G v] = A^T [A b], G += lambda I, then Cholesky-solve G x = v.
@ct.kernel
def normal_equations_kernel(
    a_data: ct.Array, b_data: ct.Array, x_data: ct.Array, info_data: ct.Array
):
    batch = ct.bid(0)
    a = ct.load(a_data, (batch, 0), shape=(M_TILE, N))
    a_transposed = ct.transpose(a)

    # lambda on the diagonal and zero elsewhere, from a row-major arange.
    index = ct.reshape(ct.arange(N * N, dtype=ct.int32), (N, N))
    regularizer = ct.where(
        index // N == index % N,
        ct.full((N, N), LAMBDA, ct.float32),
        ct.zeros((N, N), ct.float32),
    )

    g = ct.matmul(a_transposed, a) + regularizer
    v = ct.matmul(a_transposed, ct.load(b_data, (batch, 0), shape=(M_TILE, NRHS)))
    x, info = cholesky_solve(g, v, FillMode.LOWER)
    ct.store(x_data, (batch, 0), x)
    ct.store(info_data, (batch,), info)


# The same problem as a plain least-squares solve on the augmented system, whose
# normal equations are exactly the ones above.
@ct.kernel
def householder_kernel(a_data: ct.Array, b_data: ct.Array, x_data: ct.Array):
    batch = ct.bid(0)

    # Write SCALE on the diagonal of the block starting at row M. The rows of A
    # past M are already zero and b is padded with zeros to match, so the
    # augmented system is assembled without touching memory.
    index = ct.reshape(ct.arange(M_TILE * N, dtype=ct.int32), (M_TILE, N))
    regularizer = ct.where(
        index // N - M == index % N,
        ct.full((M_TILE, N), SCALE, ct.float32),
        ct.zeros((M_TILE, N), ct.float32),
    )

    a = ct.load(a_data, (batch, 0), shape=(M_TILE, N)) + regularizer
    b = ct.load(b_data, (batch, 0), shape=(M_TILE, NRHS))
    _, _, x = least_squares_solve(a, b)
    ct.store(x_data, (batch, 0), x)


# Real-valued FLOP counts, matching common/measure.hpp in the cuSOLVERDx
# examples so the throughput numbers are comparable. They count the useful
# M x N work, not the padding the tile kernels carry.
def flops_trsm(order, k):
    return k * order * (order + 1) / 2 + k * order * (order - 1) / 2


def flops_potrf(order):
    return order * ((order / 6 + 0.5) * order + 1 / 3) + order * (
        (order / 6) * order - 1 / 6
    )


def flops_geqrf(rows, cols):
    return cols * (cols * (0.5 - cols / 3 + rows) + rows + 23 / 6) + cols * (
        cols * (0.5 - cols / 3 + rows) + 5 / 6
    )


def flops_unmqr_left(rows, cols, k):
    return 2 * k * (2 * rows - k) * cols


NORMAL_FLOPS = 2 * M * N * N + N + flops_potrf(N) + 2 * flops_trsm(N, NRHS)
HOUSEHOLDER_FLOPS = (
    flops_geqrf(M + N, N) + flops_unmqr_left(M + N, NRHS, N) + flops_trsm(N, NRHS)
)


def gflops(flops_per_batch, ms):
    return flops_per_batch / (ms / 1e3 / BATCHES * 1e9)


def objective(a, b, solution):
    """||b - A x||^2 + lambda ||x||^2, summed over the batch."""
    residual = b - a @ solution
    return float((residual * residual).sum() + LAMBDA * (solution * solution).sum())


def optimality_error(a, b, solution):
    """The objective is strictly convex, so a solution minimizes it exactly when
    the gradient vanishes, that is, when (A^T A + lambda I) x equals A^T b. Measuring
    that residual certifies optimality without needing a reference solution."""
    a_transposed = a.transpose(0, 2, 1)
    lhs = a_transposed @ (a @ solution) + LAMBDA * solution
    rhs = a_transposed @ b
    return relative_error(rhs, lhs)


def main() -> None:
    rt = Runtime()

    # Random A and b over [-2, 2], as in the example this mirrors, but from a fixed
    # seed so the run is reproducible.
    generator = np.random.default_rng(1234)
    a = generator.uniform(-2, 2, (BATCHES, M, N)).astype(np.float32)
    b = generator.uniform(-2, 2, (BATCHES, M, NRHS)).astype(np.float32)

    # Zero-padded to M_TILE rows for the tile kernels.
    a_tiles = np.zeros((BATCHES, M_TILE, N), dtype=np.float32)
    a_tiles[:, :M, :] = a
    b_tiles = np.zeros((BATCHES, M_TILE, NRHS), dtype=np.float32)
    b_tiles[:, :M, :] = b

    d_a = rt.to_device(a_tiles.reshape(BATCHES * M_TILE, N))
    d_b = rt.to_device(b_tiles.reshape(BATCHES * M_TILE, NRHS))
    x = rt.zeros((BATCHES, N, NRHS), np.float32)
    x_qr = rt.zeros((BATCHES, N, NRHS), np.float32)
    info = rt.full((BATCHES,), -1, np.int32)
    grid = (BATCHES, 1, 1)

    def run_normal_equations():
        rt.launch(
            normal_equations_kernel,
            (d_a, d_b, x.reshape(BATCHES * N, NRHS), info),
            grid=grid,
        )

    def run_householder():
        rt.launch(
            householder_kernel,
            (d_a, d_b, x_qr.reshape(BATCHES * N, NRHS)),
            grid=grid,
        )

    normal_ms = rt.measure(run_normal_equations)
    householder_ms = rt.measure(run_householder)
    x = rt.to_host(x)
    x_qr = rt.to_host(x_qr)
    info = rt.to_host(info)

    error = optimality_error(a, b, x)
    qr_error = optimality_error(a, b, x_qr)
    agreement = relative_error(x, x_qr)
    print(
        f"Solving {BATCHES} regularized {M}x{N} least squares problems, lambda={LAMBDA:g}"
    )
    print(
        f"Dx normal equations {gflops(NORMAL_FLOPS, normal_ms):7.2f} GFLOP/s, "
        f"{normal_ms:7.2f} ms, objective {objective(a, b, x):e}"
    )
    print(
        f"Dx Householder      {gflops(HOUSEHOLDER_FLOPS, householder_ms):7.2f} GFLOP/s, "
        f"{householder_ms:7.2f} ms, objective {objective(a, b, x_qr):e}"
    )
    print(f"Optimality error: normal equations={error:g}, Householder={qr_error:g}")
    print(f"Agreement between the two: {agreement:g}")
    assert int(info.max()) == 0 and int(info.min()) == 0
    assert error < 1e-4 and qr_error < 1e-4 and agreement < 1e-4


if __name__ == "__main__":
    main()
