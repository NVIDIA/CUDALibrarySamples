# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blocked Cholesky factorization with nvmath.tile.linalg.

This mirrors the C++ example of the same name and uses the same problem sizes.
One block factors one 512x512 matrix. It processes 32x32 tiles in a left-looking order.
Only a few tiles are live at a time:

    for each diagonal tile k:
        A[k][k] -= sum over i < k of A[i][k]^T A[i][k]
        U[k][k]  = cholesky_factor(A[k][k])
        for each panel tile j > k:
            A[k][j] -= sum over i < k of A[i][k]^T A[i][j]
            U[k][j]  = triangular_solve(U[k][k]^T X = A[k][j])

A tile is a value. The working set contains only the named tiles below, and the compiler places them.
The kernel needs no shared-memory slicing, blocked load and store helpers, or barriers.

Only the upper triangle is read and written, and the factor satisfies U^T U = A.
The C++ example also checks against batched cuSolver.
This example checks the factor by reconstruction.
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

# Sizes taken from the example this mirrors. Both are powers of two, as tile
# extents must be, and N is an exact multiple of the tile size.
N, NB, BATCHES = 512, 32, 400
TILES = N // NB


# Left-looking blocked Cholesky over the upper triangle, one matrix per block.
# The batch is folded into the row-tile coordinate: matrix `batch` owns tile
# rows [batch * TILES, (batch + 1) * TILES).
@ct.kernel
def blocked_potrf_kernel(a_data: ct.Array, info_data: ct.Array):
    batch = ct.bid(0)
    base = batch * TILES

    status = ct.zeros((1,), ct.int32)
    for k in range(TILES):
        # Diagonal tile, brought up to date with every panel already computed.
        akk = ct.load(a_data, (base + k, k), shape=(NB, NB))
        for i in range(k):
            aik = ct.load(a_data, (base + i, k), shape=(NB, NB))
            akk = akk - ct.matmul(ct.transpose(aik), aik)
        ukk, info = cholesky_factor(akk, FillMode.UPPER)
        ct.store(a_data, (base + k, k), ukk)
        status = status + info

        # Row panel to the right of the diagonal tile.
        for j in range(k + 1, TILES):
            akj = ct.load(a_data, (base + k, j), shape=(NB, NB))
            for i in range(k):
                aik = ct.load(a_data, (base + i, k), shape=(NB, NB))
                aij = ct.load(a_data, (base + i, j), shape=(NB, NB))
                akj = akj - ct.matmul(ct.transpose(aik), aij)
            ct.store(
                a_data,
                (base + k, j),
                triangular_solve(
                    ukk,
                    akj,
                    FillMode.UPPER,
                    DiagMode.NON_UNIT,
                    SideMode.LEFT,
                    TransposeMode.TRANS,
                ),
            )
    ct.store(info_data, (batch,), status)


def main() -> None:
    rt = Runtime()

    # Symmetric diagonally dominant, hence positive definite. Only the upper
    # triangle is ever read; making the matrix symmetric keeps the reconstruction
    # check below straightforward.
    rows = np.arange(N)[None, :, None]
    cols = np.arange(N)[None, None, :]
    batches = np.arange(BATCHES)[:, None, None]
    a = (
        (np.minimum(rows, cols) * 7 + np.maximum(rows, cols) * 13 + batches * 3) % 11
    ) / 11.0 - 0.5
    a[:, np.arange(N), np.arange(N)] = N
    a = np.ascontiguousarray(a, dtype=np.float64)

    factor = rt.to_device(a.reshape(BATCHES * N, N))
    info = rt.full((BATCHES,), -1, np.int32)
    rt.launch(blocked_potrf_kernel, (factor, info), grid=(BATCHES, 1, 1))
    rt.sync()
    factor = rt.to_host(factor)
    info = rt.to_host(info)

    # U^T U must reproduce the upper triangle of A. The strict lower triangle of the
    # result is untouched scratch, so it is masked away before reconstructing.
    upper = np.triu(factor.reshape(BATCHES, N, N))
    reconstructed = np.matmul(upper.transpose(0, 2, 1), upper)
    error = float(
        np.linalg.norm(np.triu(reconstructed - a)) / np.linalg.norm(np.triu(a))
    )

    print(f"Factoring {BATCHES} {N}x{N} matrices in {NB}x{NB} tiles")
    print(f"Relative reconstruction error of U^T U against A: {error:g}")
    assert int(info.max()) == 0 and int(info.min()) == 0
    assert error < 1e-12


if __name__ == "__main__":
    main()
