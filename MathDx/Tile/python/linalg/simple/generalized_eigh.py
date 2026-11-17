# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generalized symmetric-definite eigenvalues with nvmath.tile.linalg, solved two ways."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import (
    FillMode,
    JobMode,
    cholesky_factor,
    eig_hermitian,
    generalized_eig_hermitian,
    generalized_to_standard_eig_hermitian,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime, relative_error

N = 16


@ct.kernel
def generalized_eigh_kernel(
    a_data: ct.Array, b_data: ct.Array, w_data: ct.Array, info_data: ct.Array
):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    b = ct.load(b_data, (0, 0), shape=(N, N))
    w, info = generalized_eig_hermitian(a, b, 1, FillMode.LOWER, job=JobMode.NO_VECTORS)
    ct.store(w_data, (0,), w)
    ct.store(info_data, (0,), info)


# The same eigenvalues via the steps the call above performs internally:
# factor B, reduce A to a standard eigenproblem, then solve that.
@ct.kernel
def generalized_to_standard_eigh_kernel(
    a_data: ct.Array,
    b_data: ct.Array,
    w_data: ct.Array,
    cholesky_info_data: ct.Array,
    eigh_info_data: ct.Array,
):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    b = ct.load(b_data, (0, 0), shape=(N, N))
    factor, cholesky_info = cholesky_factor(b, FillMode.LOWER)
    reduced = generalized_to_standard_eig_hermitian(a, factor, 1, FillMode.LOWER)
    w, eigh_info = eig_hermitian(reduced, FillMode.LOWER, job=JobMode.NO_VECTORS)
    ct.store(w_data, (0,), w)
    ct.store(cholesky_info_data, (0,), cholesky_info)
    ct.store(eigh_info_data, (0,), eigh_info)


def main() -> None:
    rt = Runtime()

    # Dense symmetric A, and a dense diagonally dominant (so positive-definite) B.
    rows, cols = np.arange(N)[:, None], np.arange(N)[None, :]
    distance = abs(rows - cols).astype(np.float32)
    diagonal = (rows == cols).astype(np.float32)
    a = diagonal * (np.arange(N, dtype=np.float32) + 2)[:, None] + (1 - diagonal) / (
        1 + distance
    )
    b = 4 * diagonal + (1 - diagonal) * 0.5 / (1 + distance)
    d_a = rt.to_device(a)
    d_b = rt.to_device(b)
    w = rt.zeros(N, np.float32)
    reduced_w = rt.zeros(N, np.float32)
    info = rt.full((1,), -1, np.int32)
    cholesky_info = rt.full((1,), -1, np.int32)
    eigh_info = rt.full((1,), -1, np.int32)

    rt.launch(generalized_eigh_kernel, (d_a, d_b, w, info))
    rt.launch(
        generalized_to_standard_eigh_kernel,
        (d_a, d_b, reduced_w, cholesky_info, eigh_info),
    )
    rt.sync()

    # Both paths solve the same problem, so they must agree on the eigenvalues.
    w = rt.to_host(w)
    reduced_w = rt.to_host(reduced_w)
    agreement = relative_error(w, reduced_w)
    info = int(rt.to_host(info)[0])
    cholesky_info = int(rt.to_host(cholesky_info)[0])
    eigh_info = int(rt.to_host(eigh_info)[0])
    print(
        f"HEGV: info={info}, HEGST: info={cholesky_info}/{eigh_info}, "
        f"eigenvalue agreement={agreement:g}"
    )
    assert info == 0 and cholesky_info == 0 and eigh_info == 0 and agreement < 1e-4


if __name__ == "__main__":
    main()
