# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Singular values of a bidiagonal matrix with nvmath.tile.linalg."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import JobMode, svd_bidiagonal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime

N = 16


@ct.kernel
def svd_bidiagonal_kernel(
    diag_data: ct.Array, off_diag_data: ct.Array, s_data: ct.Array, info_data: ct.Array
):
    diag = ct.load(diag_data, (0,), shape=(N,))
    off_diag = ct.load(off_diag_data, (0,), shape=(N,))
    s, info = svd_bidiagonal(diag, off_diag, JobMode.NO_VECTORS, JobMode.NO_VECTORS)
    ct.store(s_data, (0,), s)
    ct.store(info_data, (0,), info)


def main() -> None:
    rt = Runtime()

    # Upper-bidiagonal B: off_diag[i] is the superdiagonal entry in row i, and the
    # last entry is padding rather than part of B.
    diag = np.arange(1, N + 1, dtype=np.float32)
    off_diag = np.zeros(N, dtype=np.float32)
    off_diag[:-1] = 0.5
    s = rt.zeros(N, np.float32)
    info = rt.full((1,), -1, np.int32)
    rt.launch(
        svd_bidiagonal_kernel,
        (rt.to_device(diag), rt.to_device(off_diag), s, info),
    )
    rt.sync()
    s = rt.to_host(s)
    info = int(rt.to_host(info)[0])

    # The squared singular values sum to ||B||_F^2, which pins the spectrum down
    # without needing a reference SVD.
    frobenius = float((diag**2).sum() + (off_diag[:-1] ** 2).sum())
    frobenius_error = abs(frobenius - float((s**2).sum())) / frobenius
    ordered = bool((s >= 0).all() and (s[:-1] >= s[1:]).all())
    print(
        f"BDSVD: info={info}, Frobenius-norm error={frobenius_error:g}, "
        f"ordered={'yes' if ordered else 'no'}"
    )
    assert info == 0 and frobenius_error < 1e-5 and ordered


if __name__ == "__main__":
    main()
