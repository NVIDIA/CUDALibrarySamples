# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tridiagonal symmetric eigenvalues with nvmath.tile.linalg."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import JobMode, eig_hermitian_tridiagonal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime

N = 16


@ct.kernel
def eig_hermitian_tridiagonal_kernel(
    diag_data: ct.Array, off_diag_data: ct.Array, w_data: ct.Array, info_data: ct.Array
):
    diag = ct.load(diag_data, (0,), shape=(N,))
    off_diag = ct.load(off_diag_data, (0,), shape=(N,))
    w, info = eig_hermitian_tridiagonal(diag, off_diag, job=JobMode.NO_VECTORS)
    ct.store(w_data, (0,), w)
    ct.store(info_data, (0,), info)


def main() -> None:
    rt = Runtime()

    # Symmetric tridiagonal T with a varying diagonal and a constant off-diagonal.
    # The last off_diag entry is padding, not part of T.
    diag = 2 + np.arange(N, dtype=np.float32) / N
    off_diag = np.zeros(N, dtype=np.float32)
    off_diag[:-1] = 0.25
    w = rt.zeros(N, np.float32)
    info = rt.full((1,), -1, np.int32)
    rt.launch(
        eig_hermitian_tridiagonal_kernel,
        (rt.to_device(diag), rt.to_device(off_diag), w, info),
    )
    rt.sync()
    w = rt.to_host(w)
    info = int(rt.to_host(info)[0])

    # An eigendecomposition preserves the trace and the Frobenius norm, which pins
    # the spectrum down without needing a reference eigensolver.
    trace = float(diag.sum())
    frobenius = float((diag**2).sum() + 2 * (off_diag[:-1] ** 2).sum())
    trace_error = abs(trace - float(w.sum())) / trace
    frobenius_error = abs(frobenius - float((w**2).sum())) / frobenius
    print(
        f"eig_hermitian_tridiagonal: info={info}, trace error={trace_error:g}, "
        f"Frobenius-norm error={frobenius_error:g}"
    )
    assert info == 0 and trace_error < 1e-5 and frobenius_error < 1e-5


if __name__ == "__main__":
    main()
