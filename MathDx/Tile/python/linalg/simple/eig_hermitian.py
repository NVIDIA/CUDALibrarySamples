# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Symmetric eigensolve with nvmath.tile.linalg."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import FillMode, JobMode, eig_hermitian

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime

N = 16


@ct.kernel
def eig_hermitian_kernel(
    a_data: ct.Array, w_data: ct.Array, v_data: ct.Array, info_data: ct.Array
):
    a = ct.load(a_data, (0, 0), shape=(N, N))
    w, v, info = eig_hermitian(a, FillMode.LOWER, job=JobMode.ALL_VECTORS)
    ct.store(w_data, (0,), w)
    ct.store(v_data, (0, 0), v)
    ct.store(info_data, (0,), info)


def main() -> None:
    rt = Runtime()

    # Dense symmetric A: a varying diagonal plus an off-diagonal pattern that
    # depends only on row + col, which keeps it symmetric.
    rows, cols = np.arange(N)[:, None], np.arange(N)[None, :]
    a = np.where(rows == cols, 2 + rows / N, ((rows + cols) % 5) / 16 - 0.125).astype(
        np.float32
    )
    w = rt.zeros(N, np.float32)
    v = rt.zeros(a.shape, np.float32)
    info = rt.full((1,), -1, np.int32)
    rt.launch(eig_hermitian_kernel, (rt.to_device(a), w, v, info))
    rt.sync()
    w = rt.to_host(w)
    v = rt.to_host(v)
    info = int(rt.to_host(info)[0])
    residual = float(np.linalg.norm(a @ v - v * w) / np.linalg.norm(a))
    orthogonality = float(np.linalg.norm(v.T @ v - np.eye(N, dtype=np.float32)))
    print(
        f"eig_hermitian: info={info}, residual={residual:g}, orthogonality error={orthogonality:g}"
    )
    assert info == 0 and residual < 3e-4 and orthogonality < 3e-4


if __name__ == "__main__":
    main()
