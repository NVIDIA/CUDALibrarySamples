# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SVD with nvmath.tile.linalg, with and without singular vectors."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import JobMode, svd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime, relative_error

M, N, R = 16, 8, 8


@ct.kernel
def svd_values_kernel(a_data: ct.Array, s_data: ct.Array, info_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(M, N))
    s, info = svd(a, JobMode.NO_VECTORS, JobMode.NO_VECTORS)
    ct.store(s_data, (0,), s)
    ct.store(info_data, (0,), info)


@ct.kernel
def svd_vectors_kernel(
    a_data: ct.Array,
    u_data: ct.Array,
    s_data: ct.Array,
    vt_data: ct.Array,
    info_data: ct.Array,
):
    a = ct.load(a_data, (0, 0), shape=(M, N))
    u, s, vt, info = svd(a, JobMode.SOME_VECTORS, JobMode.SOME_VECTORS)
    ct.store(u_data, (0, 0), u)
    ct.store(s_data, (0,), s)
    ct.store(vt_data, (0, 0), vt)
    ct.store(info_data, (0,), info)


def main() -> None:
    rt = Runtime()

    # Dense, full-rank A so that the singular vectors are well determined.
    rows, cols = np.arange(M)[:, None], np.arange(N)[None, :]
    a = (
        4 * (rows == cols).astype(np.float32)
        + ((rows + 2 * cols) % 7).astype(np.float32) / 4
    )
    d_a = rt.to_device(a)
    values = rt.zeros(R, np.float32)
    u = rt.zeros((M, R), np.float32)
    s = rt.zeros(R, np.float32)
    vt = rt.zeros((R, N), np.float32)
    values_info = rt.full((1,), -1, np.int32)
    vectors_info = rt.full((1,), -1, np.int32)

    rt.launch(svd_values_kernel, (d_a, values, values_info))
    rt.launch(svd_vectors_kernel, (d_a, u, s, vt, vectors_info))
    rt.sync()

    values = rt.to_host(values)
    s = rt.to_host(s)
    u = rt.to_host(u)
    vt = rt.to_host(vt)
    # Both jobs see the same input, so they must agree on the singular values.
    agreement = relative_error(s, values)
    error = float(np.linalg.norm(a - (u * s) @ vt) / np.linalg.norm(a))
    values_info = int(rt.to_host(values_info)[0])
    vectors_info = int(rt.to_host(vectors_info)[0])
    print(
        f"SVD: info={values_info}/{vectors_info}, "
        f"singular-value agreement={agreement:g}, relative reconstruction error={error:g}"
    )
    assert values_info == 0 and vectors_info == 0 and agreement < 1e-4 and error < 1e-4


if __name__ == "__main__":
    main()
