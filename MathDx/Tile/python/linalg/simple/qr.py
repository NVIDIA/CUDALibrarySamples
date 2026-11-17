# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QR factorization, thin-Q generation, and implicit Q application with nvmath.tile.linalg."""

import sys
from pathlib import Path

import cuda.tile as ct
import numpy as np
from nvmath.tile.linalg import SideMode, qr_factor, qr_generate_q, qr_multiply_by_q

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from common import Runtime

M, N, NRHS = 32, 16, 4


@ct.kernel
def qr_factor_kernel(a_data: ct.Array, tau_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(M, N))
    factors, tau = qr_factor(a)
    ct.store(a_data, (0, 0), factors)
    ct.store(tau_data, (0,), tau)


@ct.kernel
def qr_generate_q_kernel(a_data: ct.Array, tau_data: ct.Array, q_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(M, N))
    tau = ct.load(tau_data, (0,), shape=(N,))
    q = qr_generate_q(a, tau)
    ct.store(q_data, (0, 0), q)


@ct.kernel
def qr_multiply_by_q_kernel(a_data: ct.Array, tau_data: ct.Array, c_data: ct.Array):
    a = ct.load(a_data, (0, 0), shape=(M, N))
    tau = ct.load(tau_data, (0,), shape=(N,))
    c = ct.load(c_data, (0, 0), shape=(M, NRHS))
    q_c = qr_multiply_by_q(a, tau, c, SideMode.LEFT)
    ct.store(c_data, (0, 0), q_c)


def main() -> None:
    rt = Runtime()

    # Well-conditioned dense A, and a dense C for the implicit-Q application.
    rows, cols = np.arange(M)[:, None], np.arange(N)[None, :]
    a = (rows == cols).astype(np.float32) + ((rows + 3 * cols) % 5).astype(
        np.float32
    ) / 32
    original = a.copy()
    c = (
        1
        + ((np.arange(M)[:, None] + 2 * np.arange(NRHS)[None, :]) % 7).astype(
            np.float32
        )
        / 8
    )
    before = float(np.linalg.norm(c))
    d_a = rt.to_device(a)
    d_c = rt.to_device(c)
    tau = rt.zeros(N, np.float32)
    q = rt.zeros(a.shape, np.float32)

    rt.launch(qr_factor_kernel, (d_a, tau))
    rt.launch(qr_generate_q_kernel, (d_a, tau, q))
    rt.launch(qr_multiply_by_q_kernel, (d_a, tau, d_c))
    rt.sync()

    a = rt.to_host(d_a)
    q = rt.to_host(q)
    c = rt.to_host(d_c)
    r = np.triu(a[:N, :])
    error = float(np.linalg.norm(original - q @ r) / np.linalg.norm(original))
    orthogonality = float(np.linalg.norm(q.T @ q - np.eye(N, dtype=np.float32)))
    # Q is orthogonal, so applying it must leave the Frobenius norm of C alone.
    preservation = abs(before - float(np.linalg.norm(c))) / before
    print(
        f"QR: relative reconstruction error={error:g}, orthogonality error={orthogonality:g}, "
        f"norm-preservation error={preservation:g}"
    )
    assert error < 2e-4 and orthogonality < 2e-4 and preservation < 2e-4


if __name__ == "__main__":
    main()
