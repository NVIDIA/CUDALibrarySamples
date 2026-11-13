# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Two-density (bilinear) direct exchange gradient through the SCF-level
# wrapper CuestDirectIntCompute.symmetric_exchange_bilinear_gradient:
#
#   B(D1, D2) = d/dR Tr(D1 K[D2]),   D_m = Cocc_m^T Cocc_m
#
# with a second, DIFFERENT occupied-coefficient matrix (Coccptrs2), checked
# against the polarization identity on the wrapper's own single-density
# gradient G(D) = d/dR Tr(D K[D]),
#
#   B(D1, D2) = 1/2 [G(D1 + D2) - G(D1) - G(D2)],
#
# where D1 + D2 = Cstack^T Cstack for the row-stacked Cstack = [Cocc1; Cocc2],
# so every term is reachable through the same Cocc interface.  Unscreened
# (schwarz_threshold = 0) so the identity is exact to rounding and the
# bilinear path's screening cannot enter.  Also: Coccptrs2 equal
# to Coccptrs reproduces the single-density gradient, and a batch of two
# pairs is handled entry by entry.  Water / HF / sto-3g and def2-svp.

import os

import numpy as np
import pytest

import cuest_scf
from cuest_scf.cuest_direct_int_compute import CuestDirectIntCompute
from cuest_scf.gpu_matrix import GPUMatrix

filedir = os.path.dirname(os.path.realpath(__file__))
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

cuest_handle = cuest_scf.CuestHandle()

_TOL_ABS = 1.0e-9
_TOL_REL = 1.0e-9


def _solve_rhf(primary_name):
    molecule = cuest_scf.Molecule.parse_from_xyz_file(
        os.path.join(filedir, 'water_bohr.xyz'), bohr_per_ang=1.0)
    auxiliary_name = 'def2-universal-jkfit'
    minao_name = 'minao-1'
    primary = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % primary_name), molecule=molecule)
    auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % auxiliary_name), molecule=molecule)
    minao = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % minao_name), molecule=molecule)
    rhf = cuest_scf.RHF(
        cuest_handle=cuest_handle,
        molecule=molecule,
        charge=0,
        xc_functional_name='HF',
        primary=primary,
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=1.0e-14,
        g_convergence=1.0e-8,
        fock_method='directjk',
        directjk_schwarz_threshold=1.0e-14,
        )
    rhf.solve()
    return rhf


def _k_gradients(rhf, Coccs, Coccs2=None):
    """Wrapper call: one (natom, 3) gradient per Cocc pair.  Coccs2 = None
    is the exchange-energy gradient (symmetric_exchange_gradient); otherwise
    the bilinear entry point."""
    natom = rhf.primary.natom
    Gs = [GPUMatrix(nrows=natom, ncols=3, dtype=np.double) for _ in Coccs]
    if Coccs2 is None:
        CuestDirectIntCompute.symmetric_exchange_gradient(
            handle=cuest_handle,
            direct_int_plan=rhf.cuest_direct_int_plan,
            noccs=[c.shape[0] for c in Coccs],
            Coccptrs=[c.pointer for c in Coccs],
            Gptrs=[g.pointer for g in Gs],
            schwarz_threshold=0.0)
    else:
        CuestDirectIntCompute.symmetric_exchange_bilinear_gradient(
            handle=cuest_handle,
            direct_int_plan=rhf.cuest_direct_int_plan,
            noccs1=[c.shape[0] for c in Coccs],
            Coccptrs1=[c.pointer for c in Coccs],
            noccs2=[c.shape[0] for c in Coccs2],
            Coccptrs2=[c.pointer for c in Coccs2],
            Gptrs=[g.pointer for g in Gs],
            schwarz_threshold=0.0)
    return [g.to_numpy() for g in Gs]


def _structured_perturbation(shape):
    """Deterministic dense (nocc, nao) perturbation (no RNG), O(1) entries
    varying in sign across both indices."""
    a = np.arange(shape[0], dtype=np.float64)[:, None]
    m = np.arange(shape[1], dtype=np.float64)[None, :]
    return np.sin(1.3 * a + 0.7 * m + 0.5) * np.cos(0.31 * m - 0.2 * a)


def _assert_close(G, G_ref, label):
    dev = float(np.abs(G - G_ref).max())
    scale = max(float(np.abs(G_ref).max()), 1.0)
    print(f'{label}: abs-dev = {dev:.3e} (scale {scale:.3e})')
    assert dev <= max(_TOL_ABS, _TOL_REL * scale), label


@pytest.mark.parametrize('primary_name', ['sto-3g', 'def2-svp'])
def test_directk_bilinear_gradient(primary_name):
    rhf = _solve_rhf(primary_name)
    Cocc1 = rhf.tensors['Cocc']                     # (nocc, nao)
    C1 = Cocc1.to_numpy()

    # A second, different coefficient matrix (deterministic perturbation): D2 =
    # C2^T C2 is symmetric and differs from D1 in every element.
    C2 = C1 + 0.3 * _structured_perturbation(C1.shape)
    Cocc2 = GPUMatrix.from_numpy(np.ascontiguousarray(C2))
    Cstack = GPUMatrix.from_numpy(np.ascontiguousarray(np.vstack([C1, C2])))

    D1 = C1.T @ C1
    D2 = C2.T @ C2
    assert float(np.abs(D1 - D2).max()) > 1.0e-3

    # Bilinear gradient vs the polarization identity.
    B = _k_gradients(rhf, [Cocc1], [Cocc2])[0]
    G1 = _k_gradients(rhf, [Cocc1])[0]
    G2 = _k_gradients(rhf, [Cocc2])[0]
    G12 = _k_gradients(rhf, [Cstack])[0]           # D1 + D2 = Cstack^T Cstack
    identity = 0.5 * (G12 - G1 - G2)
    assert float(np.abs(B).max()) > 1.0e-6
    _assert_close(B, identity, f'{primary_name}: bilinear vs identity')
    assert float(np.abs(B.sum(axis=0)).max()) < 1.0e-9   # translational invariance

    # Symmetry in the two densities (unscreened).
    B21 = _k_gradients(rhf, [Cocc2], [Cocc1])[0]
    _assert_close(B21, B, f'{primary_name}: bilinear symmetry')

    # Batched pairs: [(D1, D2), (D1, D1)] in one call; pair 1 takes a distinct
    # copy of Cocc1 as its second side and must reproduce the single-density
    # gradient.
    Cocc1_copy = GPUMatrix.from_numpy(np.ascontiguousarray(C1))
    Bb = _k_gradients(rhf, [Cocc1, Cocc1], [Cocc2, Cocc1_copy])
    _assert_close(Bb[0], B, f'{primary_name}: batched pair 0')
    _assert_close(Bb[1], G1, f'{primary_name}: batched pair 1')

    # The density entry point (symmetric_exchange_bilinear_gradient_density)
    # takes the densities as device pointers, so it also takes an INDEFINITE
    # density: with the delta density D2 - D1 on the second side, bilinearity
    # gives B(D1, D2 - D1) = B(D1, D2) - G(D1).
    natom = rhf.primary.natom
    Gd = GPUMatrix(nrows=natom, ncols=3, dtype=np.double)
    D1_gpu = GPUMatrix.from_numpy(np.ascontiguousarray(D1))
    Ddelta_gpu = GPUMatrix.from_numpy(np.ascontiguousarray(D2 - D1))
    CuestDirectIntCompute.symmetric_exchange_bilinear_gradient_density(
        handle=cuest_handle,
        direct_int_plan=rhf.cuest_direct_int_plan,
        D1ptrs=[D1_gpu.pointer],
        D2ptrs=[Ddelta_gpu.pointer],
        Gptrs=[Gd.pointer],
        schwarz_threshold=0.0)
    _assert_close(Gd.to_numpy(), B - G1, f'{primary_name}: delta density via the density entry point')


def test_directk_bilinear_gradient_length_mismatch():
    rhf = _solve_rhf('sto-3g')
    Cocc1 = rhf.tensors['Cocc']
    with pytest.raises(RuntimeError):
        _k_gradients(rhf, [Cocc1, Cocc1], [Cocc1])
