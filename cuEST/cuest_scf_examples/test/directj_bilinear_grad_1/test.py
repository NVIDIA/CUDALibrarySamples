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

# Two-density (bilinear) direct Coulomb gradient through the SCF-level
# wrapper CuestDirectIntCompute.coulomb_bilinear_gradient:
#
#   B(D1, D2) = d/dR Tr(D1 J[D2]),   D_m = Cocc_m^T Cocc_m
#
# with a second, DIFFERENT density, checked against the polarization
# identity on the wrapper's own single-density gradient
# G(D) = d/dR Tr(D J[D]) (coulomb_gradient),
#
#   B(D1, D2) = 1/2 [G(D1 + D2) - G(D1) - G(D2)].
#
# Unscreened (schwarz_threshold = 0) so the identity is exact to rounding
# and the bilinear path's screening cannot enter.  Also: D2 equal to D1
# (a distinct buffer) reproduces the single-density gradient, and a batch
# of two pairs is handled entry by entry.  Water / HF / sto-3g and
# def2-svp.

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


def _j_gradients(rhf, Ds, D2s=None):
    """Wrapper call over a batch of host densities: one (natom, 3) gradient
    per entry.  D2s = None is the Coulomb-energy gradient
    (coulomb_gradient); otherwise the bilinear entry point."""
    natom = rhf.primary.natom
    nmat = len(Ds)
    D = GPUMatrix.from_numpy(np.ascontiguousarray(np.vstack(Ds)))
    G = GPUMatrix(nrows=nmat * natom, ncols=3, dtype=np.double)
    if D2s is None:
        CuestDirectIntCompute.coulomb_gradient(
            handle=cuest_handle,
            direct_int_plan=rhf.cuest_direct_int_plan,
            Dptr=D.pointer,
            Gptr=G.pointer,
            schwarz_threshold=0.0,
            num_density_matrices=nmat)
    else:
        D2 = GPUMatrix.from_numpy(np.ascontiguousarray(np.vstack(D2s)))
        CuestDirectIntCompute.coulomb_bilinear_gradient(
            handle=cuest_handle,
            direct_int_plan=rhf.cuest_direct_int_plan,
            D1ptr=D.pointer,
            D2ptr=D2.pointer,
            Gptr=G.pointer,
            schwarz_threshold=0.0,
            num_density_matrices=nmat)
    return list(G.to_numpy().reshape(nmat, natom, 3))


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
def test_directj_bilinear_gradient(primary_name):
    rhf = _solve_rhf(primary_name)
    C1 = rhf.tensors['Cocc'].to_numpy()             # (nocc, nao)

    # A second, different coefficient matrix (deterministic perturbation): D2 =
    # C2^T C2 is symmetric and differs from D1 in every element.
    C2 = C1 + 0.3 * _structured_perturbation(C1.shape)
    D1 = C1.T @ C1
    D2 = C2.T @ C2
    assert float(np.abs(D1 - D2).max()) > 1.0e-3

    # Bilinear gradient vs the polarization identity.
    B = _j_gradients(rhf, [D1], [D2])[0]
    G1 = _j_gradients(rhf, [D1])[0]
    G2 = _j_gradients(rhf, [D2])[0]
    G12 = _j_gradients(rhf, [D1 + D2])[0]
    identity = 0.5 * (G12 - G1 - G2)
    assert float(np.abs(B).max()) > 1.0e-6
    _assert_close(B, identity, f'{primary_name}: bilinear vs identity')
    assert float(np.abs(B.sum(axis=0)).max()) < 1.0e-9   # translational invariance

    # Symmetry in the two densities (unscreened).
    B21 = _j_gradients(rhf, [D2], [D1])[0]
    _assert_close(B21, B, f'{primary_name}: bilinear symmetry')

    # Batched pairs: [(D1, D2), (D1, D1)] in one call; pair 1 (D1 on distinct
    # buffers) reproduces the single-density gradient.
    Bb = _j_gradients(rhf, [D1, D1], [D2, D1])
    _assert_close(Bb[0], B, f'{primary_name}: batched pair 0')
    _assert_close(Bb[1], G1, f'{primary_name}: batched pair 1')
