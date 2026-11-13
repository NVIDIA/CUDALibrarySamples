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




# RHF/RKS with fock_method="directjgridk" (direct 4-center J, grid-based K):
# grid-error consistency vs the other three fock methods on a distorted water
# molecule.
#
# There is no stored reference for this mode -- its absolute accuracy is
# grid-level-dependent by construction (the exact-J side is anchored to PySCF
# in rhf_directjk_1).  Instead the test isolates the grid-K error twice, once
# on each J engine, and requires the two isolations to agree:
#
#     dE_direct = E(directjgridk) - E(directjk)    (exact J on both sides)
#     dE_df     = E(dfjgridk)     - E(dfjk)        (DF J on both sides)
#
# dE_direct is the pure grid-K energy error (both J's are exact 4-center).
# dE_df is the grid-K error minus the DF-K error, because dfjk's exchange is
# itself density-fitted.  The runs use a dense exchange grid
# (gridk_grid_level=5; at the default level 2 the grid-K error is ~7e-5 /
# 8e-4 in E / G), which pushes the grid-K error well below the DF-K error
# (~9e-5 with def2-universal-jkfit), so dE_direct gets a TIGHT gate while
# dE_df and the cross-difference are bounded at the DF-K error scale.  At
# this grid density the grid-K and DF-K gradient errors are comparable, so
# the two gradient isolations agree only in magnitude, not tightly.  The
# vacuity guard lives on the gradient (the energy quadrature error crosses
# zero near convergence and can be arbitrarily small), and the directjgridk
# gradient must be translationally invariant.
#
# The test is additionally parametrized over gridk_sad_reference.  With the
# SAD reference active, K = K_direct(D_sad) + K_gridk(D - D_sad) with the
# expansion-consistent energy, so only the difference-density term carries
# grid error and both isolations shrink accordingly; the gates (including
# the vacuity floor) are set per mode.

import os

import numpy as np
import pytest

import cuest_scf

filedir = os.path.dirname(os.path.realpath(__file__))
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

cuest_handle = cuest_scf.CuestHandle()

FOCK_METHODS = ('dfjk', 'dfjgridk', 'directjk', 'directjgridk')

CASES = [('HF', 'def2-svp'), ('B3LYP5', 'def2-svp')]


def run_scf(functional, primary_name, fock_method, gridk_sad_reference):
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
        xc_functional_name=functional,
        primary=primary,
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=1.0e-14,
        g_convergence=1.0e-8,
        fock_method=fock_method,
        gridk_grid_level=5,
        gridk_sad_reference=gridk_sad_reference,
        directjk_schwarz_threshold=1.0e-14,
        xc_grid_family='UNPRUNED',
        xc_grid_level=(99, 590),
        nlc_grid_family='UNPRUNED',
        nlc_grid_level=(75, 302),
        )
    rhf.solve()
    return rhf.compute_energy(), rhf.compute_gradient().to_numpy()


# Gates per gridk_sad_reference mode, each with a ~4x margin over the
# observed values; the floor is the vacuity guard (see the test body)
GATES = {
    # Observed at gridk_grid_level=5: |dE_direct| <= 4.3e-7, dG_direct in
    # [6.1e-6, 3.1e-5] (the pure grid-K errors, tight gates); |dE_df| and
    # |dE_direct - dE_df| <= 8.6e-5 (= the DF-K energy error),
    # |dG_direct - dG_df| <= 1.2e-5
    False: {
        'dG_direct_floor': 1.0e-6,
        'dE_direct': 2.0e-6,
        'dG_direct': 1.2e-4,
        'dE_df': 3.0e-4,
        'dE_cross': 3.0e-4,
        'dG_cross': 5.0e-5,
        },
    # Observed with the SAD reference: |dE_direct| <= 3.4e-8, dG_direct in
    # [3.3e-7, 1.7e-6] (the residual grid-K error, second order in
    # D - D_sad); |dE_df| and |dE_direct - dE_df| are unchanged (the DF-K
    # error dominates); |dG_direct - dG_df| <= 1.7e-5 (= the DF-K gradient
    # error)
    True: {
        'dG_direct_floor': 1.0e-7,
        'dE_direct': 1.5e-7,
        'dG_direct': 7.0e-6,
        'dE_df': 3.0e-4,
        'dE_cross': 3.0e-4,
        'dG_cross': 7.0e-5,
        },
    }


@pytest.mark.parametrize('gridk_sad_reference', (False, True))
@pytest.mark.parametrize('functional,primary_name', CASES)
def test_rhf_directjgridk_grid_error_consistency(functional, primary_name,
                                                 gridk_sad_reference):
    results = {method: run_scf(functional, primary_name, method,
                               gridk_sad_reference)
               for method in FOCK_METHODS}
    dE_direct = results['directjgridk'][0] - results['directjk'][0]
    dE_df = results['dfjgridk'][0] - results['dfjk'][0]
    dG_direct = float(np.max(np.abs(
        results['directjgridk'][1] - results['directjk'][1])))
    dG_df = float(np.max(np.abs(
        results['dfjgridk'][1] - results['dfjk'][1])))
    trans_inv = float(np.max(np.abs(results['directjgridk'][1].sum(axis=0))))
    tag = f'{functional}/{primary_name}/sad={gridk_sad_reference}'
    print(f'{tag}: dE_direct = {dE_direct:+.6e}  '
          f'dE_df = {dE_df:+.6e}  |diff| = {abs(dE_direct - dE_df):.3e}')
    print(f'{tag}: dG_direct = {dG_direct:.6e}  '
          f'dG_df = {dG_df:.6e}  |diff| = {abs(dG_direct - dG_df):.3e}')
    print(f'{tag}: trans_inv = {trans_inv:.3e}')
    gates = GATES[gridk_sad_reference]
    # Vacuity guard: the grid-K gradient error must be visible, otherwise
    # the bounds below would pass on an accidental all-zero comparison
    assert dG_direct > gates['dG_direct_floor']
    assert abs(dE_direct) < gates['dE_direct']
    assert dG_direct < gates['dG_direct']
    assert abs(dE_df) < gates['dE_df']
    assert abs(dE_direct - dE_df) < gates['dE_cross']
    assert abs(dG_direct - dG_df) < gates['dG_cross']
    assert trans_inv < 1.0e-10
