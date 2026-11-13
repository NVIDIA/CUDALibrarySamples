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



# RHF/RKS with fock_method="directjk" (direct 4-center J/K, no DF plan) vs
# stored PySCF references: total energy and nuclear gradient on a distorted
# water molecule.  HF is tested over several basis-set rungs; the DFT
# functionals (hybrid, pure, range-separated, screened-exchange, and
# VV10-corrected) are tested at def2-svp.
#
# The geometry file is in BOHR (bohr_per_ang=1.0) and the PySCF reference
# calculations used the same coordinates with unit='Bohr', so both codes see
# the identical molecule.  Both J and K (including the erf(omega*r)/r
# attenuated exchange of the range-separated functionals) are exact 4-center
# integrals on both sides -- no density fitting anywhere -- so the only
# systematic difference is the XC quadrature, which is matched exactly:
# unpruned (99, 590) XC grid, unpruned (75, 302) VV10 grid,
# Stratmann-Scuseria-Frisch (SSF) partitioning with no atomic-radii
# adjustment, no small-density screening, and grid-weight derivatives
# included in the gradient.
#
# The exact PySCF script used to generate REFERENCES (PySCF >= 2.13 is
# required: it added grid_response support for the Stratmann scheme):
#
#   import pyscf
#   from pyscf import gto, scf, dft
#   assert tuple(int(x) for x in pyscf.__version__.split('.')[:2]) >= (2, 13)
#   PYSCF_XC = {'B3LYP5': 'b3lyp5', 'B97': 'b97', 'PBE': 'pbe',
#               'wB97X': 'wb97x', 'wB97M-V': 'wb97m-v',
#               'HSE06': 'HYB_GGA_XC_HSE06'}
#   for functional, basis in CASES:
#       mol = gto.M(atom=XYZ, unit='Bohr', basis=basis, charge=0)
#       if functional == 'HF':
#           mf = scf.RHF(mol)
#       else:
#           mf = dft.RKS(mol, xc=PYSCF_XC[functional])
#           mf.grids.atom_grid = (99, 590)
#           mf.grids.prune = None
#           mf.grids.becke_scheme = dft.gen_grid.stratmann
#           mf.grids.radii_adjust = None
#           mf.small_rho_cutoff = 0.0
#           mf.nlcgrids.atom_grid = (75, 302)
#           mf.nlcgrids.prune = None
#           mf.nlcgrids.becke_scheme = dft.gen_grid.stratmann
#           mf.nlcgrids.radii_adjust = None
#       mf.conv_tol = 1.0e-12
#       e = mf.kernel()
#       g = mf.nuc_grad_method()
#       if functional != 'HF':
#           g.grid_response = True   # include grid-weight-derivative terms
#       grad = g.kernel()

import os

import numpy as np
import pytest

import cuest_scf

filedir = os.path.dirname(os.path.realpath(__file__))
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

cuest_handle = cuest_scf.CuestHandle()

# (functional, primary basis) -> (PySCF total energy [Ha],
#                                 PySCF nuclear gradient [Ha/Bohr])
REFERENCES = {
    ('HF', 'sto-3g'): (-7.49644049199182e+01, [
        [-1.86423739146386e-03, 3.66031059221914e-02, -7.71986071985697e-04],
        [1.94733573796242e-03, -3.24776689489363e-02, 1.51989466567899e-02],
        [-8.30983464985557e-05, -4.12543697325529e-03, -1.44269605848046e-02]]),
    ('HF', 'def2-svp'): (-7.59550724758876e+01, [
        [-2.55724089636385e-03, 2.46523014386882e-02, -6.49526707732790e-02],
        [3.48932044948189e-03, -5.21207420895042e-02, 4.24195070255831e-02],
        [-9.32079553118109e-04, 2.74684406508159e-02, 2.25331637477029e-02]]),
    ('HF', 'def2-tzvp'): (-7.60527871640442e+01, [
        [-2.60061000026858e-03, 2.38949731593468e-02, -6.89927621148998e-02],
        [3.48527134001252e-03, -5.13694278923145e-02, 4.40973615998153e-02],
        [-8.84661339744020e-04, 2.74744547329719e-02, 2.48954005151034e-02]]),
    ('B3LYP5', 'def2-svp'): (-7.63188620707527e+01, [
        [-1.84352248580121e-03, 2.41099990328491e-02, -3.09773240946409e-02],
        [2.24200667729305e-03, -3.43214906037199e-02, 2.52068592658767e-02],
        [-3.98484191492313e-04, 1.02114915708715e-02, 5.77046482876531e-03]]),
    ('B97', 'def2-svp'): (-7.63301100539045e+01, [
        [-1.89146783132402e-03, 2.43508889048181e-02, -3.27421032116422e-02],
        [2.35225821888085e-03, -3.61067181453631e-02, 2.61984238197133e-02],
        [-4.60790387556864e-04, 1.17558292405486e-02, 6.54367939192446e-03]]),
    ('PBE', 'def2-svp'): (-7.62709783984151e+01, [
        [-1.64550919243129e-03, 2.42822013643863e-02, -2.07418782127951e-02],
        [1.90651803197939e-03, -2.96973271842766e-02, 2.01646228989498e-02],
        [-2.61008839548008e-04, 5.41512581989112e-03, 5.77255313848113e-04]]),
    ('wB97X', 'def2-svp'): (-7.63347400989273e+01, [
        [-2.01913408610917e-03, 2.46412303253593e-02, -3.83299191384912e-02],
        [2.54120387109390e-03, -3.86750694373543e-02, 2.91017093413475e-02],
        [-5.22069784984994e-04, 1.40338391120005e-02, 9.22820979714900e-03]]),
    ('wB97M-V', 'def2-svp'): (-7.63229364426539e+01, [
        [-1.92275739696358e-03, 2.44326680036916e-02, -3.40904639472930e-02],
        [2.40323829482242e-03, -3.68286769035633e-02, 2.69053335055198e-02],
        [-4.80480897858418e-04, 1.23960088998727e-02, 7.18513044177649e-03]]),
    ('HSE06', 'def2-svp'): (-7.62798225695336e+01, [
        [-1.96539186813742e-03, 2.43507179140905e-02, -3.64024438069497e-02],
        [2.48443467498430e-03, -3.79893656741108e-02, 2.80293872149997e-02],
        [-5.19042806847073e-04, 1.36386477600186e-02, 8.37305659194776e-03]]),
}


CASES = sorted(REFERENCES)


def run_directjk(functional, primary_name, incremental_fock, incremental_fock_rebuild=8):
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
        fock_method='directjk',
        directjk_schwarz_threshold=1.0e-14,
        incremental_fock=incremental_fock,
        incremental_fock_rebuild=incremental_fock_rebuild,
        xc_grid_family='UNPRUNED',
        xc_grid_level=(99, 590),
        nlc_grid_family='UNPRUNED',
        nlc_grid_level=(75, 302),
        )
    rhf.solve()
    return rhf.compute_energy(), rhf.compute_gradient().to_numpy()


@pytest.mark.parametrize('incremental_fock', [False, True], ids=['full', 'incremental'])
@pytest.mark.parametrize('functional,primary_name', CASES)
def test_rhf_directjk(functional, primary_name, incremental_fock):
    E, G = run_directjk(functional, primary_name, incremental_fock)
    E_ref, G_ref = REFERENCES[(functional, primary_name)]
    dE = abs(E - E_ref)
    dG = float(np.max(np.abs(G - np.asarray(G_ref))))
    print(f'{functional}/{primary_name}: dE = {dE:.3e}  dG = {dG:.3e}')
    # Observed agreement: dE <= 2.5e-8 (HF/sto-3g; every other case <= 7e-9),
    # dG <= 3.1e-8.  The gates carry a 3-4x margin over the worst case.
    assert dE < 1.0e-7
    assert dG < 1.0e-7
    if incremental_fock:
        # The incremental build shares the full build's screening and integrals, so it must agree
        # with the full build far more tightly than either does with PySCF: an error only the
        # delta logic makes (up to ~1e-7 would still pass the gates above) shows here.
        E_full, G_full = run_directjk(functional, primary_name, False)
        dE_inc = abs(E - E_full)
        dG_inc = float(np.max(np.abs(G - G_full)))
        print(f'{functional}/{primary_name}: |E_inc - E_full| = {dE_inc:.3e}  max|G_inc - G_full| = {dG_inc:.3e}')
        assert dE_inc < 1.0e-8
        assert dG_inc < 1.0e-8


def test_rhf_directjk_rejects_negative_rebuild():
    """A negative incremental_fock_rebuild would read as the never-rebuild
    period (every build after the first a delta build, screening error
    accumulating unbounded); the constructor rejects it."""
    functional, primary_name = CASES[0]
    with pytest.raises(ValueError, match='incremental_fock_rebuild must be >= 0'):
        run_directjk(functional, primary_name, True, incremental_fock_rebuild=-1)
