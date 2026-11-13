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




# UHF/UKS with fock_method="directjk" (direct 4-center J/K, no DF plan) vs
# stored PySCF references: total energy and nuclear gradient on a distorted
# CH3 radical (doublet).  The two spin exchange builds go through the
# multi-density batch of the direct J/K API.  HF is tested over several basis-set rungs; the DFT
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
#       mol = gto.M(atom=XYZ, unit='Bohr', basis=basis, charge=0, spin=1)
#       if functional == 'HF':
#           mf = scf.UHF(mol)
#       else:
#           mf = dft.UKS(mol, xc=PYSCF_XC[functional])
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
    ('HF', 'sto-3g'): (-3.90762491025792e+01, [
        [-1.79861630231433e-02, 4.37955282070479e-03, 6.99060410629671e-05],
        [6.78443860446132e-03, 4.43000461807369e-03, -2.19794895302945e-04],
        [1.04839315297081e-02, -5.65415522722246e-03, 1.70530003317154e-04],
        [7.17792888972890e-04, -3.15540221155586e-03, -2.06411490770858e-05]]),
    ('HF', 'def2-svp'): (-3.95325587694036e+01, [
        [-1.42341607896200e-02, 3.79222989621100e-03, -7.23411506613827e-04],
        [3.38711338025055e-03, 3.53121736566506e-03, 4.56423170012016e-05],
        [9.49294641546983e-03, -6.35430898706790e-03, 5.30152726049712e-04],
        [1.35410099389521e-03, -9.69138274806269e-04, 1.47616463562941e-04]]),
    ('HF', 'def2-tzvp'): (-3.95773401568763e+01, [
        [-1.36796533848595e-02, 3.10111550258499e-03, -1.07561034196758e-03],
        [9.57181741458335e-03, 3.87241047038192e-03, 2.15733071394171e-04],
        [5.91673061642839e-03, -8.35924157160939e-05, 1.95689027875368e-04],
        [-1.80889464616008e-03, -6.88993355725143e-03, 6.64188242697747e-04]]),
    ('B3LYP5', 'def2-svp'): (-3.97764507186080e+01, [
        [-1.29306958300778e-02, 4.42037609073216e-03, -7.62708181438254e-04],
        [-4.08375155781648e-03, 2.73435853522511e-03, 6.46820035276469e-06],
        [1.24174641576589e-02, -1.28838703464398e-02, 1.02658525132097e-03],
        [4.59698323023428e-03, 5.72913572047495e-03, -2.70345270235173e-04]]),
    ('B97', 'def2-svp'): (-3.97900436757140e+01, [
        [-1.29495728833838e-02, 4.51233420788624e-03, -6.57451655333263e-04],
        [-5.05184846882822e-03, 2.68625803757883e-03, -3.68870031183730e-05],
        [1.29421550440389e-02, -1.38049818477981e-02, 1.05879462961807e-03],
        [5.05926630817555e-03, 6.60638960233650e-03, -3.64455971166314e-04]]),
    ('PBE', 'def2-svp'): (-3.97391795367598e+01, [
        [-1.25715851056626e-02, 4.85480090750839e-03, -5.80444815121137e-04],
        [-8.86160633420263e-03, 2.35269104530171e-03, -9.15015149837370e-05],
        [1.46480524816419e-02, -1.72662866350499e-02, 1.28758181413367e-03],
        [6.78513895823019e-03, 1.00587946822406e-02, -6.15635484029084e-04]]),
    ('wB97X', 'def2-svp'): (-3.97905122767304e+01, [
        [-1.32715142988409e-02, 4.50810162952343e-03, -7.57462826301986e-04],
        [-3.95447431093032e-03, 2.75548363529342e-03, 4.97840730285387e-06],
        [1.25642189435132e-02, -1.30395063031088e-02, 1.03226437030679e-03],
        [4.66176966625509e-03, 5.77592103829239e-03, -2.79779951307527e-04]]),
    ('wB97M-V', 'def2-svp'): (-3.97670201107193e+01, [
        [-1.30283234448282e-02, 4.42948786801885e-03, -8.90507578976296e-04],
        [-2.48005211275348e-03, 2.76306994285060e-03, 6.41281009293240e-05],
        [1.16173421894129e-02, -1.16039030778017e-02, 9.69120001282797e-04],
        [3.89103336816299e-03, 4.41134526693032e-03, -1.42740523235624e-04]]),
    ('HSE06', 'def2-svp'): (-3.97561726434627e+01, [
        [-1.31028772798873e-02, 4.51992639589061e-03, -8.05925477754765e-04],
        [-3.43548534186211e-03, 2.75494523008718e-03, 2.52184206548378e-05],
        [1.21967435252245e-02, -1.25204740872673e-02, 1.00809946927650e-03],
        [4.34161909652508e-03, 5.24560246129391e-03, -2.27392412176924e-04]]),
}



CASES = sorted(REFERENCES)


def run_directjk(functional, primary_name, incremental_fock, incremental_fock_rebuild=8):
    molecule = cuest_scf.Molecule.parse_from_xyz_file(
        os.path.join(filedir, 'ch3_bohr.xyz'), bohr_per_ang=1.0)

    auxiliary_name = 'def2-universal-jkfit'
    minao_name = 'minao-1'
    primary = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % primary_name), molecule=molecule)
    auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % auxiliary_name), molecule=molecule)
    minao = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % minao_name), molecule=molecule)

    uhf = cuest_scf.UHF(
        cuest_handle=cuest_handle,
        molecule=molecule,
        charge=0,
        multiplicity=2,
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
    uhf.solve()
    return uhf.compute_energy(), uhf.compute_gradient().to_numpy()


@pytest.mark.parametrize('incremental_fock', [False, True], ids=['full', 'incremental'])
@pytest.mark.parametrize('functional,primary_name', CASES)
def test_uhf_directjk(functional, primary_name, incremental_fock):
    E, G = run_directjk(functional, primary_name, incremental_fock)
    E_ref, G_ref = REFERENCES[(functional, primary_name)]
    dE = abs(E - E_ref)
    dG = float(np.max(np.abs(G - np.asarray(G_ref))))
    print(f'{functional}/{primary_name}: dE = {dE:.3e}  dG = {dG:.3e}')
    # Observed agreement: dE <= 1.1e-8 (wB97M-V; every other case <= 5e-9),
    # dG <= 3.8e-8 (HF/sto-3g).  The gates carry a ~3x margin over the worst
    # case.
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


def test_uhf_directjk_rejects_negative_rebuild():
    """A negative incremental_fock_rebuild would read as the never-rebuild
    period (every build after the first a delta build, screening error
    accumulating unbounded); the constructor rejects it."""
    functional, primary_name = CASES[0]
    with pytest.raises(ValueError, match='incremental_fock_rebuild must be >= 0'):
        run_directjk(functional, primary_name, True, incremental_fock_rebuild=-1)
