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

import cuest_scf

import os
filedir = os.path.dirname(os.path.realpath(__file__))
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

cuest_handle = cuest_scf.CuestHandle()

# Use unpruned grids matching the PySCF atom grids used for the reference
# gradients below: the XC grid is (99,590) and the NLC (VV10) grid is (75,302).
# The pruned GRID/SG families cap the radial count well below 99, which leaves
# the grid-hungry SCAN family (r2SCAN) under-converged relative to the PySCF
# reference. The UNPRUNED grid family takes the (n_radial, n_angular) pair
# directly and applies the GRID family's Treutler-Ahlrichs radial quadrature
# with a flat n_angular at every radial node (see run_uhf() below).

def run_uhf(
    *,
    functional_name,
    ):

    # Reasonable default - as coarse as 1.0E-10 yields high-precision results
    # Coarser can save some CoreDF memory and lower runtime
    threshold_pq = 1.0E-18

    molecule_filename = os.path.join(filedir, 'ch3_radical.xyz')

    molecule = cuest_scf.Molecule.parse_from_xyz_file(molecule_filename)
    # Open-shell methyl radical: neutral, one unpaired electron (doublet).
    charge = 0
    multiplicity = 2

    primary_name = 'def2-tzvp'
    auxiliary_name = 'def2-universal-jkfit'
    minao_name = 'minao-1'

    primary_filename = os.path.join(basisdir, '%s.gbs' % (primary_name))
    auxiliary_filename = os.path.join(basisdir, '%s.gbs' % (auxiliary_name))
    minao_filename = os.path.join(basisdir, '%s.gbs' % (minao_name))

    primary = cuest_scf.AOBasis.parse_from_gbs_file(primary_filename, molecule=molecule)
    auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(auxiliary_filename, molecule=molecule)
    minao = cuest_scf.AOBasis.parse_from_gbs_file(minao_filename, molecule=molecule)

    uhf = cuest_scf.UHF(
        cuest_handle=cuest_handle,
        molecule=molecule,
        charge=charge,
        multiplicity=multiplicity,
        primary=primary,
        xc_functional_name=functional_name,
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=threshold_pq,
        xc_grid_family='UNPRUNED',
        xc_grid_level=(99, 590),    # unpruned (99 radial x 590 angular) XC grid
        nlc_grid_family='UNPRUNED',
        nlc_grid_level=(75, 302),   # unpruned (75 radial x 302 angular) NLC grid
        g_convergence=1.0e-8,
        maxiter=300,
        print_level=0,
        )

    uhf.solve()
    G = uhf.compute_gradient()

    return uhf.scalars['Escf'], G.to_numpy()


def test_uhf_dft_functionals():

    import numpy as np

    # Core set of exchange-correlation functionals, mirroring the RHF sweep in
    # dft_energies/test_dft_gradients.py.
    #
    # Reference total SCF energies (Hartree) and nuclear gradients (Hartree/Bohr)
    # for the methyl radical (CH3, doublet) / def2-tzvp setup in run_uhf(),
    # generated with PySCF >= 2.13 UKS using density fitting
    # (def2-universal-jkfit). To reproduce cuEST's quadrature exactly, PySCF was
    # configured with:
    #   * unpruned (99,590) XC grid and unpruned (75,302) NLC (VV10) grid,
    #   * small_rho_cutoff = 0 (no small-density screening),
    #   * Stratmann-Scuseria-Frisch (SSF) cell partitioning with NO atomic-radii
    #     adjustment (grids.becke_scheme = stratmann, grids.radii_adjust = None)
    #     -- this is the partitioning cuEST uses,
    #   * grid_response = True on the gradient.
    # PySCF >= 2.13 is REQUIRED: it added grid_response support for the Stratmann
    # scheme. With 2.12.x the XC gradient weight-derivatives were computed with
    # the original-Becke switching function regardless of becke_scheme, making
    # the grid response inconsistent with the SSF weights used in the energy and
    # producing a spurious ~2e-6 gradient error. With the fix, cuEST and PySCF
    # agree to ~1e-8 in the gradient -- the same level as the energies (verified
    # independently: a finite-difference of cuEST's own total energy reproduces
    # cuEST's analytic gradient to ~1e-9). cuEST is run on the same grids (see
    # the UNPRUNED grids in run_uhf above). The gradient rows follow the atom order in
    # ch3_radical.xyz: C, H, H, H.
    #
    # The exact PySCF script used to generate the reference_values and
    # reference_gradients below (run with PySCF >= 2.13):
    #
    #   import numpy as np
    #   from pyscf import gto, dft, scf
    #   import pyscf
    #   assert tuple(int(x) for x in pyscf.__version__.split('.')[:2]) >= (2, 13), \
    #       'Need PySCF >= 2.13 for Stratmann grid_response'
    #
    #   # cuEST functional name : pyscf functional name
    #   tasks = {
    #       'HF'        : None,
    #       'B3LYP1'    : 'b3lypg',
    #       'B3LYP5'    : 'b3lyp5',
    #       'B97'       : 'b97',
    #       'BLYP'      : 'blyp',
    #       'M06-L'     : 'm06l',
    #       'PBE'       : 'pbe',
    #       'PBE0'      : 'pbe0',
    #       'SVWN5'     : 'svwn',
    #       'LC-wPBE'   : 'HYB_GGA_XC_LC_WPBE_WHS',  # cuEST uses the WHS variant
    #       'wB97X'     : 'wb97x',
    #       'wB97X-V'   : 'wb97x-v',
    #       'wB97M-V'   : 'wb97m-v',
    #       'r2SCAN'    : 'r2scan',
    #       'B97M-V'    : 'b97m-v',
    #       'LC-wPBEh'  : 'HYB_GGA_XC_LC_WPBEH_WHS',
    #       'CAM-B3LYP' : 'camb3lyp',
    #       'HSE06'     : 'HYB_GGA_XC_HSE06',
    #       'M06'       : 'HYB_MGGA_X_M06,MGGA_C_M06',
    #       'M06-2X'    : 'HYB_MGGA_X_M06_2X,MGGA_C_M06_2X',
    #   }
    #
    #   xyz = '''
    #   C    0.000000    0.000000    0.000000
    #   H    1.079000    0.000000    0.000000
    #   H   -0.539500    0.934442    0.000000
    #   H   -0.539500   -0.934442    0.000000
    #   '''
    #
    #   mol = gto.M(atom=xyz, unit='Angstrom', basis='def2-tzvp',
    #               charge=0, spin=1)   # spin=1 -> multiplicity 2 (one unpaired e-)
    #   auxbasis = 'def2-universal-jkfit'
    #
    #   energies, gradients = {}, {}
    #   for cuest_name, pyscf_name in tasks.items():
    #       if pyscf_name is None:
    #           mf = scf.UHF(mol).density_fit(auxbasis=auxbasis)
    #       else:
    #           mf = dft.UKS(mol, xc=pyscf_name).density_fit(auxbasis=auxbasis)
    #           # Match cuEST's grid exactly: unpruned (99,590), no small-density
    #           # screening, and Stratmann-Scuseria-Frisch (SSF) partitioning with
    #           # NO atomic-radii adjustment (cuEST uses unmodified SSF cells).
    #           mf.grids.atom_grid = (99, 590)
    #           mf.grids.prune = None
    #           mf.small_rho_cutoff = 0.0
    #           mf.grids.becke_scheme = dft.gen_grid.stratmann
    #           mf.grids.radii_adjust = None
    #           # VV10 NLC grid (used only by NLC functionals; harmless otherwise)
    #           mf.nlcgrids.atom_grid = (75, 302)
    #           mf.nlcgrids.prune = None
    #           mf.nlcgrids.becke_scheme = dft.gen_grid.stratmann
    #           mf.nlcgrids.radii_adjust = None
    #       mf.conv_tol = 1e-12
    #       e = mf.kernel()
    #       g = mf.nuc_grad_method()
    #       if pyscf_name is not None:
    #           g.grid_response = True   # include grid-weight-derivative terms
    #       grad = g.kernel()
    #       energies[cuest_name], gradients[cuest_name] = e, grad
    #
    #   np.set_printoptions(precision=16, floatmode='unique', linewidth=120)
    #   for k in tasks: print("%-11s: %.15e," % ("'%s'" % k, energies[k]))
    #   for k in tasks: print("%-11s: np.array(%s)," % ("'%s'" % k,
    #       np.array2string(gradients[k], separator=', ', max_line_width=120)))
    reference_values = {
        'HF'       : -3.957770405107397e+01,
        'B3LYP1'   : -3.985879937118200e+01,
        'B3LYP5'   : -3.982655165116544e+01,
        'B97'      : -3.983817235359978e+01,
        'BLYP'     : -3.982747097350686e+01,
        'M06-L'    : -3.984466669733771e+01,
        'PBE'      : -3.978802010300634e+01,
        'PBE0'     : -3.979905797045407e+01,
        'SVWN5'    : -3.944177930339185e+01,
        'LC-wPBE'  : -3.982625075868993e+01,
        'wB97X'    : -3.983792926422222e+01,
        'wB97X-V'  : -3.983580011615681e+01,
        'wB97M-V'  : -3.981761350277876e+01,
        'r2SCAN'   : -3.982793073517892e+01,
        'B97M-V'   : -3.984994274669875e+01,
        'LC-wPBEh' : -3.982832713477232e+01,
        'CAM-B3LYP': -3.982726071505179e+01,
        'HSE06'    : -3.980302403976527e+01,
        'M06'      : -3.981477081404821e+01,
        'M06-2X'   : -3.982405195085334e+01,
    }

    reference_gradients = {
        'HF'       : np.array([[ 2.8903534238155212e-07, -4.7289209227369503e-16, -6.9724036363904812e-17],
        [ 5.4307774035613487e-03,  1.5507972086297111e-16,  2.7609982652542132e-17],
        [-2.7155332194522819e-03,  4.7035476687764532e-03,  7.5212221775113621e-16],
        [-2.7155332194523929e-03, -4.7035476687782296e-03, -6.5826134396917615e-16]]),
        'B3LYP1'   : np.array([[ 1.5047480947765453e-06,  1.1565824245641657e-15,  3.5182763531213369e-16],
        [ 7.9400172051657947e-05, -1.6485257997995424e-16, -7.7267403057481119e-17],
        [-4.0452460071893626e-05,  7.0159609824482772e-05, -2.0497810848051720e-16],
        [-4.0452460070672380e-05, -7.0159609825370950e-05, -8.6434475462301853e-17]]),
        'B3LYP5'   : np.array([[ 1.4981222333874589e-06, -5.3199829113703914e-17, -3.0472001120930528e-16],
        [-1.5119566525334527e-04,  9.3433697876610118e-17, -2.6306649760030440e-17],
        [ 7.4848771510427348e-05, -1.2954720471736181e-04,  2.6732982463896724e-16],
        [ 7.4848771509761214e-05,  1.2954720471469727e-04,  4.4525890984811822e-17]]),
        'B97'      : np.array([[ 1.6337885979557149e-06, -2.4498070326049481e-17, -1.1357249314632462e-16],
        [-1.5013728500434809e-03, -5.3322008260131484e-17, -4.6146617650958241e-17],
        [ 7.4986953072153639e-04, -1.2986950086819604e-03,  3.8072872374510136e-17],
        [ 7.4986953072053719e-04,  1.2986950086797400e-03,  1.3582775537271503e-16]]),
        'BLYP'     : np.array([[ 1.7871833473889313e-06,  2.1893473294213639e-15, -5.2004982836677795e-17],
        [-4.1177314723945813e-03, -2.7209406886897315e-16, -1.1405990908708883e-16],
        [ 2.0579721445179722e-03, -3.5644110073356217e-03,  4.0681801931729693e-16],
        [ 2.0579721445204147e-03,  3.5644110073325130e-03, -2.4567336671739135e-16]]),
        'M06-L'    : np.array([[ 4.1757123557600746e-06,  1.8798453623835795e-15, -3.1851992578826754e-17],
        [ 1.0554988727755621e-03, -2.3697178222222059e-16, -1.0245964455301129e-16],
        [-5.2983729256417256e-04,  9.2175671638217160e-04, -3.7766908338109151e-16],
        [-5.2983729256628198e-04, -9.2175671638417001e-04,  5.0987328808888470e-16]]),
        'PBE'      : np.array([[ 1.8503651966376795e-06, -1.1459824853612503e-15, -1.9103992829851016e-16],
        [-5.4642751734490513e-03,  4.5777769007324202e-16, -6.0688692851277590e-18],
        [ 2.7312124041299901e-03, -4.7305266404000790e-03,  1.3234112893713458e-15],
        [ 2.7312124041284358e-03,  4.7305266403969704e-03, -1.1379610315221705e-15]]),
        'PBE0'     : np.array([[ 1.4974826382627271e-06,  1.5049654873923636e-15, -3.1620529356049443e-16],
        [-7.2457959031124730e-04, -1.2155988320070991e-16,  1.4138989092710868e-17],
        [ 3.6154105383801038e-04, -6.2613616987117204e-04,  1.9411893131481897e-15],
        [ 3.6154105383567892e-04,  6.2613616986761933e-04, -1.6331388736032190e-15]]),
        'SVWN5'    : np.array([[ 1.6650615995598681e-06, -1.6999167882224558e-16, -4.8631554181045550e-16],
        [-7.3625584056242754e-03,  7.0841685733087552e-17,  1.2337913020300039e-16],
        [ 3.6804466720117279e-03, -6.3746933835207020e-03,  3.4029675489127744e-16],
        [ 3.6804466720117279e-03,  6.3746933835202579e-03,  2.8318061541202186e-17]]),
        'LC-wPBE'  : np.array([[ 1.5576393056837359e-06, -2.9091124812350241e-15, -9.6847740884905705e-17],
        [-7.4160548006463323e-04, -1.5186986250186306e-16,  2.8571247085733324e-17],
        [ 3.7002392038376630e-04, -6.4080206316563526e-04, -4.7056943017712242e-16],
        [ 3.7002392038565368e-04,  6.4080206316341481e-04,  4.6365197673736358e-16]]),
        'wB97X'    : np.array([[ 2.2684173086101431e-06,  4.2042147263048720e-10, -5.1072387227114232e-16],
        [-3.4556375219318980e-04,  7.8007660899579929e-12, -6.7345021602501685e-18],
        [ 1.7164726675777953e-04, -2.9744843495027595e-04, -1.4683473034642178e-16],
        [ 1.7164734373853463e-04,  2.9744857602276298e-04,  6.8275428361391885e-16]]),
        'wB97X-V'  : np.array([[ 1.4167943246929325e-06, -4.4554180228016394e-10, -2.3179202072561275e-17],
        [-1.6547899379610875e-03,  5.6355040083291219e-11, -4.4945585150145962e-17],
        [ 8.2668636721305866e-04, -1.4317604165023923e-03, -7.7292857810116958e-16],
        [ 8.2668630393167852e-04,  1.4317602810449692e-03,  8.8263230005811044e-16]]),
        'wB97M-V'  : np.array([[ 1.3802917201678292e-06, -4.3324705653305368e-10,  9.8714465700045758e-17],
        [ 1.0568981131644506e-03,  8.6000247879375794e-11, -4.1958796786270198e-17],
        [-5.2913928444286995e-04,  9.1667124099625852e-04, -7.3827493878204793e-16],
        [-5.2913925633790715e-04, -9.1667139196482950e-04,  6.8901813360281921e-16]]),
        'r2SCAN'   : np.array([[-4.5704536211471964e-07, -2.7973477013701283e-15,  6.9310271173240153e-16],
        [-4.6172146287748461e-04,  4.9230517011772920e-16, -1.4850732997491660e-16],
        [ 2.3108925411785552e-04, -4.0124719279410748e-04, -2.1513976084052942e-16],
        [ 2.3108925411541303e-04,  4.0124719279432952e-04, -3.3474159169851565e-16]]),
        'B97M-V'   : np.array([[ 1.2571001889836425e-06,  2.1483674817985176e-15,  3.0236765831943129e-18],
        [ 3.5837983557844844e-03, -4.7311381458586855e-16, -7.1181385366762625e-17],
        [-1.7925277279867702e-03,  3.1050232303682090e-03,  1.9105489028316865e-15],
        [-1.7925277279873253e-03, -3.1050232303677650e-03, -1.8484776413597347e-15]]),
        'LC-wPBEh' : np.array([[ 1.2823632724459753e-06,  2.7400527752394953e-15,  4.1876959132156770e-18],
        [ 2.6182066519688796e-03, -2.6189702362668578e-16, -6.4274798092577638e-17],
        [-1.3097445076234449e-03,  2.2686353655434477e-03,  7.6370845721375484e-16],
        [-1.3097445076241110e-03, -2.2686353655436697e-03, -6.8703409520462332e-16]]),
        'CAM-B3LYP': np.array([[ 1.3807357140692336e-06, -3.1443267607420390e-16,  3.9181016263527596e-16],
        [ 5.1008875654590113e-04, -5.0021953138286738e-17, -3.9592350487319251e-17],
        [-2.5573474612305969e-04,  4.4305004483469190e-04, -6.5699195064697549e-16],
        [-2.5573474612361480e-04, -4.4305004483358168e-04,  3.6253127774995656e-16]]),
        'HSE06'    : np.array([[ 1.3667343855063733e-06,  1.8995080871500352e-15,  1.9831209681911190e-16],
        [-4.9431747080541655e-04, -2.8190151710539903e-16, -4.4305161687964463e-18],
        [ 2.4647536820698868e-04, -4.2677714144567780e-04, -2.2684003160134923e-16],
        [ 2.4647536820476823e-04,  4.2677714144678802e-04,  5.0340479706485634e-17]]),
        'M06'      : np.array([[ 3.9174949507038504e-06, -1.3866988825801834e-15, -1.5241727759727647e-15],
        [ 2.3407146529530465e-04,  2.0366229418789183e-16,  1.3703604744724668e-16],
        [-1.1899448012231417e-04,  2.0615017732183993e-04, -1.0419441302192108e-15],
        [-1.1899448012253622e-04, -2.0615017732006358e-04,  2.4209638780866821e-15]]),
        'M06-2X'   : np.array([[-2.5554448408339456e-07, -7.2750107888175120e-16,  2.1754924883516384e-16],
        [ 1.2829307330382012e-03,  2.6809798128009813e-16, -7.0654156201445121e-17],
        [-6.4133759427764669e-04,  1.1100144077507146e-03,  9.4135370036608582e-16],
        [-6.4133759427820181e-04, -1.1100144077484941e-03, -1.1002683081702434e-15]]),
    }

    # Per-functional tolerances (cuEST vs PySCF >= 2.13, on matched grids -- same
    # quadrature, same SSF partitioning, consistent grid response). Both the
    # ENERGIES and the GRADIENTS agree to ~1e-8 or better for every non-range-
    # separated functional (cross-checked against a finite-difference of cuEST's
    # own energy, which reproduces cuEST's analytic gradient to ~1e-9). Pure HF
    # (no XC grid) agrees to ~2e-9 in the gradient.
    #
    # The outliers are the range-separated hybrids (LC-wPBE, wB97X, wB97X-V,
    # wB97M-V, LC-wPBEh, CAM-B3LYP, HSE06): their residual (~1e-6 in E, up to
    # ~3e-5 in G) lives in the density-fitted long-range erf(omega*r)/r exchange
    # -- it shows up in BOTH the energy and the gradient and is unrelated to the
    # grid (cuEST's DF of the attenuated exchange is ~1e-5 less accurate than the
    # exact/PySCF attenuated-metric RI; see the investigation notes for CAM-B3LYP).
    # NOTE: cuEST's LC-wPBE is the Weintraub-Henderson-Scuseria variant
    # (HYB_GGA_XC_LC_WPBE_WHS), NOT the Vydrov-Scuseria HYB_GGA_XC_LC_WPBE.
    # Tolerances carry several-x margin over the observed deviation while staying
    # inside GPU run-to-run noise.
    energy_tolerances = {
        'HF'       : 1.0E-8,   # observed ~1e-10 (no XC grid)
        'B3LYP1'   : 1.0E-8,   # observed ~2e-12
        'B3LYP5'   : 1.0E-8,   # observed ~4e-12
        'B97'      : 1.0E-8,   # observed ~4e-11
        'BLYP'     : 1.0E-8,   # observed ~1e-10
        'M06-L'    : 1.0E-8,   # observed ~3e-11 (meta-GGA)
        'PBE'      : 1.0E-8,   # observed ~1e-10
        'PBE0'     : 1.0E-8,   # observed ~2e-11
        'SVWN5'    : 1.0E-8,   # observed ~2e-10
        'LC-wPBE'  : 1.0E-5,   # observed ~2e-6 (range-separated exchange)
        'wB97X'    : 2.0E-5,   # observed ~4e-6 (range-separated exchange)
        'wB97X-V'  : 2.0E-5,   # observed ~5e-6 (range-separated exchange + VV10)
        'wB97M-V'  : 2.0E-5,   # observed ~6e-6 (range-separated exchange + VV10)
        'r2SCAN'   : 1.0E-8,   # observed ~1e-11 (meta-GGA)
        'B97M-V'   : 1.0E-7,   # observed ~1e-8 (meta-GGA + VV10 NLC)
        'LC-wPBEh' : 1.0E-5,   # observed ~1e-6 (range-separated exchange)
        'CAM-B3LYP': 1.0E-5,   # observed ~3e-6 (range-separated exchange)
        'HSE06'    : 2.0E-5,   # observed ~7e-6 (screened range-separated exchange)
        'M06'      : 1.0E-8,   # observed ~7e-12 (meta-GGA hybrid)
        'M06-2X'   : 1.0E-8,   # observed ~3e-11 (meta-GGA hybrid)
    }
    gradient_tolerances = {
        'HF'       : 1.0E-7,   # observed ~2e-9 (no XC grid)
        'B3LYP1'   : 1.0E-7,   # observed ~3e-9
        'B3LYP5'   : 1.0E-7,   # observed ~3e-9
        'B97'      : 1.0E-7,   # observed ~3e-9
        'BLYP'     : 1.0E-7,   # observed ~3e-9
        'M06-L'    : 1.0E-7,   # observed ~1e-8 (meta-GGA)
        'PBE'      : 1.0E-7,   # observed ~1e-9
        'PBE0'     : 1.0E-7,   # observed ~4e-9
        'SVWN5'    : 1.0E-7,   # observed ~2e-8
        'LC-wPBE'  : 1.0E-5,   # observed ~5e-6 (range-separated exchange)
        'wB97X'    : 1.0E-4,   # observed ~2e-5 (range-separated exchange)
        'wB97X-V'  : 1.0E-4,   # observed ~2e-5 (range-separated exchange + VV10)
        'wB97M-V'  : 1.0E-4,   # observed ~2e-5 (range-separated exchange + VV10)
        'r2SCAN'   : 1.0E-7,   # observed ~9e-9 (meta-GGA)
        'B97M-V'   : 1.0E-7,   # observed ~9e-10 (meta-GGA + VV10 NLC)
        'LC-wPBEh' : 1.0E-5,   # observed ~3e-6 (range-separated exchange)
        'CAM-B3LYP': 1.0E-5,   # observed ~3e-6 (range-separated exchange)
        'HSE06'    : 1.0E-4,   # observed ~3e-5 (screened range-separated exchange)
        'M06'      : 1.0E-7,   # observed ~4e-9 (meta-GGA hybrid)
        'M06-2X'   : 1.0E-7,   # observed ~4e-9 (meta-GGA hybrid)
    }

    print()
    failures = []
    for functional_name, E2 in reference_values.items():

        G2 = reference_gradients[functional_name]
        E1, G1 = run_uhf(functional_name=functional_name)
        dE = abs(E1 - E2)
        dG = np.max(np.abs(G1 - G2))
        tolE = energy_tolerances[functional_name]
        tolG = gradient_tolerances[functional_name]
        print('%10s dE %10.3e (tol %.0e) dG %10.3e (tol %.0e)'
              % (functional_name, dE, tolE, dG, tolG))
        if dE >= tolE:
            failures.append('%s: dE=%.3e exceeds tol=%.3e' % (functional_name, dE, tolE))
        if dG >= tolG:
            failures.append('%s: dG=%.3e exceeds tol=%.3e' % (functional_name, dG, tolG))

    assert not failures, '\n'.join(failures)
