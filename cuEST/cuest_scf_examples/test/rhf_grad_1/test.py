# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

def run_rhf(
    *,
    primary_name,
    ):

    # Reasonable default - as coarse as 1.0E-10 yields high-precision results
    # Coarser can save some CoreDF memory and lower runtime
    threshold_pq = 1.0E-14

    molecule_filename = os.path.join(filedir, 'paxlovid.xyz')
    
    molecule = cuest_scf.Molecule.parse_from_xyz_file(molecule_filename)
    charge = 0
    
    auxiliary_name = 'def2-universal-jkfit'
    minao_name = 'minao-1'
    
    primary_filename = os.path.join(basisdir, '%s.gbs' % (primary_name))
    auxiliary_filename = os.path.join(basisdir, '%s.gbs' % (auxiliary_name))
    minao_filename = os.path.join(basisdir, '%s.gbs' % (minao_name))
    
    primary = cuest_scf.AOBasis.parse_from_gbs_file(primary_filename, molecule=molecule)
    auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(auxiliary_filename, molecule=molecule)
    minao = cuest_scf.AOBasis.parse_from_gbs_file(minao_filename, molecule=molecule)
    
    rhf = cuest_scf.RHF(
        cuest_handle=cuest_handle,
        molecule=molecule,
        charge=charge,
        xc_functional_name='HF',
        primary=primary,
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=threshold_pq,
        g_convergence=1.0E-8, # Tighter for gradient check
        )
    
    rhf.solve()

    E = rhf.compute_energy()

    G = rhf.compute_gradient()

    return E, G

def test_rhf():

    import numpy as np

    E2 = -1.7602954939152182E+03 # Internal reference
    E3 = -1.7602954939140413E+03 # External code

    # External code we trust
    G2 = np.array([
       [ -0.0208597,   -0.0356780,    0.0281067],
       [ -0.0408624,    0.0282052,   -0.0084045],
       [  0.0253546,    0.0218243,    0.0242479],
       [  0.0090637,   -0.0044771,   -0.0618983],
       [  0.0344967,    0.0323332,    0.0372294],
       [ -0.0262053,   -0.0418410,   -0.0582010],
       [ -0.0098429,    0.0126417,    0.0546539],
       [  0.0112808,   -0.0335221,   -0.0275379],
       [  0.0127417,    0.0098751,    0.0221619],
       [  0.0508128,   -0.0012480,   -0.0168725],
       [  0.0014983,    0.0048573,   -0.0411949],
       [  0.0477820,   -0.0621228,    0.0412027],
       [ -0.0050371,    0.0225184,    0.0126377],
       [ -0.0116681,   -0.0265052,    0.0049367],
       [ -0.0106237,    0.0190111,    0.0112511],
       [  0.0227971,   -0.0156068,   -0.0105452],
       [  0.0121413,   -0.0101201,   -0.0016905],
       [ -0.0030149,    0.0082268,    0.0034705],
       [  0.0085837,   -0.0041588,   -0.0078855],
       [ -0.0434603,    0.0007589,    0.0141763],
       [ -0.0390102,    0.0286030,   -0.0122543],
       [  0.0058979,   -0.0037244,    0.0162883],
       [ -0.0113994,   -0.0164766,    0.0029828],
       [  0.0019021,   -0.0053864,    0.0070392],
       [ -0.0095757,   -0.0090207,   -0.0020937],
       [  0.0083523,    0.0004829,   -0.0004781],
       [ -0.0029454,    0.0059150,    0.0021849],
       [ -0.0057341,   -0.0055926,   -0.0192969],
       [ -0.0032889,    0.0036989,    0.0063679],
       [ -0.0385153,    0.0253514,    0.0658022],
       [ -0.0526046,    0.0668111,   -0.0438756],
       [  0.0006937,    0.0094747,    0.0103117],
       [  0.0037456,   -0.0390302,   -0.0276055],
       [  0.0532312,   -0.0174919,   -0.0588797],
       [  0.0006649,   -0.0157475,    0.0113979],
       [  0.0055180,   -0.0021850,   -0.0012850],
       [  0.0032198,    0.0022361,   -0.0062879],
       [ -0.0051346,   -0.0025929,    0.0084412],
       [  0.0064862,    0.0081799,   -0.0042450],
       [ -0.0085572,   -0.0029933,    0.0030293],
       [ -0.0064755,   -0.0008781,   -0.0026522],
       [  0.0047295,   -0.0035292,   -0.0056022],
       [  0.0018127,   -0.0028931,    0.0059839],
       [  0.0005288,    0.0079685,    0.0022861],
       [ -0.0063595,   -0.0008092,   -0.0032737],
       [  0.0018779,   -0.0028706,    0.0056260],
       [ -0.0043175,   -0.0031250,   -0.0122338],
       [ -0.0060574,   -0.0030624,    0.0232496],
       [  0.0064236,    0.0022386,   -0.0104025],
       [  0.0067458,    0.0127649,    0.0191777],
       [ -0.0055109,    0.0014812,    0.0051545],
       [  0.0061676,    0.0063676,   -0.0008665],
       [  0.0125817,   -0.0030474,    0.0010893],
       [  0.0012520,    0.0080011,   -0.0052251],
       [ -0.0048426,    0.0027812,    0.0058644],
       [ -0.0031752,   -0.0037730,   -0.0049010],
       [ -0.0037059,   -0.0042413,   -0.0042426],
       [  0.0051362,    0.0057527,   -0.0052669],
       [  0.0063753,   -0.0033135,    0.0032430],
       [  0.0093046,    0.0031410,    0.0034192],
       [ -0.0035042,    0.0025311,    0.0095088],
       [ -0.0073677,   -0.0005880,   -0.0040605],
       [ -0.0038629,   -0.0053111,   -0.0056777],
       [  0.0082404,   -0.0021031,   -0.0003102],
       [  0.0077767,    0.0035413,   -0.0001143],
       [ -0.0078661,    0.0050328,    0.0046209],
       [  0.0061681,    0.0224595,   -0.0017824],
        ])

    E1, G1 = run_rhf(primary_name='def2-tzvp')
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)
