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
        xc_functional_name='B3LYP1',
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

def test_b3lyp1():

    import numpy as np

    E2 = -1.7706120600238987E+03 # External code

    # External code we trust
    G2 = np.array([
        [ -0.0075339,   -0.0084374,    0.0100722,],
        [ -0.0129560,    0.0085460,    0.0005812,],
        [  0.0044460,    0.0022805,    0.0010485,],
        [ -0.0020780,   -0.0035621,   -0.0209380,],
        [  0.0085310,    0.0155217,    0.0060141,],
        [ -0.0078534,   -0.0143469,   -0.0203861,],
        [ -0.0039985,    0.0014855,    0.0145814,],
        [  0.0129959,   -0.0260229,   -0.0206831,],
        [  0.0072781,    0.0073819,    0.0204639,],
        [  0.0376327,   -0.0016436,   -0.0066890,],
        [  0.0007142,    0.0060887,   -0.0293362,],
        [  0.0132803,   -0.0171365,    0.0107301,],
        [ -0.0079049,    0.0167075,    0.0061632,],
        [ -0.0049003,   -0.0178073,    0.0007884,],
        [ -0.0050590,    0.0118507,    0.0137120,],
        [  0.0172510,   -0.0082200,   -0.0076180,],
        [  0.0083788,   -0.0033765,   -0.0019021,],
        [ -0.0035890,    0.0030032,    0.0026220,],
        [  0.0043617,   -0.0019763,   -0.0035190,],
        [ -0.0268906,    0.0019632,   -0.0069977,],
        [ -0.0186261,    0.0274140,    0.0066706,],
        [  0.0017896,   -0.0036556,    0.0113431,],
        [ -0.0113727,   -0.0156116,    0.0042445,],
        [ -0.0052975,    0.0008700,   -0.0042885,],
        [ -0.0055663,   -0.0056535,    0.0005495,],
        [  0.0050889,    0.0027446,   -0.0019316,],
        [ -0.0003030,    0.0040000,   -0.0006066,],
        [ -0.0031004,   -0.0063099,   -0.0115974,],
        [ -0.0066769,    0.0058210,    0.0058223,],
        [ -0.0351842,    0.0081270,    0.0334958,],
        [ -0.0159176,    0.0189273,   -0.0111036,],
        [  0.0025764,    0.0080068,    0.0079594,],
        [  0.0013813,   -0.0236045,   -0.0069523,],
        [  0.0318463,   -0.0078926,   -0.0303061,],
        [ -0.0009451,   -0.0115094,    0.0105833,],
        [  0.0008331,    0.0022270,    0.0000064,],
        [  0.0019510,    0.0046119,   -0.0002960,],
        [ -0.0029637,   -0.0033469,    0.0003911,],
        [  0.0040628,    0.0045491,    0.0016492,],
        [ -0.0008783,   -0.0055257,    0.0018305,],
        [ -0.0012048,    0.0001091,   -0.0009157,],
        [  0.0015754,   -0.0009992,   -0.0017813,],
        [  0.0009701,    0.0001342,    0.0012882,],
        [  0.0001547,    0.0015543,    0.0007688,],
        [ -0.0010241,   -0.0002313,   -0.0017558,],
        [  0.0012272,   -0.0006063,    0.0008811,],
        [  0.0015072,    0.0000462,   -0.0046693,],
        [ -0.0014937,   -0.0028328,    0.0087670,],
        [  0.0046354,    0.0016553,   -0.0012502,],
        [  0.0014366,    0.0047094,    0.0063387,],
        [ -0.0022968,    0.0000738,    0.0005287,],
        [  0.0039856,    0.0014382,    0.0012495,],
        [  0.0060950,    0.0005016,   -0.0011365,],
        [ -0.0008533,    0.0031655,   -0.0023299,],
        [ -0.0007804,    0.0005323,    0.0018985,],
        [ -0.0002041,   -0.0011274,   -0.0012298,],
        [ -0.0009716,   -0.0005681,   -0.0004076,],
        [  0.0029277,    0.0017253,   -0.0028413,],
        [  0.0021441,   -0.0008873,    0.0014046,],
        [  0.0032778,    0.0023101,    0.0021565,],
        [ -0.0008991,    0.0014957,    0.0035126,],
        [  0.0002489,    0.0004832,   -0.0023380,],
        [ -0.0015628,   -0.0015595,   -0.0014371,],
        [  0.0024810,   -0.0005167,   -0.0010865,],
        [  0.0024702,    0.0014808,    0.0032646,],
        [ -0.0018446,    0.0033328,    0.0073731,],
        [  0.0031945,    0.0080925,   -0.0024241,],
        ])

    E1, G1 = run_rhf(primary_name='def2-tzvp')
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)
