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
        xc_functional_name='B97M-V',
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

def test_b97mv():

    import numpy as np

    E2 = -1.7704492618735078E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0106596,   -0.0156799,    0.0145414,],
        [-0.0197646,    0.0137893,   -0.0018952,],
        [ 0.0088908,    0.0068987,    0.0064756,],
        [ 0.0006182,   -0.0042745,   -0.0271809,],
        [ 0.0114778,    0.0174686,    0.0108210,],
        [-0.0112086,   -0.0174245,   -0.0243939,],
        [-0.0049056,    0.0039996,    0.0214965,],
        [ 0.0131420,   -0.0238833,   -0.0197356,],
        [ 0.0069684,    0.0060010,    0.0215369,],
        [ 0.0360612,   -0.0017413,   -0.0056847,],
        [ 0.0010948,    0.0105423,   -0.0295461,],
        [ 0.0130757,   -0.0167972,    0.0104620,],
        [-0.0056738,    0.0133255,    0.0062157,],
        [-0.0053860,   -0.0188959,    0.0008743,],
        [-0.0062759,    0.0098235,    0.0106342,],
        [ 0.0172062,   -0.0082411,   -0.0046346,],
        [ 0.0057534,   -0.0003341,   -0.0062477,],
        [-0.0046297,   -0.0011964,    0.0020192,],
        [ 0.0010280,   -0.0005629,   -0.0000546,],
        [-0.0255061,    0.0011339,   -0.0057105,],
        [-0.0185690,    0.0252412,    0.0065229,],
        [-0.0000765,   -0.0033603,    0.0102771,],
        [-0.0117633,   -0.0154561,    0.0034439,],
        [-0.0055208,    0.0012165,   -0.0067281,],
        [-0.0032842,   -0.0025623,    0.0030395,],
        [ 0.0018487,    0.0056124,   -0.0044792,],
        [ 0.0033705,    0.0025224,   -0.0046141,],
        [-0.0008305,   -0.0063373,   -0.0079543,],
        [-0.0104291,    0.0039389,    0.0059832,],
        [-0.0324778,    0.0085888,    0.0316638,],
        [-0.0110840,    0.0127289,   -0.0076844,],
        [ 0.0049426,    0.0048814,    0.0045765,],
        [ 0.0016614,   -0.0221686,   -0.0066648,],
        [ 0.0301663,   -0.0081270,   -0.0294829,],
        [-0.0003956,   -0.0099346,    0.0050353,],
        [ 0.0021097,    0.0007955,    0.0002088,],
        [ 0.0021282,    0.0038946,   -0.0025163,],
        [-0.0026446,   -0.0039521,    0.0011721,],
        [ 0.0046033,    0.0046392,   -0.0000268,],
        [-0.0021881,   -0.0055075,    0.0015075,],
        [-0.0036517,   -0.0008585,   -0.0017543,],
        [ 0.0028364,   -0.0026402,   -0.0033413,],
        [ 0.0013393,   -0.0015564,    0.0034227,],
        [-0.0000501,    0.0026277,    0.0016033,],
        [-0.0034870,   -0.0006808,   -0.0022617,],
        [ 0.0014405,   -0.0017353,    0.0032705,],
        [ 0.0013591,   -0.0009285,   -0.0052868,],
        [-0.0025191,   -0.0031309,    0.0117281,],
        [ 0.0056894,    0.0010170,   -0.0023023,],
        [ 0.0019322,    0.0061851,    0.0088318,],
        [-0.0031094,    0.0008149,    0.0022319,],
        [ 0.0049323,    0.0034812,    0.0009759,],
        [ 0.0072225,    0.0006121,   -0.0003196,],
        [-0.0002752,    0.0054103,   -0.0035896,],
        [-0.0022374,    0.0017045,    0.0028161,],
        [-0.0016208,   -0.0018633,   -0.0027254,],
        [-0.0016862,   -0.0014531,   -0.0019444,],
        [ 0.0041234,    0.0034255,   -0.0042112,],
        [ 0.0031555,   -0.0010435,    0.0009888,],
        [ 0.0051177,    0.0025433,    0.0026988,],
        [-0.0014040,    0.0018589,    0.0051271,],
        [ 0.0001667,    0.0007933,   -0.0018817,],
        [-0.0024006,   -0.0026783,   -0.0028726,],
        [ 0.0044775,   -0.0008214,   -0.0012893,],
        [ 0.0047634,    0.0022124,    0.0021387,],
        [-0.0030317,    0.0038995,    0.0063474,],
        [ 0.0040435,    0.0121992,   -0.0016736,],
        ])

    E1, G1 = run_rhf(primary_name='def2-tzvp')
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)
