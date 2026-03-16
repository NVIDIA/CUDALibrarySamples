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
import numpy as np
import os
filedir = os.path.dirname(os.path.realpath(__file__))   
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

cuest_handle = cuest_scf.CuestHandle()

def run_rhf(
    *,
    primary_name,
    xc_grid_level,
    xc_grid_family,
    ):

    # Reasonable default - as coarse as 1.0E-10 yields high-precision results
    # Coarser can save some CoreDF memory and lower runtime
    threshold_pq = 1.0E-14

    molecule = cuest_scf.Molecule.parse_from_xyz_string("""
        C     -2.779381    1.358796   -0.017634
        N     -1.370791    1.405167   -0.366529
        F     -2.968627    1.790547    1.252788
        H     -3.367136    1.987227   -0.691709
        Cl    -3.330473   -0.311805   -0.186740
        O     -0.662015    0.806738    0.752014
        Br     0.836087    1.898704    1.267596
        H     -1.224487    0.696240   -1.080917""")
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
        xc_functional_name='BLYP',
        primary=primary,
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=threshold_pq,
        g_convergence=1.0E-8, # Tighter for gradient check
        xc_grid_level=xc_grid_level,
        xc_grid_family=xc_grid_family,
        )
    
    rhf.solve()

    E = rhf.compute_energy()

    G = rhf.compute_gradient()

    return E, G


def test_blyp1_grid1():

    import numpy as np

    E2 = -3.3026700806219742E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0090490,  -0.0104565,  -0.0117094],
        [-0.0100463,  -0.0018479,  -0.0243732],
        [-0.0001783,  -0.0048707,   0.0004488],
        [ 0.0044712,  -0.0040387,   0.0094299],
        [ 0.0207582,   0.0281712,   0.0004648],
        [ 0.0022800,  -0.0044285,   0.0125760],
        [-0.0020055,  -0.0027708,  -0.0023172],
        [-0.0062302,   0.0002419,   0.0154801],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='GRID',
        xc_grid_level=1,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)


def test_blyp1_grid2():

    import numpy as np

    E2 = -3.3026741241897430E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0086464,  -0.0111337,  -0.0115950],
        [-0.0121519,  -0.0020085,  -0.0230772],
        [ 0.0004133,  -0.0041646,  -0.0006530],
        [ 0.0037753,  -0.0052239,   0.0098708],
        [ 0.0207992,   0.0285744,   0.0007199],
        [ 0.0048984,  -0.0025692,   0.0129005],
        [-0.0053030,  -0.0049576,  -0.0035499],
        [-0.0037849,   0.0014830,   0.0153839],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='GRID',
        xc_grid_level=2,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)


def test_blyp1_grid3():

    import numpy as np

    E2 = -3.3026735923753740E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0085272,  -0.0110104,  -0.0109514],
        [-0.0123139,  -0.0019728,  -0.0230565],
        [ 0.0004201,  -0.0042886,  -0.0013846],
        [ 0.0038463,  -0.0051010,   0.0097640],
        [ 0.0207357,   0.0284409,   0.0009540],
        [ 0.0034311,  -0.0037979,   0.0124144],
        [-0.0036314,  -0.0037463,  -0.0029304],
        [-0.0039608,   0.0014761,   0.0151904],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='GRID',
        xc_grid_level=3,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)


def test_blyp1_grid4():

    import numpy as np

    E2 = -3.3026735859656069E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0084841,  -0.0110430,  -0.0108776],
        [-0.0123528,  -0.0020159,  -0.0231832],
        [ 0.0004196,  -0.0044269,  -0.0013943],
        [ 0.0037538,  -0.0050221,   0.0096938],
        [ 0.0207927,   0.0285202,   0.0009296],
        [ 0.0042107,  -0.0032913,   0.0127780],
        [-0.0043379,  -0.0042486,  -0.0032213],
        [-0.0040020,   0.0015277,   0.0152750],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='GRID',
        xc_grid_level=4,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)


def test_blyp1_grid5():

    import numpy as np

    E2 = -3.3026736646568511E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0084501,  -0.0109916,  -0.0107023],
        [-0.0122833,  -0.0020737,  -0.0231218],
        [ 0.0004239,  -0.0044861,  -0.0015387],
        [ 0.0037366,  -0.0049973,   0.0096706],
        [ 0.0207652,   0.0285071,   0.0009151],
        [ 0.0038299,  -0.0034625,   0.0125407],
        [-0.0040454,  -0.0040482,  -0.0030641],
        [-0.0039768,   0.0015524,   0.0153005],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='GRID',
        xc_grid_level=5,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)


def test_blyp1_sg0():

    import numpy as np

    E2 = -3.3026843865092101E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0085120,  -0.0110529,  -0.0103269],
        [-0.0114557,  -0.0016662,  -0.0232143],
        [ 0.0002911,  -0.0047466,  -0.0017821],
        [ 0.0038813,  -0.0050415,   0.0097291],
        [ 0.0209246,   0.0287900,  -0.0000189],
        [ 0.0030973,  -0.0048202,   0.0137675],
        [-0.0039907,  -0.0032770,  -0.0038259],
        [-0.0042358,   0.0018145,   0.0156714],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='SG',
        xc_grid_level=0,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)


def test_blyp1_sg1():

    E2 = -3.3026752311472001E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0085473,  -0.0112361,  -0.0108816],
        [-0.0122674,  -0.0022214,  -0.0229901],
        [ 0.0005008,  -0.0043548,  -0.0012217],
        [ 0.0036445,  -0.0049861,   0.0095828],
        [ 0.0207480,   0.0285826,   0.0008645],
        [ 0.0044558,  -0.0031672,   0.0127055],
        [-0.0048013,  -0.0042453,  -0.0032890],
        [-0.0037329,   0.0016284,   0.0152296],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='SG',
        xc_grid_level=1,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)


def test_blyp1_sg2():

    E2 = -3.3026736460982765E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0084323,  -0.0111196,  -0.0108407],
        [-0.0121870,  -0.0020630,  -0.0230002],
        [ 0.0003711,  -0.0044170,  -0.0012362],
        [ 0.0037095,  -0.0048980,   0.0095853],
        [ 0.0207128,   0.0284073,   0.0008231],
        [ 0.0026341,  -0.0039898,   0.0121270],
        [-0.0030084,  -0.0036482,  -0.0028789],
        [-0.0037998,   0.0017283,   0.0154206],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='SG',
        xc_grid_level=2,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)


def test_blyp1_sg3():

    E2 = -3.3026737110443828E+03 # External code

    # External code we trust
    G2 = np.array([
        [-0.0085442,  -0.0110449,  -0.0106980],
        [-0.0123756,  -0.0020051,  -0.0229404],
        [ 0.0003864,  -0.0045772,  -0.0015611],
        [ 0.0037599,  -0.0050463,   0.0096945],
        [ 0.0208292,   0.0286330,   0.0008832],
        [ 0.0038349,  -0.0035991,   0.0123249],
        [-0.0039995,  -0.0039321,  -0.0028949],
        [-0.0038911,   0.0015716,   0.0151917],
        ])

    E1, G1 = run_rhf(
        primary_name='def2-svp',
        xc_grid_family='SG',
        xc_grid_level=3,
        )
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)
