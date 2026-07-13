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

def run_uhf(
    *,
    primary_name,
    ):

    # Reasonable default - as coarse as 1.0E-10 yields high-precision results
    # Coarser can save some CoreDF memory and lower runtime
    threshold_pq = 1.0E-10

    molecule_filename = os.path.join(filedir, 'paxlovid.xyz')
    charge = -1
    multiplicity = 2
    
    molecule = cuest_scf.Molecule.parse_from_xyz_file(molecule_filename)
    
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
        xc_functional_name='HF',
        primary=primary,
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=threshold_pq,
        g_convergence=1.0e-8,
        )
    
    uhf.solve()

    E = uhf.compute_energy()

    G = uhf.compute_gradient()

    return E, G

def test_uhf():

    import numpy as np

    E2 = -1.7602515114551841E+03 # External code

    # External code we trust
    G2 = np.array([
       [ 0.0063749,   -0.0240336,   -0.0004685],
       [-0.0053602,    0.0125551,   -0.0214195],
       [ 0.0228326,    0.0092317,    0.0037518],
       [ 0.0042740,   -0.0038026,   -0.0608985],
       [ 0.0209348,    0.0364328,    0.0312743],
       [ 0.0174253,    0.0367618,    0.0508662],
       [-0.0117362,    0.0116870,    0.0540267],
       [ 0.0057716,   -0.0210419,   -0.0274923],
       [ 0.0068421,    0.0102026,    0.0199929],
       [-0.0483172,   -0.0233826,   -0.0044657],
       [-0.0006311,    0.0039828,   -0.0409889],
       [ 0.0465344,   -0.0610563,    0.0405039],
       [-0.0064890,    0.0230799,    0.0128165],
       [-0.0104419,   -0.0266584,    0.0050660],
       [-0.0112939,    0.0183768,    0.0099221],
       [ 0.0266131,   -0.0184792,   -0.0089958],
       [ 0.0143728,   -0.0116947,   -0.0040218],
       [-0.0035699,    0.0074124,    0.0036546],
       [ 0.0082158,   -0.0057529,   -0.0076435],
       [-0.0353164,    0.0012841,    0.0157924],
       [-0.0253225,    0.0118203,   -0.0054180],
       [ 0.0193561,   -0.0116204,   -0.0031168],
       [-0.0118964,   -0.0112002,    0.0038367],
       [ 0.0060311,   -0.0060973,    0.0070537],
       [-0.0085217,   -0.0096493,   -0.0017703],
       [ 0.0084210,   -0.0001851,   -0.0016080],
       [-0.0034952,    0.0024286,    0.0024973],
       [-0.0061115,   -0.0056210,   -0.0193774],
       [-0.0022638,    0.0033859,    0.0063721],
       [ 0.0212809,   -0.0366896,   -0.0733824],
       [-0.0524278,    0.0662166,   -0.0434678],
       [ 0.0009295,    0.0095494,    0.0103682],
       [ 0.0056379,   -0.0370324,   -0.0276852],
       [-0.0198174,    0.0113293,    0.0311209],
       [ 0.0023542,   -0.0157561,    0.0116543],
       [ 0.0050463,   -0.0012571,   -0.0012128],
       [ 0.0031821,    0.0031307,   -0.0059377],
       [-0.0055022,   -0.0024066,    0.0079177],
       [ 0.0060534,    0.0083110,   -0.0038399],
       [-0.0080359,   -0.0049993,    0.0020778],
       [-0.0061410,   -0.0009399,   -0.0026621],
       [ 0.0043235,   -0.0031114,   -0.0050875],
       [ 0.0018324,   -0.0024386,    0.0054321],
       [ 0.0022114,    0.0084445,    0.0031718],
       [-0.0066922,   -0.0019536,   -0.0032379],
       [ 0.0013884,   -0.0018232,    0.0048735],
       [-0.0030600,   -0.0029113,   -0.0120079],
       [-0.0051061,   -0.0040677,    0.0238602],
       [ 0.0063979,    0.0023222,   -0.0102854],
       [ 0.0048021,    0.0173280,    0.0246557],
       [-0.0051400,    0.0031349,    0.0064251],
       [ 0.0049196,    0.0043556,   -0.0000018],
       [ 0.0118218,   -0.0025494,    0.0010283],
       [-0.0003286,    0.0062292,   -0.0036352],
       [-0.0051309,    0.0054602,    0.0058903],
       [-0.0028403,   -0.0033192,   -0.0059331],
       [-0.0033382,   -0.0038955,   -0.0044023],
       [ 0.0035350,    0.0048686,   -0.0035544],
       [ 0.0058937,   -0.0026684,    0.0029677],
       [ 0.0085877,    0.0029930,    0.0033620],
       [-0.0031462,    0.0024865,    0.0098659],
       [-0.0070873,   -0.0006588,   -0.0043992],
       [-0.0035282,   -0.0054301,   -0.0058571],
       [ 0.0076082,   -0.0020162,   -0.0004056],
       [ 0.0071090,    0.0033543,    0.0000143],
       [-0.0080508,    0.0052998,    0.0041758],
       [ 0.0072253,    0.0227441,   -0.0016084],
        ])

    E1, G1 = run_uhf(primary_name='def2-tzvp')
    G1 = G1.to_numpy()

    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)
