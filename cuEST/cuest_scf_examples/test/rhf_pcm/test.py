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
        pcm_num_angular_points_per_hydrogen_atom=110,
        pcm_num_angular_points_per_heavy_atom=110,
        pcm_epsilon=80.0,
        )
    
    rhf.solve()

    E = rhf.compute_energy()

    G = rhf.compute_gradient()

    return E, G

def test_rhf():

    import numpy as np

    E2 = -1.7603457152124370E+03 # External code

    # External code we trust
    G2 = np.array([
       [ -0.0184988,  -0.0343845,   0.0273796],
       [ -0.0379316,   0.0278096,  -0.0081553],
       [  0.0283527,   0.0232373,   0.0250976],
       [  0.0053911,  -0.0031072,  -0.0507543],
       [  0.0309708,   0.0302571,   0.0317611],
       [ -0.0188022,  -0.0369833,  -0.0517220],
       [ -0.0065059,   0.0053585,   0.0335458],
       [  0.0092830,  -0.0376305,  -0.0282209],
       [  0.0138367,   0.0106862,   0.0273180],
       [  0.0589491,   0.0023250,  -0.0163916],
       [  0.0051612,   0.0106007,  -0.0539697],
       [  0.0462200,  -0.0602542,   0.0402997],
       [ -0.0056771,   0.0223969,   0.0126297],
       [ -0.0116759,  -0.0260531,   0.0048811],
       [ -0.0100212,   0.0187439,   0.0116294],
       [  0.0232106,  -0.0156805,  -0.0132428],
       [  0.0125720,  -0.0095002,  -0.0004171],
       [ -0.0033301,   0.0088493,   0.0037391],
       [  0.0083042,  -0.0051456,  -0.0084316],
       [ -0.0420767,   0.0002935,  -0.0006771],
       [ -0.0338492,   0.0334450,  -0.0067477],
       [  0.0067036,  -0.0032993,   0.0169881],
       [ -0.0116312,  -0.0162091,   0.0035779],
       [  0.0063908,  -0.0078621,   0.0088007],
       [ -0.0090302,  -0.0103273,  -0.0032672],
       [  0.0089824,  -0.0003942,  -0.0005032],
       [ -0.0032007,   0.0061253,   0.0036111],
       [ -0.0080791,  -0.0064064,  -0.0207911],
       [ -0.0036370,   0.0047253,   0.0060230],
       [ -0.0535494,   0.0176575,   0.0591499],
       [ -0.0513839,   0.0653972,  -0.0429843],
       [ -0.0002590,   0.0084638,   0.0102601],
       [ -0.0023809,  -0.0365184,   0.0054908],
       [  0.0449690,  -0.0199295,  -0.0597143],
       [  0.0010033,  -0.0134220,   0.0149047],
       [  0.0056473,  -0.0016361,  -0.0018693],
       [  0.0035635,   0.0023108,  -0.0060661],
       [ -0.0059505,  -0.0020846,   0.0092711],
       [  0.0074199,   0.0076297,  -0.0044440],
       [ -0.0093278,  -0.0030117,   0.0029823],
       [ -0.0060894,  -0.0011337,  -0.0026556],
       [  0.0045012,  -0.0037991,  -0.0054157],
       [  0.0018104,  -0.0030781,   0.0055600],
       [  0.0002117,   0.0076567,   0.0022764],
       [ -0.0059578,  -0.0001824,  -0.0026886],
       [  0.0016187,  -0.0026258,   0.0056940],
       [ -0.0055311,  -0.0032632,  -0.0117384],
       [ -0.0054090,  -0.0015283,   0.0209616],
       [  0.0040352,   0.0022232,  -0.0110271],
       [  0.0075102,   0.0111909,   0.0184628],
       [ -0.0043351,   0.0017731,   0.0051392],
       [  0.0059134,   0.0066259,  -0.0003254],
       [  0.0115309,  -0.0026361,   0.0025671],
       [  0.0006331,   0.0080836,  -0.0051774],
       [ -0.0047574,   0.0029485,   0.0051459],
       [ -0.0036562,  -0.0030800,  -0.0048458],
       [ -0.0029043,  -0.0040270,  -0.0048385],
       [  0.0049893,   0.0054211,  -0.0058340],
       [  0.0058859,  -0.0037815,   0.0025470],
       [  0.0091569,   0.0031425,   0.0043669],
       [ -0.0025388,   0.0048138,   0.0093207],
       [ -0.0073157,  -0.0014136,  -0.0032850],
       [ -0.0036892,  -0.0058502,  -0.0061210],
       [  0.0088250,  -0.0019554,  -0.0007071],
       [  0.0080319,   0.0033563,  -0.0012707],
       [ -0.0081279,   0.0046624,   0.0033626],
       [  0.0055249,   0.0199835,  -0.0004452],
        ])

    E1, G1 = run_rhf(primary_name='def2-tzvp')
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)
