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
        g_convergence=1.0e-6,
        pcm_num_angular_points_per_hydrogen_atom=110,
        pcm_num_angular_points_per_heavy_atom=110,
        pcm_epsilon=80.0,
        pcm_cutoff=1e-12,
        pcm_convergence_tol=1e-8,
        )
    
    uhf.solve()

    E = uhf.compute_energy()

    G = uhf.compute_gradient()

    return E, G

def test_uhf():

    import numpy as np

    E2 = -1.7603703412082559E+03 # External code

    # External code we trust
    G2 = np.array([
        [ 0.0072146,  -0.0227639,   0.0007103],
        [-0.0037383,   0.0129137,  -0.0197669],
        [ 0.0263142,   0.0126084,   0.0077959],
        [ 0.0045334,  -0.0030845,  -0.0501585],
        [ 0.0239259,   0.0319812,   0.0265799],
        [ 0.0308534,   0.0432629,   0.0590828],
        [-0.0065395,   0.0053044,   0.0335023],
        [ 0.0072488,  -0.0315114,  -0.0292094],
        [ 0.0129655,   0.0107568,   0.0271568],
        [-0.0354965,  -0.0206123,  -0.0077570],
        [ 0.0051101,   0.0105393,  -0.0539565],
        [ 0.0461562,  -0.0601613,   0.0402400],
        [-0.0061418,   0.0225695,   0.0125392],
        [-0.0116611,  -0.0260118,   0.0051724],
        [-0.0104569,   0.0185200,   0.0115704],
        [ 0.0247862,  -0.0162002,  -0.0123478],
        [ 0.0125457,  -0.0101003,  -0.0018417],
        [-0.0033213,   0.0087123,   0.0036222],
        [ 0.0082659,  -0.0053272,  -0.0084678],
        [-0.0403821,   0.0004473,  -0.0012447],
        [-0.0260499,   0.0244030,   0.0017911],
        [ 0.0145934,  -0.0112922,   0.0014009],
        [-0.0124218,  -0.0108778,   0.0044646],
        [ 0.0067271,  -0.0079193,   0.0087188],
        [-0.0085874,  -0.0106745,  -0.0021678],
        [ 0.0077164,  -0.0005453,  -0.0012100],
        [-0.0035596,   0.0030746,   0.0036814],
        [-0.0081495,  -0.0063684,  -0.0207629],
        [-0.0035949,   0.0046888,   0.0060460],
        [ 0.0029109,  -0.0490166,  -0.0857234],
        [-0.0513906,   0.0653105,  -0.0429362],
        [-0.0002701,   0.0084655,   0.0102327],
        [-0.0023410,  -0.0363827,   0.0055317],
        [-0.0254125,   0.0067581,   0.0250069],
        [ 0.0010264,  -0.0134189,   0.0149063],
        [ 0.0056853,  -0.0013179,  -0.0018870],
        [ 0.0035901,   0.0026744,  -0.0060194],
        [-0.0062416,  -0.0018329,   0.0090708],
        [ 0.0073696,   0.0077227,  -0.0041522],
        [-0.0096237,  -0.0040459,   0.0027896],
        [-0.0060356,  -0.0010635,  -0.0026435],
        [ 0.0044889,  -0.0037767,  -0.0053973],
        [ 0.0018260,  -0.0029624,   0.0055247],
        [ 0.0001698,   0.0078107,   0.0024219],
        [-0.0058870,  -0.0002131,  -0.0027061],
        [ 0.0016556,  -0.0025211,   0.0056088],
        [-0.0056160,  -0.0028931,  -0.0117941],
        [-0.0053504,  -0.0016010,   0.0211328],
        [ 0.0040324,   0.0022266,  -0.0110259],
        [ 0.0052913,   0.0156256,   0.0246006],
        [-0.0041269,   0.0024101,   0.0055233],
        [ 0.0052031,   0.0056289,  -0.0003451],
        [ 0.0112869,  -0.0025715,   0.0025638],
        [ 0.0006830,   0.0073495,  -0.0043679],
        [-0.0050330,   0.0035800,   0.0052801],
        [-0.0034573,  -0.0030274,  -0.0047168],
        [-0.0028223,  -0.0038588,  -0.0047003],
        [ 0.0039152,   0.0054489,  -0.0046313],
        [ 0.0057667,  -0.0035633,   0.0025053],
        [ 0.0091214,   0.0031075,   0.0043475],
        [-0.0025306,   0.0048114,   0.0093296],
        [-0.0072857,  -0.0013904,  -0.0033442],
        [-0.0036859,  -0.0058543,  -0.0061245],
        [ 0.0088044,  -0.0019540,  -0.0006951],
        [ 0.0080212,   0.0033477,  -0.0012659],
        [-0.0081383,   0.0046667,   0.0033539],
        [ 0.0055445,   0.0199891,  -0.0004380],
        ])

    E1, G1 = run_uhf(primary_name='def2-tzvp')
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)
