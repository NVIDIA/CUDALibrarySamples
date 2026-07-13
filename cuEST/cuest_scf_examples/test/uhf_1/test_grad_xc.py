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
        xc_functional_name='B3LYP1',
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

    E2 = -1.7706054887023663E+03 # External code

    # External code we trust
    G2 = np.array([
     [ 0.0106948,    0.0009948,   -0.0083230],
     [ 0.0096110,   -0.0008430,   -0.0053086],
     [ 0.0035439,   -0.0042744,   -0.0102502],
     [-0.0114389,   -0.0031577,   -0.0046469],
     [-0.0163369,    0.0082428,   -0.0167247],
     [ 0.0113219,    0.0195781,    0.0254358],
     [-0.0055113,   -0.0005581,    0.0135634],
     [ 0.0110913,   -0.0000922,   -0.0180158],
     [-0.0036247,    0.0035165,    0.0129358],
     [ 0.0025853,   -0.0095887,   -0.0007182],
     [-0.0011266,    0.0042075,   -0.0289250],
     [ 0.0092296,   -0.0110057,    0.0065573],
     [-0.0073924,    0.0171425,    0.0061164],
     [-0.0045704,   -0.0176909,    0.0014017],
     [-0.0058012,    0.0110907,    0.0128566],
     [ 0.0187402,   -0.0130064,   -0.0028385],
     [ 0.0037131,   -0.0055465,   -0.0083349],
     [-0.0036214,    0.0019630,    0.0026601],
     [ 0.0041876,   -0.0026366,   -0.0031896],
     [-0.0091411,    0.0053066,   -0.0139936],
     [ 0.0110389,    0.0092461,    0.0266792],
     [-0.0028455,    0.0005680,    0.0012743],
     [-0.0095092,   -0.0143044,    0.0059967],
     [ 0.0005496,    0.0019660,   -0.0058822],
     [-0.0057434,   -0.0058871,    0.0014352],
     [ 0.0037829,    0.0010196,   -0.0019973],
     [-0.0009523,    0.0016072,   -0.0011787],
     [-0.0032134,   -0.0062388,   -0.0114038],
     [-0.0061601,    0.0049501,    0.0051543],
     [-0.0112591,   -0.0269678,   -0.0275364],
     [-0.0138487,    0.0129779,   -0.0068250],
     [ 0.0029655,    0.0079562,    0.0081709],
     [ 0.0031049,   -0.0195667,   -0.0064923],
     [-0.0156506,    0.0082926,    0.0185550],
     [ 0.0003196,   -0.0111988,    0.0105871],
     [ 0.0005367,    0.0030475,    0.0001610],
     [ 0.0020248,    0.0056462,    0.0002075],
     [-0.0031749,   -0.0028594,   -0.0011839],
     [ 0.0042085,    0.0042243,    0.0036561],
     [ 0.0006410,   -0.0067554,    0.0013330],
     [-0.0005185,    0.0003874,   -0.0008214],
     [ 0.0011929,   -0.0008816,   -0.0014252],
     [ 0.0008904,    0.0007268,    0.0006033],
     [ 0.0004964,    0.0019314,    0.0016866],
     [-0.0006969,   -0.0007771,   -0.0016619],
     [ 0.0008217,    0.0001181,    0.0000067],
     [ 0.0026138,   -0.0001940,   -0.0044547],
     [-0.0012350,   -0.0032976,    0.0093612],
     [ 0.0049939,    0.0018955,   -0.0009486],
     [ 0.0006821,    0.0081765,    0.0090148],
     [-0.0018713,    0.0011311,    0.0007965],
     [ 0.0033706,   -0.0007296,    0.0020254],
     [ 0.0064845,    0.0014593,   -0.0012852],
     [-0.0013994,    0.0014272,   -0.0010979],
     [-0.0006781,    0.0019489,    0.0015604],
     [ 0.0000687,   -0.0005397,   -0.0017000],
     [-0.0006123,    0.0000905,   -0.0005654],
     [ 0.0021564,    0.0005356,   -0.0017239],
     [ 0.0028516,   -0.0003197,    0.0011320],
     [ 0.0020346,    0.0019412,    0.0021488],
     [-0.0004192,    0.0015178,    0.0040164],
     [ 0.0010846,    0.0008361,   -0.0028156],
     [-0.0010913,   -0.0015055,   -0.0018456],
     [ 0.0018079,   -0.0003739,   -0.0012676],
     [ 0.0017663,    0.0011969,    0.0034724],
     [-0.0018207,    0.0035474,    0.0070641],
     [ 0.0040574,    0.0083854,   -0.0022447],
        ])

    E1, G1 = run_uhf(primary_name='def2-tzvp')
    G1 = G1.to_numpy()
    dE = abs(E1 - E2)
    dG = np.max(np.abs(G1 - G2))

    print(dE)
    print(dG)

    assert(dE < 1.0E-6)
    assert(dG < 1.0E-6)
