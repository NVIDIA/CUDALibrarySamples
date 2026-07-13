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

def run_rhf(
    *,
    functional_name,
    ):

    # Reasonable default - as coarse as 1.0E-10 yields high-precision results
    # Coarser can save some CoreDF memory and lower runtime
    threshold_pq = 1.0E-18

    molecule_filename = os.path.join(filedir, 'benzylpenicillin.xyz')
    
    molecule = cuest_scf.Molecule.parse_from_xyz_file(molecule_filename)
    charge = 0
    
    primary_name = 'def2-tzvp'
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
        primary=primary,
        xc_functional_name=functional_name,
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=threshold_pq,
        xc_grid_family='SG',
        xc_grid_level=0,
        nlc_grid_family='SG',
        nlc_grid_level=0,
        g_convergence=1.0e-6,
        print_level=0,
        )
    
    rhf.solve()
    properties = rhf.compute_properties()

    return rhf.scalars['Escf'], properties


def test_dft_functionals():

    reference_values = {
        'HF'        : [ -1.422651565914184e+03, [  0.00458416123186, 1.50979520054170, -0.80327963287596,],],
        'B3LYP1'    : [ -1.429567122227856e+03, [ -0.24240656805426, 1.43723969877400, -0.73115563505112,],],
        'B3LYP5'    : [ -1.428913097794791e+03, [ -0.24331876157005, 1.43610854528413, -0.73093803806052,],],
        'B97'       : [ -1.429147445847821e+03, [ -0.26579678715483, 1.42693643120386, -0.73513735125318,],],
        'BLYP'      : [ -1.429194420620358e+03, [ -0.31355517605338, 1.39962555196724, -0.70889897069814,],],
        'M06-L'     : [ -1.429412651718125e+03, [ -0.36307578494349, 1.37345858398082, -0.74216669313293,],],
        'PBE'       : [ -1.428158369044055e+03, [ -0.35538781944093, 1.39716435582796, -0.71183964334111,],],
        'PBE0'      : [ -1.428241640043572e+03, [ -0.26460093910261, 1.43865927546809, -0.73825703648439,],],
        'SVWN5'     : [ -1.419574658456410e+03, [ -0.35891617402714, 1.44879619652011, -0.71297357499954,],],
        'LC-wPBE'   : [ -1.428646311949541e+03, [ -0.25160027921401, 1.44206482655074, -0.75184636473981,],],
        'wB97X'     : [ -1.429291382206683e+03, [ -0.24327999950988, 1.43325042242688, -0.74531044697911,],],
        'wB97X-V'   : [ -1.429213014106862e+03, [ -0.23134133268637, 1.43428656216930, -0.74279067360407,],],
        'wB97M-V'   : [ -1.429190443003292e+03, [ -0.20959875847244, 1.43289504150134, -0.73613357336939,],],
        'r2SCAN'    : [ -1.428969446876130e+03, [ -0.32015760043066, 1.43132710840997, -0.73650079226324,],],
        'B97M-V'    : [ -1.429394572236334e+03, [ -0.26153738448026, 1.39317178170207, -0.73332646898571,],],
        'LC-wPBEh'  : [ -1.428626949558679e+03, [ -0.18415943076069, 1.46877901179819, -0.76578555543790,],],
        'CAM-B3LYP' : [ -1.429073274015881e+03, [ -0.21204975998501, 1.45275659694822, -0.74250091169786,],],
        'HSE06'     : [ -1.428347110643239e+03, [ -0.26183561573804, 1.44137291876564, -0.73673874217298,],],
        'M06'       : [ -1.428871872083314e+03, [ -0.24955204313544, 1.42747873168316, -0.73311565358732,],],
        'M06-2X'    : [ -1.429132965583964e+03, [ -0.20327473651393, 1.47488317552861, -0.74511213838743,],],
    }

    print()
    for functional_name, [E2, [mu_x2, mu_y2, mu_z2,]] in reference_values.items():

        E1, properties1 = run_rhf(functional_name=functional_name)
        dE = abs(E1 - E2)
        assert(dE < 1.0E-6)
        
        dmu_x = abs(properties1['mu_x'] - mu_x2)
        assert(dmu_x < 1.0E-6)
        
        dmu_y = abs(properties1['mu_y'] - mu_y2)
        assert(dmu_y < 1.0E-6)
        
        dmu_z = abs(properties1['mu_z'] - mu_z2)
        assert(dmu_z < 1.0E-6)
        
        print('%10s %10.3e %10.3e %10.3e %10.3e' % (functional_name, dE, dmu_x, dmu_y, dmu_z))

