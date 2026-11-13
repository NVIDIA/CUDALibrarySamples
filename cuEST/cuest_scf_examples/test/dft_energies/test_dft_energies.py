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
        g_convergence=1.0e-8,
        print_level=0,
        )
    
    rhf.solve()
    properties = rhf.compute_properties()

    return rhf.scalars['Escf'], properties


def test_dft_functionals():

    reference_values = {
        'HF'        : [ -1.422651565914184e+03, [ 0.00455599169747, 1.50977483044355, -0.80325389829654,],],
        'B3LYP1'    : [ -1.429567122227856e+03, [ -0.24237462739605, 1.43723219027012, -0.73119656185187,],],
        'B3LYP5'    : [ -1.428913097794791e+03, [ -0.24328689648192, 1.43610112996096, -0.73097924532952,],],
        'B97'       : [ -1.429147445847821e+03, [ -0.26576625503246, 1.42693021811119, -0.73515320470323,],],
        'BLYP'      : [ -1.429194420620358e+03, [ -0.31357252597816, 1.39959942874909, -0.70893442172865,],],
        'M06-L'     : [ -1.429412651718125e+03, [ -0.36308319408444, 1.37346788928550, -0.74215890309813,],],
        'PBE'       : [ -1.428158369044055e+03, [ -0.35540425080925, 1.39713718161917, -0.71187559013658,],],
        'PBE0'      : [ -1.428241640043572e+03, [ -0.26458810371727, 1.43866116992777, -0.73824291251597,],],
        'SVWN5'     : [ -1.419574658456410e+03, [ -0.35894152112994, 1.44875573150914, -0.71301698075050,],],
        'LC-wPBE'   : [ -1.428646311949541e+03, [ -0.25159580300870, 1.44202552525231, -0.75182940778211,],],
        'wB97X'     : [ -1.429291382206683e+03, [ -0.24328345404139, 1.43321215394471, -0.74530367976638,],],
        'wB97X-D'   : [ -1.429187003337816e+03, [ -0.26038838489332, 1.43527707771284, -0.74631619369471,],],
        'wB97X-D3'  : [ -1.429253235431202e+03, [ -0.25329020675852, 1.43491126736304, -0.74723239621547,],],
        'wB97X-V'   : [ -1.429213014106862e+03, [ -0.23134771386334, 1.43425203974508, -0.74277184009611,],],
        'wB97M-V'   : [ -1.429190443003292e+03, [ -0.20961144963896, 1.43285465527620, -0.73610899724015,],],
        'r2SCAN'    : [ -1.428969446876130e+03, [ -0.32016603196021, 1.43136058511682, -0.73642064196701,],],
        'B97M-V'    : [ -1.429394572236334e+03, [ -0.26154953774292, 1.39319048189758, -0.73334555057337,],],
        'LC-wPBEh'  : [ -1.428626949558679e+03, [ -0.18418172003282, 1.46874409281621, -0.76575070540109,],],
        'CAM-B3LYP' : [ -1.429073274015881e+03, [ -0.21206359810267, 1.45275409915983, -0.74253574085433,],],
        'HSE06'     : [ -1.428347110643239e+03, [ -0.26183488450197, 1.44137721195029, -0.73673493691829,],],
        'M06'       : [ -1.428871872083314e+03, [ -0.24952111864239, 1.42745822823390, -0.73313967990728,],],
        'M06-2X'    : [ -1.429132965583964e+03, [ -0.20325237918241, 1.47484642515600, -0.74509027291944,],],
    }

    print()
    for functional_name, [E2, [mu_x2, mu_y2, mu_z2,]] in reference_values.items():

        E1, properties1 = run_rhf(functional_name=functional_name)
        dE = abs(E1 - E2)
        assert(dE < 1.0E-6)
        
        dmu_x = abs(properties1['mu_x'] - mu_x2)
        assert(dmu_x < 1.0E-5)
        
        dmu_y = abs(properties1['mu_y'] - mu_y2)
        assert(dmu_y < 1.0E-5)
        
        dmu_z = abs(properties1['mu_z'] - mu_z2)
        assert(dmu_z < 1.0E-5)
        
        print('%10s %10.3e %10.3e %10.3e %10.3e' % (functional_name, dE, dmu_x, dmu_y, dmu_z))

