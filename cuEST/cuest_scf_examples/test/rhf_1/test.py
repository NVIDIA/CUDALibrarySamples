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
        primary=primary,
        xc_functional_name='HF',
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=threshold_pq,
        )
    
    rhf.solve()

    return rhf.scalars['Escf']

def test_rhf():

    reference_values = {
        'sto-3g'    : -1.7371712774897262E+03,
        'def2-svp'  : -1.7583044466471508E+03,
        'def2-tzvp' : -1.7602954939152182E+03,
        # def2-QZVP really wants an 80 GB A100 - disabling by default
        # 'def2-qzvp' : -1.7603826412465432E+03,
    }

    for primary_name, E2 in reference_values.items():

        E1 = run_rhf(primary_name=primary_name)
        dE = abs(E1 - E2)
        assert(dE < 1.0E-6)
        
        print(dE)
