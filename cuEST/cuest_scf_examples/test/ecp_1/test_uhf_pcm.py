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

    threshold_pq = 1.0E-10

    molecule_filename = os.path.join(filedir, 'ch2i2.xyz')

    molecule = cuest_scf.Molecule.parse_from_xyz_file(molecule_filename, bohr_per_ang=1.0)
    charge = -1
    multiplicity = 2

    auxiliary_name = 'def2-universal-jkfit'
    minao_name = 'minao-1'
    ecp_name = 'def2-svp-ecp'

    primary_filename   = os.path.join(basisdir, '%s.gbs' % (primary_name))
    auxiliary_filename = os.path.join(basisdir, '%s.gbs' % (auxiliary_name))
    minao_filename     = os.path.join(basisdir, '%s.gbs' % (minao_name))
    ecp_filename       = os.path.join(basisdir, f'{ecp_name}.gbs')

    primary   = cuest_scf.AOBasis.parse_from_gbs_file(primary_filename,   molecule=molecule)
    auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(auxiliary_filename, molecule=molecule)
    minao     = cuest_scf.AOBasis.parse_from_gbs_file(minao_filename,     molecule=molecule)

    ecp_basis_full = cuest_scf.ECPBasis.parse_from_gbs_file(ecp_filename, molecule=molecule)

    uhf = cuest_scf.UHF(
        cuest_handle=cuest_handle,
        molecule=molecule,
        charge=charge,
        maxiter=250,
        multiplicity=multiplicity,
        primary=primary,
        ecp_basis=ecp_basis_full,
        xc_functional_name='HF',
        auxiliary=auxiliary,
        minao=minao,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        threshold_pq=threshold_pq,
        g_convergence=1.0E-8,
        pcm_epsilon=80.0,
        pcm_cutoff=1.0E-12,
        pcm_convergence_tol=1.0E-8,
        pcm_num_angular_points_per_heavy_atom=110,
        pcm_num_angular_points_per_hydrogen_atom=110,
        )

    uhf.solve()

    E = uhf.compute_energy()
    G = uhf.compute_gradient()

    return E, G

def test_uhf_ecp_pcm():

    import numpy as np

    reference_values = {
        'def2-tzvp' : (
            -6.3223951795653488E+02,
            np.array([
                [ 0.000000000243, -0.000000000000, -0.265089265920],
                [-0.227888947741, -0.000000000000,  0.128947115065],
                [ 0.227888947501,  0.000000000000,  0.128947114917],
                [-0.000000000001,  0.027187967405,  0.003597517971],
                [-0.000000000001, -0.027187967404,  0.003597517970],
            ]),
        ),
    }

    for primary_name, (E2, G2) in reference_values.items():

        E1, G1 = run_uhf(primary_name=primary_name)
        G1 = G1.to_numpy()

        dE = abs(E1 - E2)
        dG = np.max(np.abs(G1 - G2))

        print(f'dE = {dE:.3E}')
        print(f'dG = {dG:.3E}')

        assert(dE < 1.0E-5)
        assert(dG < 1.0E-6)
