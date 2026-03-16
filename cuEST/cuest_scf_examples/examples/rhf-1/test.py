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

import cuest.bindings as ce
import cuest_scf

import os
filedir = os.path.dirname(os.path.realpath(__file__))   
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

# Usually you want one CuestHandle for many computations - e.g., do not spin up
# separate handles for every RHF instance
cuest_handle = cuest_scf.CuestHandle()

# This will set the cuEST handle so all operations are performed with FP64 arithmetic
# cuest_handle.set_math_mode(mode=ce.CuestMathMode.CUEST_NATIVE_FP64_MATH_MODE)

# Reasonable default - as coarse as 1.0E-10 yields high-precision results
# Coarser can save some CoreDF memory and lower runtime
threshold_pq = 1.0E-10

molecule = cuest_scf.Molecule.parse_from_xyz_file('paxlovid.xyz')
#molecule = cuest_scf.Molecule.parse_from_xyz_file('192atoms_benzene16.xyz')
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
# NOTE: xc_functional_name is used to specify the XC functional.
# Currently supported functionals are listed below. 
    xc_functional_name='HF',
    #xc_functional_name='B3LYP1',
    #xc_functional_name='B3LYP5',
#    xc_functional_name='B97',
#    xc_functional_name='BLYP',
#    xc_functional_name='M06-L',
#    xc_functional_name='PBE',
#    xc_functional_name='PBE0',
#    xc_functional_name='r2SCAN',
#    xc_functional_name='SVWN5',
    #xc_functional_name='B97M-V',
    primary=primary,
    auxiliary=auxiliary,
    minao=minao,
    primary_name=primary_name,
    auxiliary_name=auxiliary_name,
    minao_name=minao_name,
    threshold_pq=threshold_pq,
    # NOTE: Delicate applications or tight comparisons of gradients may require
    # tighter than the working default of g_convergence=1.0E-6
    # g_convergence=1.0E-8,
    xc_grid_family="SG",
    xc_grid_level=1,
    xc_threshold_collocation=1.0e-18,  # Collocation threshold for local XC potential
    nlc_threshold_collocation=1.0e-18, # Collocation threshold for nonlocal (VV10) potential
    dfk_int8_modulus_count_start=4,
    dfk_int8_modulus_count_end=10,
    dfk_int8_slice_count_start=3,
    dfk_int8_slice_count_end=6,
    pcm_num_angular_points_per_hydrogen_atom=110,
    pcm_num_angular_points_per_heavy_atom=194,
    pcm_epsilon=1.0,
    pcm_cutoff=1e-10,
    pcm_convergence_tol=1e-10,
    benchmark=True,
    )

rhf.solve()

print('SCF Energy: %24.16E' % (rhf.compute_energy()))
print('')

#print('SCF Gradient:')
#print(rhf.compute_gradient().to_numpy())
