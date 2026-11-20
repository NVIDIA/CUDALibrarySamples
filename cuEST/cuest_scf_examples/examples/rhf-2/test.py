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

# This example runs the same RHF computation with each of the four Fock
# build methods (DF-J/DF-K, integral-direct J/K, and compensated grid-K
# paired with either DF-J or direct J) and prints a table of the energies
# and their errors relative to the integral-direct J/K reference.

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

# Conservative default - as coarse as 1.0E-10 yields high-precision results
# Coarser can save some memory and lower runtime
threshold_pq = 1.0E-13

molecule = cuest_scf.Molecule.parse_from_xyz_file(os.path.join(filedir, 'paxlovid.xyz'))
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

# Everything the four Fock methods share; each run function below adds its
# method-specific options on top of these
common_kwargs = dict(
    cuest_handle=cuest_handle,
    molecule=molecule,
    charge=charge,
    xc_functional_name='wB97M-V',
    primary=primary,
    auxiliary=auxiliary,
    minao=minao,
    primary_name=primary_name,
    auxiliary_name=auxiliary_name,
    minao_name=minao_name,
    threshold_pq=threshold_pq,
    g_convergence=1.0E-6,
    xc_grid_family="SG",
    xc_grid_level=2,
    nlc_grid_family="SG",
    nlc_grid_level=0,
    )


# DF-J/DF-K: both the Coulomb and exchange matrices use density-fitting
# approximations. The modulus/slice counts adjust the precision of the DF-K contraction.
def run_dfjk():
    rhf = cuest_scf.RHF(
        fock_method='dfjk',
        dfk_int8_modulus_count_start=8,
        dfk_int8_modulus_count_end=8,
        dfk_int8_slice_count_start=5,
        dfk_int8_slice_count_end=5,
        **common_kwargs,
        )
    rhf.solve()
    return rhf.compute_energy()


# Integral-direct J/K: Coulomb and exchange are constructed from
# Schwarz-screened 4-center integrals generated on-the-fly.
def run_directjk():
    rhf = cuest_scf.RHF(
        fock_method='directjk',
        directjk_schwarz_threshold=1.0e-13,
        incremental_fock=True,              # Efficiency is improved by incremental builds
                                            # Use with caution, may cause instability
        **common_kwargs,
        )
    rhf.solve()
    return rhf.compute_energy()


# Compensated Grid-K (with density-fitted Couloimb). Trades DF error
# in the Coulomb matrix for improved efficiency. Using Coulomb-fitting
# auxiliary basis sets can be considered in this context. 
def run_dfjgridk():
    rhf = cuest_scf.RHF(
        fock_method='dfjgridk',
        gridk_grid_family='SG',                      # SG-0 is a reasonable default choice
        gridk_grid_level=0,
        gridk_fitting_algorithm='overlap_symmetric', # Enables symmetric fitting
        gridk_sad_reference=True,                    # Analytic corrections (based on the SAD density matrix)
        directjk_schwarz_threshold=1.0e-13,          # Affects screening of the analytic corrections
        **common_kwargs,
        )
    rhf.solve()
    return rhf.compute_energy()


# Compensated Grid-K (with integral direct Couloimb). This approach 
# converges to the exact J/K matrices in the limit of a complete
# Grid-K integration grid.
def run_directjgridk():
    rhf = cuest_scf.RHF(
        fock_method='directjgridk',
        gridk_grid_family='SG',
        gridk_grid_level=0,
        gridk_fitting_algorithm='overlap_symmetric',
        gridk_sad_reference=True,
        directjk_schwarz_threshold=1.0e-13,
        **common_kwargs,
        )
    rhf.solve()
    return rhf.compute_energy()


runs = [
    ('Integral-Direct J/K', run_directjk),
    ('DF-J / DF-K', run_dfjk),
    ('DF-J / Compensated Grid-K', run_dfjgridk),
    ('Direct-J / Compensated Grid-K', run_directjgridk),
    ]

energies = {}
for name, run in runs:
    print('==> %s <==' % name)
    energies[name] = run()
    print('SCF Energy: %24.16E' % energies[name])
    print('')

reference = energies['Integral-Direct J/K']
print('%-32s %24s %14s' % ('Method', 'Energy (Eh)', 'Error (Eh)'))
for name, _ in runs:
    print('%-32s %24.16E %+14.6E' % (name, energies[name], energies[name] - reference))

