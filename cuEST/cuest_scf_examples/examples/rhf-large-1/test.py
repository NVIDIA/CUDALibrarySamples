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

#
# This example provides guidance for performing efficient SCF calcuations
# on larger-scale systems. It shows how to use integral direct and 
# Grid-K based techniques. 
#
# As of cuEST v0.3.0, these J/K algorithms are only recommended for 
# GPUs with full rate FP64 compute (H100, H200 and B200). Similarly,
# these J/K algorithms are only recommended for use on CTK 13.0.2
# or later. 
#

import cuest.bindings as ce
import cuest_scf

import os
filedir = os.path.dirname(os.path.realpath(__file__))   
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

cuest_handle = cuest_scf.CuestHandle()

# Pair screening threshold
threshold_pq = 1.0E-10

molecule = cuest_scf.Molecule.parse_from_xyz_file(os.path.join(filedir, 'urea_376.xyz'))
charge = 0

primary_name = 'def2-tzvp'
auxiliary_name = 'def2-universal-jfit'
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
#    xc_functional_name='wB97M-V',
    primary=primary,
    auxiliary=auxiliary,
    minao=minao,
    primary_name=primary_name,
    auxiliary_name=auxiliary_name,
    minao_name=minao_name,
    print_level=2,
    print_timings=False,
    benchmark=True,
    threshold_pq=threshold_pq,
    g_convergence=1.0E-6,
    cpu_diis=False, # When true, this option moves DIIS storage to the CPU. Useful for very large calculations.
#
# ** DFT-specific options **
#
#    xc_grid_family="SG",
#    xc_grid_level=2,
#    xc_threshold_collocation=1.0e-10,
#    nlc_grid_family="SG",
#    nlc_grid_level=0,
#    nlc_threshold_collocation=1.0e-10,
#
# ** PCM-specific options **
#
#    pcm_epsilon=78.0,
#    pcm_num_angular_points_per_heavy_atom=194,
#    pcm_num_angular_points_per_hydrogen_atom=110,
#
# ** Compensated Grid-K. **
#
# See: https://arxiv.org/abs/2611.16420
#
# This uses Grid-K for exchange (including overlap fitting) and a SAD-based analytic correction.
# Coulomb interactions can be evaluated with or without density fitting. Density fitted
# Coulomb (DF-J) provides better performance, integral direct Coulomb provides better agreement with
# fully integral direct calculations. When DF-J is used, the error relative to integral direct
# calculations tends to be dominated by the density fitting error.
#
# Here, we use a "trimmed" SG-0(-2) grid. This is a systematic truncation of the SG-0 standard
# grid. In practice, this grid has been found to provide exchange energies accurate to roughly
# a microhartree per atom or roughly an order of magnitude better than the density-fitted
# Coulomb build when paired with TZ- or QZ-quality basis sets. Work on additional recommendations
# for integration grids and purpose-built grids is underway.
#
    fock_method='dfjgridk',                      # Uses density-fitted Coulomb matrices
#    fock_method='directjgridk',                 # Uses integral direct Coulomb matrices
    gridk_grid_family='SG0TRIMMED',
    gridk_grid_level=-2,
    gridk_threshold_collocation=1.0e-10,
    gridk_fitting_algorithm='overlap_symmetric', # Enables symmetric fitting
    gridk_sad_reference=True,                    # Analytic corrections (based on the SAD density matrix)
    directjk_schwarz_threshold=1.0e-13,          # Affects density-based screening of the analytic corrections
#
# ** Four-center integral direct J/K. **
#
# This code path relies on JIT compilation. The first time the code is run, certain kernels are compiled.
# Subsequent runs will use the JIT-compiled kernels if they are available.
# 
#    fock_method='directjk',
#    directjk_schwarz_threshold=1.0e-13, # Affects density-based screening
#    incremental_fock=True,              # Efficiency is improved by incremental builds
#    incremental_fock_rebuild=8,         # Forces a full rebuild every N-th iteration
    )

rhf.solve()
_ = rhf.compute_gradient()
rhf.timer.print_out()

