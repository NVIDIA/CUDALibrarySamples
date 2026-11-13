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

import numpy as np

from .memoized_property import memoized_property

from .molecule import Molecule
from .ao_basis import AOBasis
from .ecp_basis import ECPBasis
from .sad_guess import SADGuess
from .diis import DIIS, DIISCPU
from .xc_functionals import XCFunctionalInfo

from .cuest_handle import CuestHandle
from .cuest_ao_basis import CuestAOBasis
from .cuest_ecp_basis import CuestECPBasis
from .cuest_molecular_grid import CuestMolecularGrid
from .cuest_ao_pair_list import CuestAOPairList
from .cuest_oe_int_plan import CuestOEIntPlan
from .cuest_oe_int_compute import CuestOEIntCompute
from .cuest_df_int_plan import CuestDFIntPlan
from .cuest_df_int_compute import CuestDFIntCompute
from .cuest_gridk_int_plan import CuestGridKIntPlan
from .cuest_gridk_int_compute import CuestGridKIntCompute
from .cuest_direct_int_plan import CuestDirectIntPlan
from .cuest_direct_int_compute import CuestDirectIntCompute
from .cuest_xc_int_plan import CuestXCIntPlan
from .cuest_xc_int_compute import CuestXCIntCompute
from .cuest_pcm_int_plan import CuestPCMIntPlan
from .cuest_pcm_int_compute import CuestPCMIntCompute
from .cuest_ecp_int_plan import CuestECPIntPlan
from .cuest_ecp_int_compute import CuestECPIntCompute

from .gpu_matrix import GPUMatrix
from .gpu_matrix_utility import GPUMatrixUtility

from .timer import Timer

import time

class UHF(object):

    def __init__(
        self,
        *,
        cuest_handle : CuestHandle,
        molecule : Molecule,
        charge : int,
        multiplicity : int,
        xc_functional_name : str = None,
        primary : AOBasis,
        auxiliary : AOBasis,
        minao : AOBasis,
        ecp_basis : ECPBasis = None,
        primary_name : str = None,
        auxiliary_name : str = None,
        minao_name : str = None,
        ecp_basis_name : str = None,
        print_level : int = 2,
        print_timings : bool = True,
        threshold_pq : float = 1.0E-12,
        threshold_canonical : float = 1.0E-6,
        df_fitting_eigenvalue_cutoff : float = 1.0e-12,
        df_integral_direct : bool = False,
        fock_method : str = "dfjk",
        directjk_schwarz_threshold : float = 1.0e-13,
        incremental_fock : bool = False,
        incremental_fock_rebuild : int = 8,
        gridk_threshold_collocation : float = 1.0e-18,
        gridk_grid_level : int | tuple[int, int] = 2,
        gridk_grid_family : str = "GRID",
        gridk_fitting_algorithm : str = "none",
        gridk_fitting_orthogonalization_cutoff : float = 1.0e-6,
        gridk_fitting_cutoff : float = 1.0e-2,
        gridk_sad_reference : bool = False,
        diis_max_nvector : int = 6,
        cpu_diis : bool = False,
        maxiter : int = 100,
        g_convergence : float = 1.0E-6,
        xc_threshold_collocation : float = 1.0e-18,
        xc_grid_level : int | tuple[int, int] = 2,
        xc_grid_family : str = "GRID",
        nlc_threshold_collocation : float = 1.0e-18,
        nlc_grid_level : int | tuple[int, int] = 1,
        nlc_grid_family : str = "GRID",
        dfk_int8_slice_count_start : int = 6,
        dfk_int8_modulus_count_start : int = 5,
        dfk_int8_slice_count_end : int = 6,
        dfk_int8_modulus_count_end : int = 9,
        pcm_epsilon : float = 1.0, # 1.0 for gas phase, >1.0 for solvated calc.
        pcm_x_prefactor : float = 0.0, # 0.0 for CPCM, 0.5 for COSMO
        pcm_cutoff : float = 1.0e-10,
        pcm_convergence_tol : float = 1e-10,
        pcm_num_angular_points_per_hydrogen_atom : int = 110,
        pcm_num_angular_points_per_heavy_atom : int = 194,
        pcm_atomic_radii_scale : float = 1.2,
        benchmark : bool = False,
        level_shift_a : float = 0.0,
        level_shift_b : float = 0.0,
        ):

        self.uhf = True
        self.rhf = False

        self.cuest_handle = cuest_handle

        self.molecule = molecule
        self.charge = charge
        self.multiplicity = multiplicity

        self.xc_functional = XCFunctionalInfo.string_to_enum(xc_functional_name)

        self.primary = primary
        self.auxiliary = auxiliary
        self.minao = minao
        self.ecp_basis = ecp_basis

        self.primary_name = primary_name
        self.auxiliary_name = auxiliary_name
        self.minao_name = minao_name
        self.ecp_basis_name = ecp_basis_name

        self.xc_threshold_collocation = xc_threshold_collocation
        self.xc_grid_level = xc_grid_level
        self.xc_grid_family = xc_grid_family

        self.nlc_threshold_collocation = nlc_threshold_collocation
        self.nlc_grid_level = nlc_grid_level
        self.nlc_grid_family = nlc_grid_family

        self.print_level = print_level
        self.print_timings = print_timings

        self.threshold_pq = threshold_pq
        self.threshold_canonical = threshold_canonical
        self.df_fitting_eigenvalue_cutoff = df_fitting_eigenvalue_cutoff
        self.df_integral_direct = df_integral_direct

        # Fock build method: "dfjk" (DF-J/DF-K), "dfjgridk" (DF-J/Grid-K),
        # "directjk" (direct 4-center J/K), "directjgridk" (direct-J/Grid-K)
        self.fock_method = fock_method.lower()
        supported_fock_methods = ["dfjk", "dfjgridk", "directjk", "directjgridk"]
        if self.fock_method not in supported_fock_methods:
            raise RuntimeError(
                f"Unrecognized fock_method '{fock_method}'. "
                f"Supported values are {', '.join(supported_fock_methods)}"
                )
        # Orthogonal J-side / K-side selectors
        self.do_gridk = self.fock_method in ("dfjgridk", "directjgridk")
        self.do_directjk = self.fock_method == "directjk"
        self.do_directj = self.fock_method in ("directjk", "directjgridk")

        self.directjk_schwarz_threshold = directjk_schwarz_threshold

        # Incremental (delta-density) Fock: after the first full J/K build,
        # later iterations contract only the spin-density deltas and
        # accumulate into the previous J/Ka/Kb (exact by linearity, up to
        # Schwarz screening -- which then acts on the shrinking deltas).
        # directjk only (the DF-K and grid-K engines contract Cocc, which
        # cannot be differenced), OFF by default (opt in with
        # incremental_fock=True).  Screening error accumulates across delta
        # builds, so every incremental_fock_rebuild-th build is a FULL
        # rebuild from the densities: one full build followed by
        # incremental_fock_rebuild - 1 delta builds (1 = rebuild every
        # iteration, 0 = never rebuild; negative values are rejected, they
        # would read as never-rebuild), which bounds the accumulated error.
        if incremental_fock and not self.do_directjk:
            raise ValueError(
                'incremental_fock requires fock_method="directjk" (the '
                'DF/grid exchange engines contract Cocc, not a density '
                'that can be differenced)')
        self.do_incremental_fock = incremental_fock
        self.incremental_fock_rebuild = int(incremental_fock_rebuild)
        if self.incremental_fock_rebuild < 0:
            raise ValueError(
                'incremental_fock_rebuild must be >= 0 (0 = never rebuild, '
                'n = a full rebuild every n-th build), got %d'
                % self.incremental_fock_rebuild)

        self.gridk_threshold_collocation = gridk_threshold_collocation
        self.gridk_grid_level = gridk_grid_level
        self.gridk_grid_family = gridk_grid_family
        self.gridk_fitting_algorithm = gridk_fitting_algorithm
        self.gridk_fitting_orthogonalization_cutoff = gridk_fitting_orthogonalization_cutoff
        self.gridk_fitting_cutoff = gridk_fitting_cutoff

        # SAD-reference exchange split: per spin K(D) = K_direct(D_sad) +
        # K_gridk(D - D_sad) with the expansion-consistent energy (see
        # compute_gridk_sad_reference); only active for the grid-K fock methods
        self.do_gridk_sad_reference = self.do_gridk and gridk_sad_reference
        # Per-spin reference data (D_ref = D_sad, K_ref = K_direct(D_ref),
        # K_ref . D_ref), populated by compute_gridk_sad_reference before the
        # SCF iterations
        self.gridk_sad_reference_data = None

        self.diis_max_nvector = diis_max_nvector
        self.cpu_diis = cpu_diis
        self.maxiter = maxiter
        self.g_convergence = g_convergence

        self.pcm_num_angular_points_per_heavy_atom = pcm_num_angular_points_per_heavy_atom
        self.pcm_num_angular_points_per_hydrogen_atom = pcm_num_angular_points_per_hydrogen_atom
        self.pcm_epsilon = pcm_epsilon
        self.pcm_x_prefactor = pcm_x_prefactor
        self.pcm_cutoff = pcm_cutoff
        self.pcm_convergence_tol = pcm_convergence_tol
        self.pcm_atomic_radii_scale = pcm_atomic_radii_scale
        self.pcm_q = None
        self.benchmark = benchmark
        self.level_shift_a = level_shift_a
        self.level_shift_b = level_shift_b

        self.dfk_int8_slice_count_start = dfk_int8_slice_count_start
        self.dfk_int8_modulus_count_start = dfk_int8_modulus_count_start
        self.dfk_int8_slice_count_end = dfk_int8_slice_count_end
        self.dfk_int8_modulus_count_end = dfk_int8_modulus_count_end

        # See if we're activating PCM
        self.do_pcm = self.pcm_epsilon != 1.0
        if self.pcm_epsilon < 1.0:
            raise RuntimeError('pcm_epsilon < 1.0')

        # See if we have ECPs
        self.do_ecp = False
        if self.ecp_basis is not None:
            self.do_ecp = self.ecp_basis.is_active

        if primary.natom != molecule.natom: raise RuntimeError('primary.natom != molecule.natom')
        if auxiliary.natom != molecule.natom: raise RuntimeError('auxiliary.natom != molecule.natom')
        if minao.natom != molecule.natom: raise RuntimeError('minao.natom != molecule.natom')
        if self.do_ecp:
            if ecp_basis.natom != molecule.natom: raise RuntimeError('ecp_basis.natom != molecule.natom')

        self.sizes = {}
        self.scalars = {}
        self.tensors = {}

        self.timer = Timer(detailed=self.print_timings or self.benchmark)

        self.is_solved = False
        self.is_converged = False

        self.gpu_xyz = GPUMatrix.from_numpy(self.molecule.xyz)

        # Store the effective Z if ECPs are present real Z otherwise
        self.gpu_Z = GPUMatrix.from_numpy(self.molecule.Z.reshape(-1, 1).astype(np.float64))
        GPUMatrixUtility.dscal(
            alpha=-1.0,
            x=self.gpu_Z,
            )
        if self.do_ecp:
            nelectron = np.array([atom.nelectron for atom in self.ecp_basis.atoms], dtype=np.float64).reshape(-1, 1)
            GPUMatrixUtility.daxpy(
                alpha=1.0,
                x=GPUMatrix.from_numpy(nelectron),
                y=self.gpu_Z,
                )
        self.Zeff = self.gpu_Z.to_numpy()

    # => Cuest Objects <= #

    @memoized_property
    def cuest_primary(self):
        return CuestAOBasis(
            handle=self.cuest_handle,
            basis=self.primary,
            )

    @memoized_property
    def cuest_auxiliary(self):
        return CuestAOBasis(
            handle=self.cuest_handle,
            basis=self.auxiliary,
            )

    @memoized_property
    def cuest_ecp_basis(self):
        if self.do_ecp:
            return CuestECPBasis(
                handle=self.cuest_handle,
                ecp_basis=self.ecp_basis,
                )
        else:
            return None

    @memoized_property
    def cuest_xc_grid(self):
        return CuestMolecularGrid(
            handle=self.cuest_handle,
            grid_level=self.xc_grid_level,
            xyz=self.molecule.xyz,
            Ns=self.molecule.N,
            family=self.xc_grid_family,
            )

    @memoized_property
    def cuest_nlc_grid(self):
        return CuestMolecularGrid(
            handle=self.cuest_handle,
            grid_level=self.nlc_grid_level,
            xyz=self.molecule.xyz,
            Ns=self.molecule.N,
            family=self.nlc_grid_family,
            )

    @memoized_property
    def cuest_ao_pair_list(self):
        return CuestAOPairList(
            handle=self.cuest_handle,
            basis=self.cuest_primary,
            xyz=self.molecule.xyz,
            threshold_pq=self.threshold_pq,
            )

    @memoized_property
    def cuest_oe_int_plan(self):
        return CuestOEIntPlan(
            handle=self.cuest_handle,
            basis=self.cuest_primary,
            ao_pair_list=self.cuest_ao_pair_list,
            )

    @memoized_property
    def cuest_df_int_plan(self):
        # No DF plan is needed in the direct-J modes
        if self.do_directj:
            return None
        return CuestDFIntPlan(
            handle=self.cuest_handle,
            primary=self.cuest_primary,
            auxiliary=self.cuest_auxiliary,
            ao_pair_list=self.cuest_ao_pair_list,
            exchange_scale=self.cuest_xc_int_plan.exchange_scale,
            lrc_exchange_scale=self.cuest_xc_int_plan.lrc_exchange_scale,
            lrc_omega=self.cuest_xc_int_plan.lrc_omega,
            df_fitting_eigenvalue_cutoff=self.df_fitting_eigenvalue_cutoff,
            # Grid-K uses the DF plan only for J, which requires integral direct
            integral_direct=self.df_integral_direct or self.do_gridk,
            )

    @memoized_property
    def cuest_gridk_grid(self):
        if not self.do_gridk:
            return None
        return CuestMolecularGrid(
            handle=self.cuest_handle,
            grid_level=self.gridk_grid_level,
            xyz=self.molecule.xyz,
            Ns=self.molecule.N,
            family=self.gridk_grid_family,
            )

    @memoized_property
    def cuest_gridk_int_plan(self):
        if not self.do_gridk:
            return None
        return CuestGridKIntPlan(
            handle=self.cuest_handle,
            grid=self.cuest_gridk_grid,
            exchange_scale=self.cuest_xc_int_plan.exchange_scale,
            lrc_exchange_scale=self.cuest_xc_int_plan.lrc_exchange_scale,
            lrc_omega=self.cuest_xc_int_plan.lrc_omega,
            threshold_collocation=self.gridk_threshold_collocation,
            fitting_algorithm=self.gridk_fitting_algorithm,
            fitting_orthogonalization_cutoff=self.gridk_fitting_orthogonalization_cutoff,
            fitting_cutoff=self.gridk_fitting_cutoff,
            oe_int_plan=self.cuest_oe_int_plan,
            )

    @memoized_property
    def cuest_direct_int_plan(self):
        # Built for both direct-J modes ("directjk" J and K, "directjgridk" J
        # only) and for the SAD-reference exchange
        if not (self.do_directj or (self.do_gridk_sad_reference and self.do_exchange)):
            return None
        return CuestDirectIntPlan(
            handle=self.cuest_handle,
            primary=self.cuest_primary,
            ao_pair_list=self.cuest_ao_pair_list,
            exchange_scale=self.cuest_xc_int_plan.exchange_scale,
            lrc_exchange_scale=self.cuest_xc_int_plan.lrc_exchange_scale,
            lrc_omega=self.cuest_xc_int_plan.lrc_omega,
            schwarz_threshold=self.directjk_schwarz_threshold,
            )

    @memoized_property
    def cuest_xc_int_plan(self):
        return CuestXCIntPlan(
            handle=self.cuest_handle,
            basis=self.cuest_primary,
            grid=self.cuest_xc_grid,
            functional=self.xc_functional,
            xc_threshold_collocation=self.xc_threshold_collocation,
            )

    @memoized_property
    def cuest_nlc_int_plan(self):
        return CuestXCIntPlan(
            handle=self.cuest_handle,
            basis=self.cuest_primary,
            grid=self.cuest_nlc_grid,
            functional=self.xc_functional,
            xc_threshold_collocation=self.nlc_threshold_collocation,
            name='Nonlocal XCIntPlan',
            print_functional=False,
            )

    @memoized_property
    def cuest_pcm_int_plan(self):
        if self.do_pcm:
            return CuestPCMIntPlan(
                handle=self.cuest_handle,
                intPlan=self.cuest_oe_int_plan,
                num_angular_points_per_heavy_atom=self.pcm_num_angular_points_per_heavy_atom,
                num_angular_points_per_hydrogen_atom=self.pcm_num_angular_points_per_hydrogen_atom,
                epsilon=self.pcm_epsilon,
                x_prefactor=self.pcm_x_prefactor,
                Ns=self.molecule.N,
                effective_nuclear_charges=-1.0 * self.Zeff,
                cutoff=self.pcm_cutoff,
                convergence_tol=self.pcm_convergence_tol,
                atomic_radii_scale=self.pcm_atomic_radii_scale,
                )
        else:
            return None


    @memoized_property
    def cuest_ecp_int_plan(self):
        if self.do_ecp:
            return CuestECPIntPlan(
                handle=self.cuest_handle,
                basis=self.cuest_primary,
                xyzs=self.molecule.xyz,
                numECPAtoms=self.cuest_ecp_basis.num_active_ecp,
                activeIndices=self.cuest_ecp_basis.ecp_indices,
                activeAtoms=self.cuest_ecp_basis.ecp_atoms,
                )
        else:
            return None

    @property
    def do_exchange(self):
        return self.cuest_xc_int_plan.exchange_scale != 0.0 or self.cuest_xc_int_plan.lrc_exchange_scale != 0.0

    # => Integrals <= #

    def compute_overlap(self):

        S = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        CuestOEIntCompute.overlap(
            handle=self.cuest_handle,
            oe_int_plan=self.cuest_oe_int_plan,
            Sptr=S.pointer,
            )

        return S

    def compute_kinetic(self):

        T = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            )

        CuestOEIntCompute.kinetic(
            handle=self.cuest_handle,
            oe_int_plan=self.cuest_oe_int_plan,
            Tptr=T.pointer,
            )

        return T

    def compute_potential(self):

        xyz = self.gpu_xyz
        q = self.gpu_Z

        V = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        CuestOEIntCompute.potential(
            handle=self.cuest_handle,
            oe_int_plan=self.cuest_oe_int_plan,
            Vptr=V.pointer,
            ncharge=self.molecule.natom,
            xyz=xyz.pointer,
            q=q.pointer,
            )

        return V

    def compute_dipole(self):

        origin = [0.0,0.0,0.0]
        orders = [0,0,0]

        M = []
        for n in range(3):
            Mtemp = GPUMatrix(
                nrows=self.primary.nao,
                ncols=self.primary.nao,
                )

            orders[n] = 1
            CuestOEIntCompute.multipole(
                handle=self.cuest_handle,
                oe_int_plan=self.cuest_oe_int_plan,
                multipoleOrder=orders,
                origin=origin,
                Mptr=Mtemp.pointer,
                )
            orders[n] = 0

            M.append(Mtemp)

        return M

    def compute_ecp_potential(self):

        Vecp = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        CuestECPIntCompute.potential(
            handle=self.cuest_handle,
            ecp_int_plan=self.cuest_ecp_int_plan,
            Vptr=Vecp.pointer,
            )

        return Vecp

    def compute_coulomb(
        self,
        D,
        ):

        J = D.clone()

        if self.do_directj:
            CuestDirectIntCompute.coulomb(
                handle=self.cuest_handle,
                direct_int_plan=self.cuest_direct_int_plan,
                Dptr=D.pointer,
                Jptr=J.pointer,
                schwarz_threshold=self.directjk_schwarz_threshold,
                num_density_matrices=1,
                )
        else:
            CuestDFIntCompute.coulomb(
                handle=self.cuest_handle,
                df_int_plan=self.cuest_df_int_plan,
                Jptr=J.pointer,
                Dptr=D.pointer,
                )

        return J

    def compute_exchange(
        self,
        Coccs,
        dfk_int8_slice_count=None,
        dfk_int8_modulus_count=None,
        ):

        nao = self.primary.nao

        Ks = []
        for _ in Coccs:
            K = GPUMatrix(
                nrows=nao,
                ncols=nao,
                dtype=np.double,
                )
            Ks.append(K)

        # Return zeros if no exchange to calculate
        if not self.do_exchange:
            return Ks

        noccs = [_.shape[0] for _ in Coccs]
        Coccptrs = [_.pointer for _ in Coccs]
        Kptrs = [_.pointer for _ in Ks]

        if self.do_directjk:
            CuestDirectIntCompute.symmetric_exchange(
                handle=self.cuest_handle,
                direct_int_plan=self.cuest_direct_int_plan,
                noccs=noccs,
                Coccptrs=Coccptrs,
                Kptrs=Kptrs,
                schwarz_threshold=self.directjk_schwarz_threshold,
                )
        elif self.do_gridk:
            CuestGridKIntCompute.symmetric_exchange(
                handle=self.cuest_handle,
                gridk_int_plan=self.cuest_gridk_int_plan,
                noccs=noccs,
                Coccptrs=Coccptrs,
                Kptrs=Kptrs,
                )
        else:
            CuestDFIntCompute.symmetric_exchange(
                handle=self.cuest_handle,
                df_int_plan=self.cuest_df_int_plan,
                noccs=noccs,
                Coccptrs=Coccptrs,
                Kptrs=Kptrs,
                dfk_int8_slice_count=dfk_int8_slice_count,
                dfk_int8_modulus_count=dfk_int8_modulus_count,
                )

        return Ks

    def compute_exchange_density(
        self,
        Ds,
        ):

        # Density-matrix twin of compute_exchange. Each density must be
        # symmetric but need not be positive semidefinite (a difference
        # density is valid input). The DF-K engine is coefficient matrix
        # based, so this entry point supports only the direct and grid-K
        # engines.

        nao = self.primary.nao

        Ks = []
        for _ in Ds:
            K = GPUMatrix(
                nrows=nao,
                ncols=nao,
                dtype=np.double,
                )
            Ks.append(K)

        # Return zeros if no exchange to calculate
        if not self.do_exchange:
            return Ks

        Dptrs = [_.pointer for _ in Ds]
        Kptrs = [_.pointer for _ in Ks]

        if self.do_directjk:
            CuestDirectIntCompute.symmetric_exchange_density(
                handle=self.cuest_handle,
                direct_int_plan=self.cuest_direct_int_plan,
                Dptrs=Dptrs,
                Kptrs=Kptrs,
                schwarz_threshold=self.directjk_schwarz_threshold,
                )
        elif self.do_gridk:
            CuestGridKIntCompute.symmetric_exchange_density(
                handle=self.cuest_handle,
                gridk_int_plan=self.cuest_gridk_int_plan,
                Dptrs=Dptrs,
                Kptrs=Kptrs,
                )
        else:
            raise NotImplementedError(
                'Density-based exchange is not wired for fock_method="dfjk" '
                '(the DF-K engine is coefficient matrix based); use '
                'compute_exchange')

        return Ks

    def compute_nonsymmetric_exchange(
        self,
        Cocc_left : GPUMatrix,
        Coccs_right : list,
        ):

        nao = self.primary.nao
        nRight = len(Coccs_right)

        Kout = []
        for i in range(nRight):
            Ktemp = GPUMatrix(
                nrows=nao,
                ncols=nao,
                dtype=np.double,
                )
            Kout.append(Ktemp)

        # Return zeros if no exchange to calculate (valid for every
        # fock_method, so this precedes the directjk capability guard)
        if not self.do_exchange:
            return Kout

        if self.do_directjk:
            raise NotImplementedError(
                'Nonsymmetric exchange is not wired for fock_method="directjk" '
                '(the direct C API is symmetric-only); use "dfjk"/"dfjgridk"')

        nocc = Cocc_left.shape[0]

        for i, C in enumerate(Coccs_right):
            if C.shape != (nocc, nao):
                raise ValueError(
                    f'Coccs_right[{i}] has shape {C.shape}, expected ({nocc}, {nao})'
                    )

        if self.do_gridk:
            CuestGridKIntCompute.nonsymmetric_exchange(
                handle=self.cuest_handle,
                gridk_int_plan=self.cuest_gridk_int_plan,
                nocc=nocc,
                Cocc_left_ptr=Cocc_left.pointer,
                Cocc_right_ptrs=[_.pointer for _ in Coccs_right],
                Kptrs=[_.pointer for _ in Kout],
                )
        else:
            CuestDFIntCompute.nonsymmetric_exchange(
                handle=self.cuest_handle,
                df_int_plan=self.cuest_df_int_plan,
                nocc=nocc,
                Cocc_left_ptr=Cocc_left.pointer,
                Cocc_right_ptrs=[_.pointer for _ in Coccs_right],
                Kptrs=[_.pointer for _ in Kout],
                )

        return Kout

    def compute_gridk_sad_reference(self):

        # SAD-reference exchange: the exchange energy of each spin channel is
        # expanded exactly about the SAD guess spin density, with only the
        # quadratic remainder on the grid,
        # E_K = -1/2 sum_s [D_ref . K_ref + 2 dD . K_ref + dD . K_gridk(dD)],
        # K_ref = K_direct(D_ref), dD = D_s - D_ref,s.
        # The variational Fock matrices are K(D) = K_ref + K_gridk(dD) and
        # the energy error is second order in the difference densities. Alpha
        # and beta references are kept separate (the SAD guess need not be
        # spin-restricted). The per-spin K_ref and K_ref . D_ref are computed
        # once here.

        self.scalars['E_gridk_sad_reference'] = 0.0

        if not (self.do_gridk_sad_reference and self.do_exchange):
            return

        nao = self.primary.nao

        # Clone the guess spin densities: the SCF loop rebinds tensors['Da']
        # and tensors['Db'] each iteration. The occupation-weighted SAD factors
        # are kept with them (the guess sets D_ref,s = Cocc_ref,s^T Cocc_ref,s);
        # the gradient's bilinear direct builds take the factors as input
        D_refs = [self.tensors['Da'].clone(), self.tensors['Db'].clone()]
        Cocc_refs = [self.tensors['Cocca'].clone(), self.tensors['Coccb'].clone()]

        Kdir_refs = []
        for _ in D_refs:
            Kdir_refs.append(GPUMatrix(
                nrows=nao,
                ncols=nao,
                dtype=np.double,
                ))

        CuestDirectIntCompute.symmetric_exchange_density(
            handle=self.cuest_handle,
            direct_int_plan=self.cuest_direct_int_plan,
            Dptrs=[_.pointer for _ in D_refs],
            Kptrs=[_.pointer for _ in Kdir_refs],
            schwarz_threshold=self.directjk_schwarz_threshold,
            )

        self.gridk_sad_reference_data = []
        for D_ref, Cocc_ref, Kdir_ref in zip(D_refs, Cocc_refs, Kdir_refs):
            self.gridk_sad_reference_data.append({
                'Dref': D_ref,
                'Coccref': Cocc_ref,
                'Kref': Kdir_ref,
                'Kref_dot_Dref': GPUMatrixUtility.ddot(
                    x=D_ref,
                    y=Kdir_ref,
                    ),
                })

    def compute_gridk_sad_reference_exchange(
        self,
        Ds,
        ):

        # SAD-reference exchange build, per spin
        # K(D) = K_ref + K_gridk(dD),  dD = D_s - D_ref,s,
        # with the difference densities formed explicitly and grid-K applied
        # to them (Ds ordered as the references, [Da, Db]). Returns
        # (Ks, dDs, Kgrid_dDs); dDs and Kgrid_dDs are reused by the energy
        # expression.

        dDs = []
        for D, ref in zip(Ds, self.gridk_sad_reference_data):
            dD = D.clone()
            GPUMatrixUtility.daxpy(
                alpha=-1.0,
                x=ref['Dref'],
                y=dD,
                )
            dDs.append(dD)

        Kgrid_dDs = self.compute_exchange_density(Ds=dDs)

        Ks = []
        for Kgrid_dD, ref in zip(Kgrid_dDs, self.gridk_sad_reference_data):
            K = Kgrid_dD.clone()
            GPUMatrixUtility.daxpy(
                alpha=1.0,
                x=ref['Kref'],
                y=K,
                )
            Ks.append(K)

        return Ks, dDs, Kgrid_dDs

    def compute_gridk_sad_reference_energy(
        self,
        dDs,
        Kgrid_dDs,
        ):

        # The SAD-reference exchange energy, evaluated directly from the
        # per-spin expansion
        # E_K = -1/2 sum_s [D_ref . K_ref + 2 dD . K_ref + dD . K_gridk(dD)]:
        # the reference and cross terms are exact, only the quadratic
        # remainder is on the grid (lists ordered as the references,
        # [alpha, beta]). Returns 0.0 unless the reference is active.

        if self.gridk_sad_reference_data is None:
            return 0.0

        E_gridk_sad_reference = 0.0
        for dD, Kgrid_dD, ref in zip(dDs, Kgrid_dDs, self.gridk_sad_reference_data):
            E_gridk_sad_reference -= 0.5 * ref['Kref_dot_Dref']
            E_gridk_sad_reference -= 1.0 * GPUMatrixUtility.ddot(
                x=dD,
                y=ref['Kref'],
                )
            E_gridk_sad_reference -= 0.5 * GPUMatrixUtility.ddot(
                x=dD,
                y=Kgrid_dD,
                )
        return E_gridk_sad_reference

    def compute_local_exchange_correlation(
        self,
        Cocca,
        Coccb,
        ):

        if self.xc_functional == 0:
            return 0.0, None, None

        Vxca = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        Vxcb = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        Exc = CuestXCIntCompute.local_exchange_correlation_uks(
            handle=self.cuest_handle,
            xc_int_plan=self.cuest_xc_int_plan,
            Vxcptr=Vxca.pointer,
            nocc=Cocca.shape[0],
            Coccptr=Cocca.pointer,
            Vxcbptr=Vxcb.pointer,
            noccb=Coccb.shape[0],
            Coccbptr=Coccb.pointer,
            )

        return Exc, Vxca, Vxcb

    def compute_nonlocal_exchange_correlation(
        self,
        Cocca,
        Coccb,
        ):

        if self.cuest_nlc_int_plan.vv10_scale == 0.0:
            return 0.0, None

        Vxc = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        Exc = CuestXCIntCompute.nonlocal_exchange_correlation_uks(
            handle=self.cuest_handle,
            xc_int_plan=self.cuest_nlc_int_plan,
            Vxcptr=Vxc.pointer,
            nocc=Cocca.shape[0],
            Coccptr=Cocca.pointer,
            noccb=Coccb.shape[0],
            Coccbptr=Coccb.pointer,
            )
        return Exc, Vxc

    def compute_pcm_energy_and_potential(
        self,
        q,
        D,
        ):

        outQ = GPUMatrix(
            nrows=self.cuest_pcm_int_plan.npoint,
            ncols=1,
            dtype=np.double,
            )

        V = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        Epcm, converged, residual = CuestPCMIntCompute.potential(
            handle=self.cuest_handle,
            pcm_int_plan=self.cuest_pcm_int_plan,
            Dptr=D.pointer,
            inQptr=q.pointer,
            outQptr=outQ.pointer,
            Vptr=V.pointer,
            )

        return Epcm, V, outQ, converged

    # => Integral Derivatives (Gradients) <= #

    def compute_pcm_gradient(
        self,
        q,
        D,
        ):

        npoint = self.cuest_pcm_int_plan.npoint

        outQ = GPUMatrix(
            nrows=self.cuest_pcm_int_plan.npoint,
            ncols=1,
            dtype=np.double,
            )

        G = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        converged, residual = CuestPCMIntCompute.potential_gradient(
            handle=self.cuest_handle,
            pcm_int_plan=self.cuest_pcm_int_plan,
            Dptr=D.pointer,
            inQptr=q.pointer,
            outQptr=outQ.pointer,
            Gptr=G.pointer,
            )
        if not converged:
            raise RuntimeError('PCM not converged')

        return G

    def compute_overlap_gradient(
        self,
        W, # Energy-weighted density matrix
        ):

        G = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        CuestOEIntCompute.overlap_gradient(
            handle=self.cuest_handle,
            oe_int_plan=self.cuest_oe_int_plan,
            Gptr=G.pointer,
            Wptr=W.pointer,
            )

        return G

    def compute_kinetic_gradient(
        self,
        D, # Density matrix
        ):

        G = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        CuestOEIntCompute.kinetic_gradient(
            handle=self.cuest_handle,
            oe_int_plan=self.cuest_oe_int_plan,
            Gptr=G.pointer,
            Dptr=D.pointer,
            )

        return G

    def compute_potential_gradient(
        self,
        D, # Density matrix
        ):

        Gbasis = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )
        Gcharge = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        CuestOEIntCompute.potential_gradient(
            handle=self.cuest_handle,
            oe_int_plan=self.cuest_oe_int_plan,
            ncharge=self.molecule.natom,
            xyz=self.gpu_xyz.pointer,
            q=self.gpu_Z.pointer,
            Dptr=D.pointer,
            GptrBasis=Gbasis.pointer,
            GptrCharge=Gcharge.pointer,
            )

        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=Gcharge,
            y=Gbasis,
            )

        return Gbasis

    def compute_ecp_gradient(
        self,
        D,
        ):

        Gecp = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        CuestECPIntCompute.potential_gradient(
            handle=self.cuest_handle,
            ecp_int_plan=self.cuest_ecp_int_plan,
            Dptr=D.pointer,
            Gptr=Gecp.pointer,
            )

        return Gecp

    def compute_coulomb_gradient_direct(
        self,
        *,
        scaleJ,
        DJ,
        G,
        ):

        self.timer.start(key='dJ', stream_handle=self.cuest_handle.stream_handle)
        GJ = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )
        CuestDirectIntCompute.coulomb_gradient(
            handle=self.cuest_handle,
            direct_int_plan=self.cuest_direct_int_plan,
            Dptr=DJ.pointer,
            Gptr=GJ.pointer,
            schwarz_threshold=self.directjk_schwarz_threshold,
            num_density_matrices=1,
            )
        GPUMatrixUtility.daxpy(
            alpha=scaleJ,
            x=GJ,
            y=G,
            )
        self.timer.stop(key='dJ', stream_handle=self.cuest_handle.stream_handle)

    def compute_exchange_gradient_direct(
        self,
        *,
        scaleK,
        CoccsK,
        G,
        ):

        self.timer.start(key='dK', stream_handle=self.cuest_handle.stream_handle)
        if scaleK != 0.0 and self.do_exchange:
            GKs = []
            for _ in CoccsK:
                GK = GPUMatrix(
                    nrows=self.primary.natom,
                    ncols=3,
                    dtype=np.double,
                    )
                GKs.append(GK)
            CuestDirectIntCompute.symmetric_exchange_gradient(
                handle=self.cuest_handle,
                direct_int_plan=self.cuest_direct_int_plan,
                noccs=[_.shape[0] for _ in CoccsK],
                Coccptrs=[_.pointer for _ in CoccsK],
                Gptrs=[_.pointer for _ in GKs],
                schwarz_threshold=self.directjk_schwarz_threshold,
                )
            for GK in GKs:
                GPUMatrixUtility.daxpy(
                    alpha=scaleK,
                    x=GK,
                    y=G,
                    )
        self.timer.stop(key='dK', stream_handle=self.cuest_handle.stream_handle)

    def compute_exchange_gradient_gridk(
        self,
        *,
        scaleK,
        CoccsK,
        G,
        ):

        self.timer.start(key='dK', stream_handle=self.cuest_handle.stream_handle)
        if scaleK != 0.0 and self.do_exchange:
            GKs = []
            for _ in CoccsK:
                GK = GPUMatrix(
                    nrows=self.primary.natom,
                    ncols=3,
                    dtype=np.double,
                    )
                GKs.append(GK)
            CuestGridKIntCompute.symmetric_exchange_gradient(
                handle=self.cuest_handle,
                gridk_int_plan=self.cuest_gridk_int_plan,
                noccs=[_.shape[0] for _ in CoccsK],
                Coccptrs=[_.pointer for _ in CoccsK],
                Gptrs=[_.pointer for _ in GKs],
                )
            for GK in GKs:
                GPUMatrixUtility.daxpy(
                    alpha=scaleK,
                    x=GK,
                    y=G,
                    )
        self.timer.stop(key='dK', stream_handle=self.cuest_handle.stream_handle)

    def compute_gridk_sad_reference_gradient(
        self,
        *,
        scaleK,
        CoccsK,
        G,
        ):

        # Exchange gradient for the SAD-reference split (replaces the plain
        # grid-K exchange gradient):
        # E_K^x = scaleK * sum_s [Tr(K_direct^x(D_ref) D_ref)
        #                         + 2 Tr(K_direct^x(D_ref) dD)
        #                         + Tr(K_gridk^x(dD) dD)],  dD = D_s - D_ref,s,
        # with scaleK (-0.5) carrying the spin factor and the spin densities
        # formed from CoccsK (ordered as the references, [Cocca, Coccb]). The
        # exact direct terms are evaluated with bilinear builds via
        # Tr(K^x(D_ref) D_ref) + 2 Tr(K^x(D_ref) dD)
        #     = 2 Tr(K^x(D_ref) D) - Tr(K^x(D_ref) D_ref),
        # pairing the atom-block-diagonal D_ref (as its SAD factor Cocc_ref)
        # against D and against itself, so every direct term screens on
        # D_ref instead of a full symmetric build on D. The scheme is
        # variational and D_ref is geometry-independent, so there are no
        # response terms.

        self.timer.start(key='dK', stream_handle=self.cuest_handle.stream_handle)
        if scaleK != 0.0 and self.do_exchange:

            if self.gridk_sad_reference_data is None:
                raise RuntimeError('compute_gridk_sad_reference must be called first')

            natom = self.primary.natom

            Ds = []
            for Cocc in CoccsK:
                Ds.append(GPUMatrixUtility.matrix_multiply(
                    mat1=Cocc,
                    mat2=Cocc,
                    transpose1=True,
                    transpose2=False,
                    ))

            dDs = []
            for D, ref in zip(Ds, self.gridk_sad_reference_data):
                dD = D.clone()
                GPUMatrixUtility.daxpy(
                    alpha=-1.0,
                    x=ref['Dref'],
                    y=dD,
                    )
                dDs.append(dD)
            del Ds, D

            # Direct derivative as bilinear builds pairing the reference
            # factors with the full spin densities and with themselves, in one
            # batched call; grid derivative on the difference densities
            Coccrefs = [ref['Coccref'] for ref in self.gridk_sad_reference_data]
            direct_Coccs = Coccrefs + Coccrefs
            direct_Coccs2 = list(CoccsK) + Coccrefs
            Gdirs = []
            for _ in direct_Coccs:
                Gdirs.append(GPUMatrix(
                    nrows=natom,
                    ncols=3,
                    dtype=np.double,
                    ))
            Ggrids = []
            for _ in dDs:
                Ggrids.append(GPUMatrix(
                    nrows=natom,
                    ncols=3,
                    dtype=np.double,
                    ))

            self.timer.start(key='dK_direct', stream_handle=self.cuest_handle.stream_handle)
            CuestDirectIntCompute.symmetric_exchange_bilinear_gradient(
                handle=self.cuest_handle,
                direct_int_plan=self.cuest_direct_int_plan,
                noccs1=[_.shape[0] for _ in direct_Coccs],
                Coccptrs1=[_.pointer for _ in direct_Coccs],
                noccs2=[_.shape[0] for _ in direct_Coccs2],
                Coccptrs2=[_.pointer for _ in direct_Coccs2],
                Gptrs=[_.pointer for _ in Gdirs],
                schwarz_threshold=self.directjk_schwarz_threshold,
                )
            self.timer.stop(key='dK_direct', stream_handle=self.cuest_handle.stream_handle)
            self.timer.start(key='dK_gridk', stream_handle=self.cuest_handle.stream_handle)
            CuestGridKIntCompute.symmetric_exchange_gradient_density(
                handle=self.cuest_handle,
                gridk_int_plan=self.cuest_gridk_int_plan,
                Dptrs=[_.pointer for _ in dDs],
                Gptrs=[_.pointer for _ in Ggrids],
                )
            self.timer.stop(key='dK_gridk', stream_handle=self.cuest_handle.stream_handle)

            nmat = len(dDs)
            for i in range(nmat):
                GPUMatrixUtility.daxpy(
                    alpha=2.0 * scaleK,
                    x=Gdirs[i],
                    y=G,
                    )
                GPUMatrixUtility.daxpy(
                    alpha=-scaleK,
                    x=Gdirs[nmat + i],
                    y=G,
                    )
                GPUMatrixUtility.daxpy(
                    alpha=scaleK,
                    x=Ggrids[i],
                    y=G,
                    )
        self.timer.stop(key='dK', stream_handle=self.cuest_handle.stream_handle)

    def compute_coulomb_and_exchange_gradient_df(
        self,
        *,
        scaleJ,
        DJ,
        scaleK,
        CoccsK,
        G,
        ):

        CuestDFIntCompute.coulomb_and_exchange_gradient(
            handle=self.cuest_handle,
            df_int_plan=self.cuest_df_int_plan,
            scaleJ=scaleJ,
            DJptr=DJ.pointer,
            scaleK=scaleK,
            noccsK=[_.shape[0] for _ in CoccsK],
            CoccsKptrs=[_.pointer for _ in CoccsK],
            Gptr=G.pointer,
            )

    def compute_coulomb_and_exchange_gradient(
        self,
        *,
        scaleJ,
        DJ,
        scaleK,
        CoccsK,
        ):

        # Every engine returns d/dR Tr(J/K(D) D), so the same scaleJ/scaleK
        # apply on all paths
        G = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        if self.do_directjk:
            self.compute_coulomb_gradient_direct(
                scaleJ=scaleJ,
                DJ=DJ,
                G=G,
                )
            self.compute_exchange_gradient_direct(
                scaleK=scaleK,
                CoccsK=CoccsK,
                G=G,
                )
        elif self.do_directj:
            self.compute_coulomb_gradient_direct(
                scaleJ=scaleJ,
                DJ=DJ,
                G=G,
                )
            if self.do_gridk_sad_reference:
                self.compute_gridk_sad_reference_gradient(
                    scaleK=scaleK,
                    CoccsK=CoccsK,
                    G=G,
                    )
            else:
                self.compute_exchange_gradient_gridk(
                    scaleK=scaleK,
                    CoccsK=CoccsK,
                    G=G,
                    )
        elif self.do_gridk:
            # The integral-direct DF plan provides only the Coulomb gradient
            # here (scaleK = 0); Grid-K provides the exchange gradient
            self.timer.start(key='dJ', stream_handle=self.cuest_handle.stream_handle)
            self.compute_coulomb_and_exchange_gradient_df(
                scaleJ=scaleJ,
                DJ=DJ,
                scaleK=0.0,
                CoccsK=CoccsK,
                G=G,
                )
            self.timer.stop(key='dJ', stream_handle=self.cuest_handle.stream_handle)
            if self.do_gridk_sad_reference:
                self.compute_gridk_sad_reference_gradient(
                    scaleK=scaleK,
                    CoccsK=CoccsK,
                    G=G,
                    )
            else:
                self.compute_exchange_gradient_gridk(
                    scaleK=scaleK,
                    CoccsK=CoccsK,
                    G=G,
                    )
        else:
            if not self.do_exchange:
                scaleK = 0.0
            self.compute_coulomb_and_exchange_gradient_df(
                scaleJ=scaleJ,
                DJ=DJ,
                scaleK=scaleK,
                CoccsK=CoccsK,
                G=G,
                )

        return G

    def compute_local_exchange_correlation_gradient(
        self,
        Cocca,
        Coccb,
        ):

        G = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )
        if self.xc_functional == 0:
            G.zero()
            return G

        CuestXCIntCompute.local_exchange_correlation_uks_gradient(
            handle=self.cuest_handle,
            xc_int_plan=self.cuest_xc_int_plan,
            Gptr=G.pointer,
            nocc=Cocca.shape[0],
            Coccptr=Cocca.pointer,
            noccb=Coccb.shape[0],
            Coccbptr=Coccb.pointer,
            )

        return G

    def compute_nonlocal_exchange_correlation_gradient(
        self,
        Cocca,
        Coccb,
        ):

        G = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        if self.cuest_nlc_int_plan.vv10_scale == 0.0:
            G.zero()
            return G

        CuestXCIntCompute.nonlocal_exchange_correlation_uks_gradient(
            handle=self.cuest_handle,
            xc_int_plan=self.cuest_nlc_int_plan,
            Gptr=G.pointer,
            nocc=Cocca.shape[0],
            Coccptr=Cocca.pointer,
            noccb=Coccb.shape[0],
            Coccbptr=Coccb.pointer,
            )

        return G

    # => Main Solve Routines <= #

    def solve(self):

        self.timer.start(key='solve', stream_handle=self.cuest_handle.stream_handle, always=True)

        self.is_solved = False
        self.is_converged = False

        # Plan initializations (to cleanly separate from iterations).  The orthogonalization
        # calls for S, so most of these are automatically instantiated in that routine.  Build
        # them here to get an idea of how long they take.

        # These functions have no dependencies
        self.timer.start(key='primary_basis', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_primary
        self.timer.stop(key='primary_basis', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='auxiliary_basis', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_auxiliary
        self.timer.stop(key='auxiliary_basis', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='xc_grid', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_xc_grid
        self.timer.stop(key='xc_grid', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='nlc_grid', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_nlc_grid
        self.timer.stop(key='nlc_grid', stream_handle=self.cuest_handle.stream_handle)

        # Depends on cuest_primary
        self.timer.start(key='ao_pair_list', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_ao_pair_list
        self.timer.stop(key='ao_pair_list', stream_handle=self.cuest_handle.stream_handle)

        # Depends on cuest_primary and cuest_ao_pair_list
        self.timer.start(key='oe_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_oe_int_plan
        self.timer.stop(key='oe_int_plan', stream_handle=self.cuest_handle.stream_handle)

        # Depends on cuest_primary and cuest_xc_grid
        self.timer.start(key='xc_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_xc_int_plan
        self.timer.stop(key='xc_int_plan', stream_handle=self.cuest_handle.stream_handle)

        # Depends on cuest_primary, cuest_auxiliary, cuest_ao_pair_list, and
        # cuest_xc_int_plan (None in the direct-J modes, where the DF plan
        # is disabled).
        self.timer.start(key='df_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_df_int_plan
        self.timer.stop(key='df_int_plan', stream_handle=self.cuest_handle.stream_handle)

        # Grid-K exchange (built for fock_method "dfjgridk" and "directjgridk")
        # Depends on cuest_primary, cuest_ao_pair_list, cuest_gridk_grid, and cuest_xc_int_plan
        self.timer.start(key='gridk_grid', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_gridk_grid
        self.timer.stop(key='gridk_grid', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='gridk_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_gridk_int_plan
        self.timer.stop(key='gridk_int_plan', stream_handle=self.cuest_handle.stream_handle)

        # Direct 4-center J/K (built for fock_method "directjk", J-only for
        # "directjgridk", and for the SAD-reference exchange)
        # Depends on cuest_primary and cuest_ao_pair_list
        self.timer.start(key='direct_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_direct_int_plan
        self.timer.stop(key='direct_int_plan', stream_handle=self.cuest_handle.stream_handle)

        # Depends on cuest_primary and cuest_nlc_grid
        self.timer.start(key='nlc_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_nlc_int_plan
        self.timer.stop(key='nlc_int_plan', stream_handle=self.cuest_handle.stream_handle)

        # Depends on cuest_oe_int_plan
        self.timer.start(key='pcm_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_pcm_int_plan
        self.timer.stop(key='pcm_int_plan', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='ecp_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_ecp_int_plan
        self.timer.stop(key='ecp_int_plan', stream_handle=self.cuest_handle.stream_handle)

        self.header()

        self.timer.start(key='guess', stream_handle=self.cuest_handle.stream_handle)
        self.compute_guess()
        self.timer.stop(key='guess', stream_handle=self.cuest_handle.stream_handle)

        # SAD-reference exchange data, computed once from the guess densities
        # before the SCF iterations start
        if self.do_gridk_sad_reference:
            self.timer.start(key='Kref', stream_handle=self.cuest_handle.stream_handle)
            self.compute_gridk_sad_reference()
            self.timer.stop(key='Kref', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='orthogonalize', stream_handle=self.cuest_handle.stream_handle)
        self.compute_orthogonalization()
        self.timer.stop(key='orthogonalize', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='occupations', stream_handle=self.cuest_handle.stream_handle)
        self.compute_occupations()
        self.timer.stop(key='occupations', stream_handle=self.cuest_handle.stream_handle)

        if self.print_level > 1 and self.print_timings and not self.do_directj:
            print('DFIntPlan Initialize Time: %11.3E [s]\n' % (self.timer.times['df_int_plan']))

        self.timer.start(key='solve_diis', stream_handle=self.cuest_handle.stream_handle)
        self.solve_diis()
        self.timer.stop(key='solve_diis', stream_handle=self.cuest_handle.stream_handle)

        self.timer.stop(key='solve', stream_handle=self.cuest_handle.stream_handle, always=True)

        self.trailer()

        self.is_solved = True


    def header(self):

        if self.print_level:
            print('==> cuEST UHF (Python) <==\n')

        if self.print_level > 1:

            print('%-19s = %11d' % ('natom', self.molecule.natom))
            print('%-19s = %11d' % ('charge', self.charge))
            print('')

            print('Primary Basis: %s\n' % (self.primary_name if self.primary_name is not None else ''))
            print(self.primary)

            print('Auxiliary Basis: %s\n' % (self.auxiliary_name if self.auxiliary_name is not None else ''))
            print(self.auxiliary)

            print('MinAO Basis: %s\n' % (self.minao_name if self.minao_name is not None else ''))
            print(self.minao)

            print(self.cuest_xc_int_plan)

            print('XC Grid:\n')
            print(self.cuest_xc_grid)

            print(self.cuest_nlc_int_plan)

            print('Nonlocal XC Grid:\n')
            print(self.cuest_nlc_grid)

        # The Fock method and the pair screening it runs with are reported at
        # every print level; the plans behind them stay level 2
        if self.print_level > 0:

            if self.do_directj and self.do_gridk:
                print('Fock Method: Direct-J / Grid-K (grid-based exchange)\n')
            elif self.do_gridk:
                print('Fock Method: DF-J / Grid-K (grid-based exchange)\n')
            elif self.do_directjk:
                print('Fock Method: Direct-J / Direct-K (integral-direct 4-center%s)\n'
                      % (', incremental' if self.do_incremental_fock else ''))
            else:
                print('Fock Method: DF-J / DF-K\n')

            print('%-19s = %11.3E\n' % ('threshold_pq', self.threshold_pq))

        if self.print_level > 1:

            if self.do_gridk:
                print(self.cuest_gridk_int_plan)
                print('Grid-K Grid:\n')
                print(self.cuest_gridk_grid)

            if self.cuest_df_int_plan is not None:
                print(self.cuest_df_int_plan)

            if self.cuest_direct_int_plan is not None:
                print(self.cuest_direct_int_plan)

            if self.do_gridk_sad_reference:
                print('Grid-K SAD-Reference Exchange: K_s = K_direct(D_sad,s) + K_gridk(D_s - D_sad,s)\n')

            if self.do_pcm:
                print('PCM:\n')
                print(self.cuest_pcm_int_plan)

            if self.do_ecp:
                print('ECP Basis: %s\n' % (self.ecp_basis_name if self.ecp_basis_name is not None else ''))
                print(self.ecp_basis)

    def trailer(self):

        if self.print_level:
            print('cuEST UHF Time: %11.3E [s]\n' % (self.timer.times['solve']))

        if self.print_timings:
            print('cuEST UHF Timing breakdown:\n')
            print(self.timer.string())

        if self.print_level:
            print('==> End cuEST UHF (Python) <==\n')

    def compute_guess(self):

        self.sad_guess = SADGuess.build(
            molecule=self.molecule,
            primary=self.primary,
            minao=self.minao,
            ecp_basis=self.ecp_basis,
            charge=self.charge,
            multiplicity=self.multiplicity,
            )

        # For now this will be size (nminao, nao), which is usually larger than
        # (nocc, nao) due to fractional occupations in SAD
        self.tensors['Cocca'] = GPUMatrix.from_numpy(self.sad_guess.compute_Cocc())
        self.tensors['Coccb'] = GPUMatrix.from_numpy(self.sad_guess.compute_Cocc())

        # For now this is not a pure state, and corresponds to the neutral SAD
        # guess
        self.tensors['Da'] = GPUMatrix.from_numpy(self.sad_guess.compute_Docc())
        self.tensors['Db'] = GPUMatrix.from_numpy(self.sad_guess.compute_Docc())

    def compute_orthogonalization(self):

        if self.print_level > 0:
            print('Orthogonalization:\n')

        gridk_int_plan = self.cuest_gridk_int_plan
        do_gridk_fitting = (
            gridk_int_plan is not None
            and gridk_int_plan.fitting_algorithm != 'none'
            )

        self.timer.start(key='S', stream_handle=self.cuest_handle.stream_handle)
        S = self.compute_overlap()
        self.timer.stop(key='S', stream_handle=self.cuest_handle.stream_handle)

        # For consistency, use the same X matrix for Grid-K fitting and SCF orthogonalization
        if do_gridk_fitting:

            nao = gridk_int_plan.primary.nao
            nmo = gridk_int_plan.fitting_num_mo

            X = GPUMatrix(
                nrows=nmo,
                ncols=nao,
                dtype=np.double,
                )

            gridk_int_plan.query_canonical_orthogonalization_matrix(
                Xptr=X.pointer,
                num_mo=nmo,
                nao=nao,
                )

            self.tensors['S'] = S
            self.tensors['X'] = X

            if self.print_level > 0:
                print('%-19s = %11.3E' % ('threshold_canonical', gridk_int_plan.fitting_orthogonalization_cutoff))
                print('%-19s = %11d' % ('nao', nao))
                print('%-19s = %11d' % ('nmo', nmo))
                print('%-19s = %11d' % ('ndiscard', nao - nmo))
                print('')

            return

        s, U = GPUMatrixUtility.eigh(matrix=S)
        s = s.to_numpy().reshape(-1)
        U = U.to_numpy()

        # Canonical orthogonalization is traditionally done on an absolute eigenvalue threshold basis
        sind = s > self.threshold_canonical

        s2 = s[sind]
        U2 = U[:, sind]

        X = np.einsum('ij,j->ji', U2, s2**(-0.5))

        # This should be numerically zero:
        # d = np.max(np.abs(np.dot(np.dot(X, S), X.T) - np.eye(X.shape[0])))

        self.tensors['S'] = S
        self.tensors['X'] = GPUMatrix.from_numpy(X)

        if self.print_level > 0:
            print('%-19s = %11.3E' % ('threshold_canonical', self.threshold_canonical))
            print('%-19s = %11d' % ('nao', X.shape[1]))
            print('%-19s = %11d' % ('nmo', X.shape[0]))
            print('%-19s = %11d' % ('ndiscard', X.shape[1] - X.shape[0]))
            print('')

    def compute_occupations(self):

        if self.print_level > 0:
            print('Occupations:\n')

        Q = -np.sum(self.Zeff) - self.charge
        d = self.multiplicity - 1

        if (Q + d) % 2 != 0:
            raise RuntimeError(
                f"Incompatible electron count Q={Q} and multiplicity={self.multiplicity}"
            )

        nocc_a = int((Q + d) / 2)
        nocc_b = int((Q - d) / 2)

        nvir_a = self.tensors['X'].shape[0] - nocc_a
        nvir_b = self.tensors['X'].shape[0] - nocc_b

        self.sizes['nocca'] = nocc_a
        self.sizes['noccb'] = nocc_b
        self.sizes['nvira'] = nvir_a
        self.sizes['nvirb'] = nvir_b

        if self.print_level > 0:
            print('%-19s = %11d' % ('nocca', self.sizes['nocca']))
            print('%-19s = %11d' % ('noccb', self.sizes['noccb']))
            print('%-19s = %11d' % ('nvira', self.sizes['nvira']))
            print('%-19s = %11d' % ('nvirb', self.sizes['nvirb']))
            print('')

    def compute_s2(self):

        Cocca = self.tensors['Cocca']
        Coccb = self.tensors['Coccb']
        S = self.tensors['S']

        tmp = GPUMatrixUtility.matrix_multiply(
            mat1=Cocca,
            mat2=S,
            transpose1=False,
            transpose2=False,
        )

        Sab = GPUMatrixUtility.matrix_multiply(
            mat1=tmp,
            mat2=Coccb,
            transpose1=False,
            transpose2=True,
        )

        Sab2 = GPUMatrixUtility.ddot(
            x=Sab,
            y=Sab,
            )

        nocca = self.sizes['nocca']
        noccb = self.sizes['noccb']

        Sz = 0.5 * (nocca - noccb)
        s2 = Sz * (Sz + 1.0) + noccb - Sab2

        self.scalars['Sz'] = Sz
        self.scalars['S2'] = s2

        return s2

    def compute_Enuc(self):

        xyz = self.molecule.xyz
        Z = -1.0 * self.Zeff.reshape((self.molecule.natom,))

        Enuc = 0.0
        for A in range(xyz.shape[0]):
            xyzA = xyz[A, :]
            ZA = Z[A]
            for B in range(xyz.shape[0]):
                if A == B: continue
                xyzB = xyz[B, :]
                ZB = Z[B]
                rAB = np.sqrt(np.sum((xyzA - xyzB)**2))
                Enuc += 0.5 * ZA * ZB / rAB
        return Enuc

    def compute_nuc_dipole(self):

        xyz = self.molecule.xyz
        Z = -1.0 * self.Zeff.reshape((self.molecule.natom,))

        munuc = np.zeros(3)
        for A in range(xyz.shape[0]):
            xyzA = xyz[A, :]
            ZA = Z[A]
            munuc += ZA * xyzA
        return munuc

    def compute_Enuc_gradient(self):

        xyz = self.molecule.xyz
        Z = -1.0 * self.Zeff.reshape((self.molecule.natom,))

        Gnuc = np.zeros((self.molecule.natom, 3))
        for A in range(xyz.shape[0]):
            xyzA = xyz[A, :]
            ZA = Z[A]
            for B in range(xyz.shape[0]):
                if A == B: continue
                xyzB = xyz[B, :]
                ZB = Z[B]
                rAB = np.sqrt(np.sum((xyzA - xyzB)**2))
                Gnuc[A, :] -= ZA * ZB / rAB**3 * (xyzA - xyzB)
        return GPUMatrix.from_numpy(Gnuc)

    def find_slice_count(
        self,
        *,
        slice_start,
        slice_end,
        dG_start,
        dG_end,
        dG,
        ):

        if dG >= dG_start:
            return slice_start

        if dG <= dG_end:
            return slice_end

        log_dG_start = np.log10(dG_start)
        log_dG_end = np.log10(dG_end)
        log_dG = np.log10(dG)

        t = (log_dG - log_dG_start) / (log_dG_end - log_dG_start)

        return int(round(slice_start + t * ( slice_end - slice_start)))

    def solve_diis(self):

        if self.print_level > 0:
            print('=> Solve DIIS <=\n')

        # PCM Initialization

        self.timer.start(key='PCM', stream_handle=self.cuest_handle.stream_handle)
        if self.do_pcm:
            self.pcm_q = GPUMatrix(
                nrows=self.cuest_pcm_int_plan.npoint,
                ncols=1,
                dtype=np.double,
                )
        self.timer.stop(key='PCM', stream_handle=self.cuest_handle.stream_handle)

        # External Nuclear-repulsion energy

        self.timer.start(key='Enuc', stream_handle=self.cuest_handle.stream_handle)
        Enuc = self.compute_Enuc()
        self.timer.stop(key='Enuc', stream_handle=self.cuest_handle.stream_handle)
        self.scalars['Enuc'] = Enuc

        if self.print_level > 0:
            print('%-19s = %24.16E' % ('Enuc', Enuc))
            print('')

        S = self.tensors['S']
        X = self.tensors['X']

        # Core Hamiltonian

        self.timer.start(key='T', stream_handle=self.cuest_handle.stream_handle)
        T = self.compute_kinetic()
        self.timer.stop(key='T', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='V', stream_handle=self.cuest_handle.stream_handle)
        V = self.compute_potential()
        self.timer.stop(key='V', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='Vecp', stream_handle=self.cuest_handle.stream_handle)
        if self.do_ecp:
            Vecp = self.compute_ecp_potential()
        self.timer.stop(key='Vecp', stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='Hcore', stream_handle=self.cuest_handle.stream_handle)
        H = T.clone()
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=V,
            y=H,
            )
        if self.do_ecp:
            GPUMatrixUtility.daxpy(
                alpha=1.0,
                x=Vecp,
                y=H,
                )

        self.timer.stop(key='Hcore', stream_handle=self.cuest_handle.stream_handle)

        del T, V
        if self.do_ecp:
            del Vecp

        diis_class = DIISCPU if self.cpu_diis else DIIS
        diis = diis_class(max_nvector=self.diis_max_nvector)

        if self.print_level > 1:
            print('DIIS:\n')
            print('%-19s = %11d' % ('max nvector', self.diis_max_nvector))
            print('%-19s = %11s' % ('CPU', self.cpu_diis))
            print('')

        if self.print_level > 1:
            print('Convergence Options:\n')
            print('%-19s = %11d' % ('max iterations', self.maxiter))
            print('%-19s = %11.3E' % ('g convergence', self.g_convergence))
            print('%-19s = %11.3f' % ('level shift a', self.level_shift_a))
            print('%-19s = %11.3f' % ('level shift b', self.level_shift_b))
            print('')

        # Main SCF Iterative Loop

        start = time.time()
        self.is_converged = False
        Eold = 0.0
        E = 0.0
        dG = 1.0
        U2a_prev = None
        U2b_prev = None
        if self.level_shift_a != 0.0 or self.level_shift_b != 0.0:
            level_shift_ones = GPUMatrix.from_numpy(np.ones((X.shape[0], 1)))
        dG_best = float('inf')
        dG_no_improve = 0
        skip_diis = False
        # Incremental-Fock state: densities and J/K of the previous build
        incr_D_prev = None
        incr_n_delta = 0   # delta builds since the last full build
        incr_Da_prev = None
        incr_Db_prev = None
        incr_J = None
        incr_Ka = None
        incr_Kb = None
        if self.print_level:
            print('%4s: %24s %11s %11s %8s' % ('Iter', 'Energy', 'dE', 'dG', 'Time[s]'))
        for iteration in range(self.maxiter):

            Da = self.tensors['Da']
            Db = self.tensors['Db']
            Cocca = self.tensors['Cocca']
            Coccb = self.tensors['Coccb']

            D = Da.clone()
            GPUMatrixUtility.daxpy(
                alpha=1.0,
                x=Db,
                y=D,
                )

            # Find slice count (DF-K int8 tuning; only used for fock_method "dfjk")
            dfk_int8_slice_count = None
            dfk_int8_modulus_count = None
            if self.fock_method == "dfjk":
                dfk_int8_slice_count = self.find_slice_count(
                    slice_start=self.dfk_int8_slice_count_start,
                    slice_end=self.dfk_int8_slice_count_end,
                    dG_start=1.0,
                    dG_end=self.g_convergence,
                    dG=dG,
                    )
                dfk_int8_modulus_count = self.find_slice_count(
                    slice_start=self.dfk_int8_modulus_count_start,
                    slice_end=self.dfk_int8_modulus_count_end,
                    dG_start=1.0,
                    dG_end=self.g_convergence,
                    dG=dG,
                    )

            # Fock Matrix

            # Incremental directjk build: contract the delta densities and
            # accumulate into the previous J/Ka/Kb.
            D_contract = D
            Da_contract = Da
            Db_contract = Db
            # This build is the (incr_n_delta + 1)-th since the last full one:
            # a delta while that count stays below the rebuild period.
            incr_delta = (incr_D_prev is not None
                          and (self.incremental_fock_rebuild == 0
                               or incr_n_delta + 1 < self.incremental_fock_rebuild))
            if incr_delta:
                D_contract = GPUMatrixUtility.dgeam(
                    transpose1=False,
                    alpha=1.0,
                    mat1=D,
                    transpose2=False,
                    beta=-1.0,
                    mat2=incr_D_prev,
                    )
                Da_contract = GPUMatrixUtility.dgeam(
                    transpose1=False,
                    alpha=1.0,
                    mat1=Da,
                    transpose2=False,
                    beta=-1.0,
                    mat2=incr_Da_prev,
                    )
                Db_contract = GPUMatrixUtility.dgeam(
                    transpose1=False,
                    alpha=1.0,
                    mat1=Db,
                    transpose2=False,
                    beta=-1.0,
                    mat2=incr_Db_prev,
                    )
            elif incr_D_prev is not None:
                incr_D_prev = incr_Da_prev = incr_Db_prev = None
                incr_J = incr_Ka = incr_Kb = None

            self.timer.start(key='J', stream_handle=self.cuest_handle.stream_handle)
            J = self.compute_coulomb(D=D_contract)
            if incr_delta:
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=incr_J,
                    y=J,
                    )
            self.timer.stop(key='J', stream_handle=self.cuest_handle.stream_handle)

            self.timer.start(key='K', stream_handle=self.cuest_handle.stream_handle)
            if self.gridk_sad_reference_data is not None:
                (Ka, Kb), (dDa, dDb), (Kgrid_dDa, Kgrid_dDb) = self.compute_gridk_sad_reference_exchange(
                    Ds=[Da, Db],
                    )
            elif self.do_incremental_fock:
                # Both spin exchange builds in one batched multi-density call.
                dDa = dDb = None
                Kgrid_dDa = Kgrid_dDb = None
                Ka, Kb = self.compute_exchange_density(Ds=[Da_contract, Db_contract])
                if incr_delta:
                    GPUMatrixUtility.daxpy(
                        alpha=1.0,
                        x=incr_Ka,
                        y=Ka,
                        )
                    GPUMatrixUtility.daxpy(
                        alpha=1.0,
                        x=incr_Kb,
                        y=Kb,
                        )
            else:
                dDa = dDb = None
                Kgrid_dDa = Kgrid_dDb = None
                Ka, Kb = self.compute_exchange(
                    Coccs=[Cocca, Coccb],
                    dfk_int8_slice_count=dfk_int8_slice_count,
                    dfk_int8_modulus_count=dfk_int8_modulus_count,
                    )
            self.timer.stop(key='K', stream_handle=self.cuest_handle.stream_handle)

            if self.do_incremental_fock:
                incr_n_delta = incr_n_delta + 1 if incr_delta else 0
                # Da/Db/D are not mutated later in the iteration and
                # tensors['Da']/['Db'] are rebound, not overwritten, so
                # these references stay the as-built quantities.
                incr_D_prev = D
                incr_Da_prev = Da
                incr_Db_prev = Db
                incr_J = J
                incr_Ka = Ka
                incr_Kb = Kb

            del D_contract, Da_contract, Db_contract

            self.timer.start(key='Vxc', stream_handle=self.cuest_handle.stream_handle)
            Exc, Vxca, Vxcb = self.compute_local_exchange_correlation(Cocca=Cocca,Coccb=Coccb)
            self.timer.stop(key='Vxc', stream_handle=self.cuest_handle.stream_handle)

            self.timer.start(key='VV10', stream_handle=self.cuest_handle.stream_handle)
            Enlc, Vnlc = self.compute_nonlocal_exchange_correlation(Cocca=Cocca,Coccb=Coccb)
            self.timer.stop(key='VV10', stream_handle=self.cuest_handle.stream_handle)

            # NOTE: This is typically where other electron-dependent Fock
            # matrix terms are added, e.g., PCM, wK.

            self.timer.start(key='F', stream_handle=self.cuest_handle.stream_handle)
            Fa = H.clone()
            GPUMatrixUtility.daxpy(
                alpha=1.0,
                x=J,
                y=Fa,
                )
            GPUMatrixUtility.daxpy(
                alpha=-1.0,
                x=Ka,
                y=Fa,
                )

            Fb = H.clone()
            GPUMatrixUtility.daxpy(
                alpha=1.0,
                x=J,
                y=Fb,
                )
            GPUMatrixUtility.daxpy(
                alpha=-1.0,
                x=Kb,
                y=Fb,
                )

            self.timer.stop(key='F', stream_handle=self.cuest_handle.stream_handle)

            # SCF Energy

            self.timer.start(key='E', stream_handle=self.cuest_handle.stream_handle)
            E = 0.0
            E += Enuc
            E += 1.0 * GPUMatrixUtility.ddot(
                x=D,
                y=H,
                )
            E += 0.5 * GPUMatrixUtility.ddot(
                x=D,
                y=J,
                )
            # Exchange energy: from the SAD-reference expansion,
            # E_K = -1/2 sum_s [D_ref . K_ref + 2 dD . K_ref + dD . K_gridk(dD)],
            # when active, otherwise E_K = -1/2 sum_s D_s . K_s
            if self.gridk_sad_reference_data is not None:
                self.scalars['E_gridk_sad_reference'] = self.compute_gridk_sad_reference_energy(
                    dDs=[dDa, dDb],
                    Kgrid_dDs=[Kgrid_dDa, Kgrid_dDb],
                    )
                E += self.scalars['E_gridk_sad_reference']
            else:
                self.scalars['E_gridk_sad_reference'] = 0.0
                E += -0.5 * GPUMatrixUtility.ddot(
                    x=Da,
                    y=Ka,
                    )
                E += -0.5 * GPUMatrixUtility.ddot(
                    x=Db,
                    y=Kb,
                    )
            self.timer.stop(key='E', stream_handle=self.cuest_handle.stream_handle)

            del J, Ka, Kb, dDa, dDb, Kgrid_dDa, Kgrid_dDb

            # Include PCM
            pcm_converged = True
            self.timer.start(key='PCM', stream_handle=self.cuest_handle.stream_handle)
            if self.do_pcm:
                Epcm, Vpcm, q_out, pcm_converged = self.compute_pcm_energy_and_potential(
                    q=self.pcm_q,
                    D=D)

                self.pcm_q = q_out

                E += Epcm
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vpcm,
                    y=Fa,
                    )
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vpcm,
                    y=Fb,
                    )
                del Vpcm
                if not pcm_converged and self.print_level:
                    print('Warning: PCM solver did not converge')
            self.timer.stop(key='PCM', stream_handle=self.cuest_handle.stream_handle)

            del D

            E += Exc
            E += Enlc
            self.scalars['Escf'] = E

            # Add Vxc to the Fock matrix after calculating the energy

            self.timer.start(key='F', stream_handle=self.cuest_handle.stream_handle)
            if Vxca is not None:
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vxca,
                    y=Fa,
                    )
            if Vxcb is not None:
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vxcb,
                    y=Fb,
                    )
            if Vnlc is not None:
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vnlc,
                    y=Fa,
                    )
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vnlc,
                    y=Fb,
                    )

            self.timer.stop(key='F', stream_handle=self.cuest_handle.stream_handle)

            del Vxca, Vxcb, Vnlc

            dE = E - Eold
            Eold = E

            # Orbital Gradient (zero at convergence)
            # G = XSDFX' - XFDSX'

            # G = np.dot(np.dot(np.dot(np.dot(X, S), D), F), X.T)
            # Faster in large basis sets

            self.timer.start(key='orbital_gradient', stream_handle=self.cuest_handle.stream_handle)

            # alpha
            L1 = GPUMatrixUtility.matrix_multiply(
                mat1=Cocca,
                mat2=S,
                transpose1=False,
                transpose2=False,
                )
            L = GPUMatrixUtility.matrix_multiply(
                mat1=L1,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )
            del L1
            R1 = GPUMatrixUtility.matrix_multiply(
                mat1=Cocca,
                mat2=Fa,
                transpose1=False,
                transpose2=False,
                )
            R = GPUMatrixUtility.matrix_multiply(
                mat1=R1,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )
            del R1

            Ga = GPUMatrixUtility.matrix_multiply(
                mat1=L,
                mat2=R,
                transpose1=True,
                transpose2=False,
                )
            del L, R
            Ga = GPUMatrixUtility.dgeam(
                mat1=Ga,
                alpha=1.0,
                transpose1=False,
                mat2=Ga,
                beta=-1.0,
                transpose2=True,
                )

            # beta
            L1 = GPUMatrixUtility.matrix_multiply(
                mat1=Coccb,
                mat2=S,
                transpose1=False,
                transpose2=False,
                )
            L = GPUMatrixUtility.matrix_multiply(
                mat1=L1,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )
            del L1
            R1 = GPUMatrixUtility.matrix_multiply(
                mat1=Coccb,
                mat2=Fb,
                transpose1=False,
                transpose2=False,
                )
            R = GPUMatrixUtility.matrix_multiply(
                mat1=R1,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )
            del R1

            Gb = GPUMatrixUtility.matrix_multiply(
                mat1=L,
                mat2=R,
                transpose1=True,
                transpose2=False,
                )
            del L, R
            Gb = GPUMatrixUtility.dgeam(
                mat1=Gb,
                alpha=1.0,
                transpose1=False,
                mat2=Gb,
                beta=-1.0,
                transpose2=True,
                )

            dGa = GPUMatrixUtility.damax(x=Ga)
            dGb = GPUMatrixUtility.damax(x=Gb)
            dG = max(dGa, dGb)

            self.timer.stop(key='orbital_gradient', stream_handle=self.cuest_handle.stream_handle)

            # Convergence Trace

            stop = time.time()
            if self.print_level:
                if self.fock_method == "dfjk":
                    print('%4d: %24.16E %11.3E %11.3E %8.3f (%2d moduli, %2d slices)' % (iteration, E, dE, dG, stop-start, dfk_int8_modulus_count, dfk_int8_slice_count))
                else:
                    print('%4d: %24.16E %11.3E %11.3E %8.3f' % (iteration, E, dE, dG, stop-start))
            start = stop

            # Convergence Check

            if iteration > 0 and dG < self.g_convergence and pcm_converged:
                self.is_converged = True
                break

            # This ensures that we do not extrapolate/diagonalize if maxiter is reached
            if iteration >= self.maxiter - 1:
                break

            # reset history if dG hasn't improved for 10 iterations
            if dG < dG_best * 0.95:
                dG_best = dG
                dG_no_improve = 0
                skip_diis = False
            else:
                dG_no_improve += 1
                if dG_no_improve >= 10:
                    diis = diis_class(max_nvector=self.diis_max_nvector)
                    dG_best = dG
                    dG_no_improve = 0
                    skip_diis = True

            # DIIS Extrapolation (also updates DIIS history)

            self.timer.start(key='DIIS', stream_handle=self.cuest_handle.stream_handle)
            if iteration > 0 and not skip_diis:
                Fa, Fb = diis.iterate(
                    state_vectors=[Fa, Fb],
                    error_vectors=[Ga, Gb],
                    )
            skip_diis = False

            del Ga, Gb

            self.timer.stop(key='DIIS', stream_handle=self.cuest_handle.stream_handle)

            # Fock Matrix Diagonalization

            self.timer.start(key='orbital_update', stream_handle=self.cuest_handle.stream_handle)

            tmp = GPUMatrixUtility.matrix_multiply(
                mat1=X,
                mat2=Fa,
                transpose1=False,
                transpose2=False,
                )
            F2a = GPUMatrixUtility.matrix_multiply(
                mat1=tmp,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )
            del tmp, Fa

            if self.level_shift_a != 0.0 and U2a_prev is not None:
                # F2a += level_shift_a * (1 - P_occ) with P_occ built from the
                # previous occupied eigenvectors, without leaving the GPU
                GPUMatrixUtility.daxpy(
                    n=F2a.nrows,
                    alpha=self.level_shift_a,
                    x=level_shift_ones,
                    offx=0,
                    incx=1,
                    y=F2a,
                    offy=0,
                    incy=F2a.nrows + 1,
                    )
                GPUMatrixUtility.c_dgemm(
                    transa=False,
                    transb=True,
                    m=F2a.nrows,
                    n=F2a.nrows,
                    k=self.sizes['nocca'],
                    alpha=-self.level_shift_a,
                    a=U2a_prev,
                    offa=0,
                    lda=F2a.nrows,
                    b=U2a_prev,
                    offb=0,
                    ldb=F2a.nrows,
                    beta=1.0,
                    c=F2a,
                    offc=0,
                    ldc=F2a.nrows,
                    )

            self.timer.start(key='diagonalize', stream_handle=self.cuest_handle.stream_handle)
            epsa, U2a = GPUMatrixUtility.eigh(matrix=F2a)
            self.timer.stop(key='diagonalize', stream_handle=self.cuest_handle.stream_handle)
            del F2a

            if self.level_shift_a != 0.0:
                U2a_prev = U2a

            Ca = GPUMatrixUtility.matrix_multiply(
                mat1=U2a,
                mat2=X,
                transpose1=True,
                transpose2=False,
                )
            del U2a

            # New Occupied Orbitals and Density Matrix

            Cocca = GPUMatrix(
                nrows=self.sizes['nocca'],
                ncols=Ca.shape[1],
                dtype=np.double,
                )
            GPUMatrixUtility.dcopy(
                n=self.sizes['nocca'] * Ca.shape[1],
                x=Ca,
                offx=0,
                incx=1,
                y=Cocca,
                offy=0,
                incy=1,
                )

            Da = GPUMatrixUtility.matrix_multiply(
                mat1=Cocca,
                mat2=Cocca,
                transpose1=True,
                transpose2=False,
                )

            tmp = GPUMatrixUtility.matrix_multiply(
                mat1=X,
                mat2=Fb,
                transpose1=False,
                transpose2=False,
                )
            F2b = GPUMatrixUtility.matrix_multiply(
                mat1=tmp,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )
            del tmp, Fb

            if self.level_shift_b != 0.0 and U2b_prev is not None:
                # F2b += level_shift_b * (1 - P_occ) with P_occ built from the
                # previous occupied eigenvectors, without leaving the GPU
                GPUMatrixUtility.daxpy(
                    n=F2b.nrows,
                    alpha=self.level_shift_b,
                    x=level_shift_ones,
                    offx=0,
                    incx=1,
                    y=F2b,
                    offy=0,
                    incy=F2b.nrows + 1,
                    )
                GPUMatrixUtility.c_dgemm(
                    transa=False,
                    transb=True,
                    m=F2b.nrows,
                    n=F2b.nrows,
                    k=self.sizes['noccb'],
                    alpha=-self.level_shift_b,
                    a=U2b_prev,
                    offa=0,
                    lda=F2b.nrows,
                    b=U2b_prev,
                    offb=0,
                    ldb=F2b.nrows,
                    beta=1.0,
                    c=F2b,
                    offc=0,
                    ldc=F2b.nrows,
                    )

            self.timer.start(key='diagonalize', stream_handle=self.cuest_handle.stream_handle)
            epsb, U2b = GPUMatrixUtility.eigh(matrix=F2b)
            self.timer.stop(key='diagonalize', stream_handle=self.cuest_handle.stream_handle)
            del F2b

            if self.level_shift_b != 0.0:
                U2b_prev = U2b

            Cb = GPUMatrixUtility.matrix_multiply(
                mat1=U2b,
                mat2=X,
                transpose1=True,
                transpose2=False,
                )
            del U2b

            # New Occupied Orbitals and Density Matrix

            Coccb = GPUMatrix(
                nrows=self.sizes['noccb'],
                ncols=Cb.shape[1],
                dtype=np.double,
                )
            GPUMatrixUtility.dcopy(
                n=self.sizes['noccb'] * Cb.shape[1],
                x=Cb,
                offx=0,
                incx=1,
                y=Coccb,
                offy=0,
                incy=1,
                )

            Db = GPUMatrixUtility.matrix_multiply(
                mat1=Coccb,
                mat2=Coccb,
                transpose1=True,
                transpose2=False,
                )

            self.tensors['Da'] = Da
            self.tensors['Cocca'] = Cocca

            self.tensors['Ca'] = Ca
            self.tensors['epsa'] = epsa

            self.tensors['Db'] = Db
            self.tensors['Coccb'] = Coccb

            self.tensors['Cb'] = Cb
            self.tensors['epsb'] = epsb


            self.timer.stop(key='orbital_update', stream_handle=self.cuest_handle.stream_handle)

        # Remove the level shift from the converged orbitals and orbital
        # energies with a final unshifted diagonalization
        if self.is_converged and (self.level_shift_a != 0.0 or self.level_shift_b != 0.0):
            tmp = GPUMatrixUtility.matrix_multiply(
                mat1=X,
                mat2=Fa,
                transpose1=False,
                transpose2=False,
                )
            F2a = GPUMatrixUtility.matrix_multiply(
                mat1=tmp,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )

            self.timer.start(key='diagonalize', stream_handle=self.cuest_handle.stream_handle)
            epsa, U2a = GPUMatrixUtility.eigh(matrix=F2a)
            self.timer.stop(key='diagonalize', stream_handle=self.cuest_handle.stream_handle)

            Ca = GPUMatrixUtility.matrix_multiply(
                mat1=U2a,
                mat2=X,
                transpose1=True,
                transpose2=False,
                )

            Cocca = GPUMatrix(
                nrows=self.sizes['nocca'],
                ncols=Ca.shape[1],
                dtype=np.double,
                )
            GPUMatrixUtility.dcopy(
                n=self.sizes['nocca'] * Ca.shape[1],
                x=Ca,
                offx=0,
                incx=1,
                y=Cocca,
                offy=0,
                incy=1,
                )

            Da = GPUMatrixUtility.matrix_multiply(
                mat1=Cocca,
                mat2=Cocca,
                transpose1=True,
                transpose2=False,
                )

            tmp = GPUMatrixUtility.matrix_multiply(
                mat1=X,
                mat2=Fb,
                transpose1=False,
                transpose2=False,
                )
            F2b = GPUMatrixUtility.matrix_multiply(
                mat1=tmp,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )

            self.timer.start(key='diagonalize', stream_handle=self.cuest_handle.stream_handle)
            epsb, U2b = GPUMatrixUtility.eigh(matrix=F2b)
            self.timer.stop(key='diagonalize', stream_handle=self.cuest_handle.stream_handle)

            Cb = GPUMatrixUtility.matrix_multiply(
                mat1=U2b,
                mat2=X,
                transpose1=True,
                transpose2=False,
                )

            Coccb = GPUMatrix(
                nrows=self.sizes['noccb'],
                ncols=Cb.shape[1],
                dtype=np.double,
                )
            GPUMatrixUtility.dcopy(
                n=self.sizes['noccb'] * Cb.shape[1],
                x=Cb,
                offx=0,
                incx=1,
                y=Coccb,
                offy=0,
                incy=1,
                )

            Db = GPUMatrixUtility.matrix_multiply(
                mat1=Coccb,
                mat2=Coccb,
                transpose1=True,
                transpose2=False,
                )

            self.tensors['Da'] = Da
            self.tensors['Cocca'] = Cocca

            self.tensors['Ca'] = Ca
            self.tensors['epsa'] = epsa

            self.tensors['Db'] = Db
            self.tensors['Coccb'] = Coccb

            self.tensors['Cb'] = Cb
            self.tensors['epsb'] = epsb

        # => Print Convergence <= #

        if self.print_level > 0:
            print('')
            if self.is_converged:
                self.timer.start(key='S2', stream_handle=self.cuest_handle.stream_handle)
                self.compute_s2()
                self.timer.stop(key='S2', stream_handle=self.cuest_handle.stream_handle)
                print('SCF Converged\n')
                S = 0.5 * (self.multiplicity - 1)
                S2_exact = S * (S + 1.0)
                print('%-19s = %11d' % ('multiplicity', self.multiplicity))
                print('%-19s = %11.3E' % ('<Sz>', self.scalars['Sz']))
                print('%-19s = %11.3E' % ('<S2>', self.scalars['S2']))
                print('%-19s = %11.3E' % ('Spin contamination', self.scalars['S2'] - S2_exact))
                print('')
            else:
                print('SCF Failed\n')

        # => Print Final Energy <= #

        if self.print_level:
            print('%-19s = %24.16E\n' % ('SCF Energy', E))

        if self.print_level > 0:
            print('=> End Solve DIIS <=\n')

    def compute_energy(self):

        if not self.is_solved:
            self.solve()

        if not self.is_converged:
            raise RuntimeError('Not converged')

        return self.scalars['Escf']

    def compute_properties(self):

        if not self.is_solved:
            self.solve()

        if not self.is_converged:
            raise RuntimeError('Not converged')

        self.timer.start(key='properties', stream_handle=self.cuest_handle.stream_handle, always=True)

        properties = {}

        Da = self.tensors['Da']
        Db = self.tensors['Db']

        D = Da.clone()
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=Db,
            y=D,
            )

        # Dipole moment
        dipole = self.compute_nuc_dipole()
        mu = self.compute_dipole()

        for i in range(3):
            dipole[i] -= 1.0 * GPUMatrixUtility.ddot(
                x=D,
                y=mu[i],
                )

        properties['mu_x'] = dipole[0]
        properties['mu_y'] = dipole[1]
        properties['mu_z'] = dipole[2]

        self.timer.stop(key='properties', stream_handle=self.cuest_handle.stream_handle, always=True)

        if self.print_level:
            print('%-19s = %14.4E %14.4E %14.4E\n' % ('SCF Dipole', dipole[0], dipole[1], dipole[2],))
            print('cuEST UHF Properties Time: %11.3E [s]\n' % (self.timer.times['properties']))

        return properties

    def compute_gradient(self):

        if not self.is_solved:
            self.solve()

        if not self.is_converged:
            raise RuntimeError('Not converged')

        self.timer.start(key='gradient', stream_handle=self.cuest_handle.stream_handle, always=True)

        # Occupied orbitals / density matrix

        Cocca = self.tensors['Cocca']
        Da = self.tensors['Da']
        Coccb = self.tensors['Coccb']
        Db = self.tensors['Db']

        D = Da.clone()
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=Db,
            y=D,
            )

        # Energy-weighted density matrix

        # W = Cocc^T diag(eps_occ) Cocc, formed on the GPU
        self.timer.start(key='W', stream_handle=self.cuest_handle.stream_handle)
        epsa_occ = GPUMatrix(
            nrows=self.sizes['nocca'],
            ncols=self.sizes['nocca'],
            dtype=np.double,
            )
        GPUMatrixUtility.dcopy(
            n=self.sizes['nocca'],
            x=self.tensors['epsa'],
            offx=0,
            incx=1,
            y=epsa_occ,
            offy=0,
            incy=self.sizes['nocca'] + 1,
            )
        Va = GPUMatrixUtility.matrix_multiply(
            mat1=epsa_occ,
            mat2=Cocca,
            transpose1=False,
            transpose2=False,
            )
        Wa = GPUMatrixUtility.matrix_multiply(
            mat1=Va,
            mat2=Cocca,
            transpose1=True,
            transpose2=False,
            )

        epsb_occ = GPUMatrix(
            nrows=self.sizes['noccb'],
            ncols=self.sizes['noccb'],
            dtype=np.double,
            )
        GPUMatrixUtility.dcopy(
            n=self.sizes['noccb'],
            x=self.tensors['epsb'],
            offx=0,
            incx=1,
            y=epsb_occ,
            offy=0,
            incy=self.sizes['noccb'] + 1,
            )
        Vb = GPUMatrixUtility.matrix_multiply(
            mat1=epsb_occ,
            mat2=Coccb,
            transpose1=False,
            transpose2=False,
            )
        Wb = GPUMatrixUtility.matrix_multiply(
            mat1=Vb,
            mat2=Coccb,
            transpose1=True,
            transpose2=False,
            )

        # Total energy-weighted density
        W = Wa.clone()
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=Wb,
            y=W,
            )
        del epsa_occ, Va, Wa, epsb_occ, Vb, Wb

        self.timer.stop(key='W', stream_handle=self.cuest_handle.stream_handle)

        # Nuclear gradient (classical)

        self.timer.start(key='dEnuc', stream_handle=self.cuest_handle.stream_handle)
        Gnuc = self.compute_Enuc_gradient()
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=Gnuc,
            )

        self.timer.stop(key='dEnuc', stream_handle=self.cuest_handle.stream_handle)

        # Overlap gradient

        self.timer.start(key='dS', stream_handle=self.cuest_handle.stream_handle)
        GS = self.compute_overlap_gradient(W)
        self.timer.stop(key='dS', stream_handle=self.cuest_handle.stream_handle)
        GPUMatrixUtility.dscal(
            alpha=-1.0,
            x=GS,
            )
        del W

        # Kinetic gradient

        self.timer.start(key='dT', stream_handle=self.cuest_handle.stream_handle)
        GT = self.compute_kinetic_gradient(D)
        self.timer.stop(key='dT', stream_handle=self.cuest_handle.stream_handle)
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=GT,
            )

        # Potential gradient

        self.timer.start(key='dV', stream_handle=self.cuest_handle.stream_handle)
        GV = self.compute_potential_gradient(D)
        self.timer.stop(key='dV', stream_handle=self.cuest_handle.stream_handle)
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=GV,
            )

        # ECP gradient

        self.timer.start(key='dVecp', stream_handle=self.cuest_handle.stream_handle)
        Gecp = None
        if self.do_ecp:
            Gecp = self.compute_ecp_gradient(D)
            GPUMatrixUtility.dscal(
                alpha=+1.0,
                x=Gecp,
                )

        self.timer.stop(key='dVecp', stream_handle=self.cuest_handle.stream_handle)

        # United Coulomb + Exchange gradient

        self.timer.start(key='dJK', stream_handle=self.cuest_handle.stream_handle)
        GJK = self.compute_coulomb_and_exchange_gradient(
            scaleJ=+0.5,
            DJ=D,
            scaleK=-0.5,
            CoccsK=[Cocca,Coccb],
            )
        self.timer.stop(key='dJK', stream_handle=self.cuest_handle.stream_handle)

        if not self.do_pcm:
            del D

        # Local exchange-correlation gradient

        self.timer.start(key='dVxc', stream_handle=self.cuest_handle.stream_handle)
        GVxc = self.compute_local_exchange_correlation_gradient(Cocca=Cocca,Coccb=Coccb)
        self.timer.stop(key='dVxc', stream_handle=self.cuest_handle.stream_handle)
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=GVxc,
            )

        # Nonlocal exchange-correlation gradient

        self.timer.start(key='dVV10', stream_handle=self.cuest_handle.stream_handle)
        GVnlc = self.compute_nonlocal_exchange_correlation_gradient(Cocca=Cocca,Coccb=Coccb)
        self.timer.stop(key='dVV10', stream_handle=self.cuest_handle.stream_handle)
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=GVnlc,
            )

        # PCM gradient

        Gpcm = None
        self.timer.start(key='dPCM', stream_handle=self.cuest_handle.stream_handle)
        if self.do_pcm:
            Gpcm = self.compute_pcm_gradient(
                q=self.pcm_q,
                D=D)
        self.timer.stop(key='dPCM', stream_handle=self.cuest_handle.stream_handle)

        G = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            initialize=True,
            )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=Gnuc,
            y=G,
            )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=GS,
            y=G,
            )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=GT,
            y=G,
            )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=GV,
            y=G,
            )
        if self.do_ecp:
            GPUMatrixUtility.daxpy(
                alpha=1.0,
                x=Gecp,
                y=G,
                )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=GJK,
            y=G,
            )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=GVxc,
            y=G,
            )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=GVnlc,
            y=G,
            )
        if self.do_pcm:
            GPUMatrixUtility.daxpy(
                alpha=1.0,
                x=Gpcm,
                y=G,
                )

        self.timer.stop(key='gradient', stream_handle=self.cuest_handle.stream_handle, always=True)

        if self.print_level:
            print('cuEST UHF Gradient Time: %11.3E [s]\n' % (self.timer.times['gradient']))

        return G

