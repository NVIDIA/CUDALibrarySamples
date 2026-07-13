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
from .diis import DIIS
from .xc_functionals import XCFunctionalInfo

from .cuda_utility import CudaUtility

from .cuest_handle import CuestHandle
from .cuest_ao_basis import CuestAOBasis
from .cuest_molecular_grid import CuestMolecularGrid
from .cuest_ao_pair_list import CuestAOPairList
from .cuest_oe_int_plan import CuestOEIntPlan
from .cuest_oe_int_compute import CuestOEIntCompute
from .cuest_df_int_plan import CuestDFIntPlan
from .cuest_df_int_compute import CuestDFIntCompute
from .cuest_xc_int_plan import CuestXCIntPlan
from .cuest_xc_int_compute import CuestXCIntCompute
from .cuest_pcm_int_plan import CuestPCMIntPlan
from .cuest_pcm_int_compute import CuestPCMIntCompute
from .cuest_ecp_basis import CuestECPBasis
from .cuest_ecp_int_plan import CuestECPIntPlan
from .cuest_ecp_int_compute import CuestECPIntCompute

from .timer import Timer

from .gpu_matrix import GPUMatrix
from .gpu_matrix_utility import GPUMatrixUtility

import time

class UHF(object):

    def __init__(
        self,
        *,
        cuest_handle : CuestHandle,
        molecule : Molecule,
        charge : int,
        multiplicity: int,
        xc_functional_name : str = None,
        primary : AOBasis,
        auxiliary : AOBasis,
        minao : AOBasis,
        ecp_basis : ECPBasis = None,
        primary_name : str = None,
        auxiliary_name : str = None,
        minao_name : str = None,
        print_level : int = 2,
        print_timings : bool = True,
        threshold_pq : float = 1.0E-12,
        threshold_canonical : float = 1.0E-6,
        df_fitting_eigenvalue_cutoff : float = 1.0e-12,
        diis_max_nvector : int = 6,
        diis_flush_niter : int = 40,
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
        pcm_epsilon: float = 1.0, # 1.0 for gas phase, >1.0 for solvated calc.
        pcm_x_prefactor: float = 0.0, # 0.0 for CPCM, 0.5 for COSMO
        pcm_cutoff: float = 1.0e-10,
        pcm_convergence_tol: float = 1e-10,
        pcm_num_angular_points_per_hydrogen_atom: int = 110,
        pcm_num_angular_points_per_heavy_atom: int = 194,
        benchmark: bool = False,
        ): 

        self.uhf = True 
        self.benchmark=benchmark

        self.timer = Timer()

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
        self.diis_max_nvector = diis_max_nvector
        self.diis_flush_niter = diis_flush_niter
        self.maxiter = maxiter
        self.g_convergence = g_convergence
        self.df_fitting_eigenvalue_cutoff = df_fitting_eigenvalue_cutoff

        self.pcm_num_angular_points_per_heavy_atom = pcm_num_angular_points_per_heavy_atom
        self.pcm_num_angular_points_per_hydrogen_atom = pcm_num_angular_points_per_hydrogen_atom
        self.pcm_epsilon = pcm_epsilon
        self.pcm_x_prefactor = pcm_x_prefactor
        self.pcm_cutoff = pcm_cutoff
        self.pcm_convergence_tol = pcm_convergence_tol
        self.pcm_q = None

        # pcm_epsilon == 1.0 is the gas-phase default (PCM off); only sub-vacuum
        # dielectrics are unphysical and rejected.
        if self.pcm_epsilon < 1.0:
            raise RuntimeError('pcm_epsilon < 1.0')

        self.dfk_int8_slice_count_start = dfk_int8_slice_count_start
        self.dfk_int8_modulus_count_start = dfk_int8_modulus_count_start
        self.dfk_int8_slice_count_end = dfk_int8_slice_count_end
        self.dfk_int8_modulus_count_end = dfk_int8_modulus_count_end

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

        self.is_solved = False
        self.is_converged = False

        self.gpu_xyz = GPUMatrix.from_numpy(self.molecule.xyz)
        self.Zeff = -1.0 * np.array(self.molecule.Z.reshape(-1,1))
        if self.do_ecp:
            for idx, atom in enumerate(self.ecp_basis.atoms):
                self.Zeff[idx] = self.Zeff[idx] + atom.nelectron
        self.gpu_Z = GPUMatrix.from_numpy(self.Zeff)

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
        return CuestDFIntPlan(
            handle=self.cuest_handle,
            primary=self.cuest_primary,
            auxiliary=self.cuest_auxiliary,
            ao_pair_list=self.cuest_ao_pair_list,
            exchange_scale=self.cuest_xc_int_plan.exchange_scale,
            lrc_exchange_scale=self.cuest_xc_int_plan.lrc_exchange_scale,
            lrc_omega=self.cuest_xc_int_plan.lrc_omega,
            df_fitting_eigenvalue_cutoff=self.df_fitting_eigenvalue_cutoff,
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
            ) 

    @memoized_property
    def cuest_pcm_int_plan(self):
        if self.pcm_epsilon<=1.0:
            return None
        else:
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
                ) 

    @property
    def do_pcm(self):
        return self.cuest_pcm_int_plan is not None


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

    def compute_ecp_potential(self):

        Vecp = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        CuestECPIntCompute.compute_ecp_potential(
            handle=self.cuest_handle,
            ecp_int_plan=self.cuest_ecp_int_plan,
            Vptr=Vecp.pointer,
            )

        return Vecp

    def compute_ecp_gradient(
        self,
        D,
        ):

        Gecp = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        CuestECPIntCompute.compute_ecp_gradient(
            handle=self.cuest_handle,
            ecp_int_plan=self.cuest_ecp_int_plan,
            Dptr=D.pointer,
            Gptr=Gecp.pointer,
            )

        return Gecp

    def compute_coulomb(
        self,
        D,
        ):

        J = D.clone()

        CuestDFIntCompute.coulomb(
            handle=self.cuest_handle,
            df_int_plan=self.cuest_df_int_plan,
            Jptr=J.pointer,
            Dptr=D.pointer,
            )

        return J
    
    def compute_exchange(
        self,   
        Cocc,
        dfk_int8_slice_count,
        dfk_int8_modulus_count,
        ):

        K = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        if self.cuest_df_int_plan.exchange_scale == 0.0 and self.cuest_df_int_plan.lrc_exchange_scale == 0.0:
            K.zero()
            return K

        CuestDFIntCompute.symmetric_exchange(
            handle=self.cuest_handle,
            df_int_plan=self.cuest_df_int_plan,
            Kptr=K.pointer,
            nocc=Cocc.shape[0],
            Coccptr=Cocc.pointer,
            dfk_int8_slice_count=dfk_int8_slice_count,
            dfk_int8_modulus_count=dfk_int8_modulus_count,
            )

        return K

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

        Exc = CuestXCIntCompute.local_exchange_correlation(
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

        Exc = CuestXCIntCompute.nonlocal_exchange_correlation(
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

        Epcm, converged, residual = CuestPCMIntCompute.compute_pcm_energy_and_potential(
            handle=self.cuest_handle,
            pcm_int_plan=self.cuest_pcm_int_plan,
            Dptr=D.pointer, 
            inQptr=q.pointer,
            outQptr=outQ.pointer,
            Vptr=V.pointer,
            )

        return Epcm, V, outQ

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

        CuestPCMIntCompute.compute_pcm_gradient(
            handle=self.cuest_handle,
            pcm_int_plan=self.cuest_pcm_int_plan,
            Dptr=D.pointer, 
            inQptr=q.pointer,
            outQptr=outQ.pointer,
            Gptr=G.pointer,
            )

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

    def compute_coulomb_and_exchange_gradient(
        self,   
        *,
        scaleJ,
        DJ,
        scaleK,
        CoccsK,
        ):

        noccsK = [_.shape[0] for _ in CoccsK]

        Cocca_np = CoccsK[0].to_numpy()
        Coccb_np = CoccsK[1].to_numpy()
 
        Cocc_ab_np = np.vstack([Cocca_np, Coccb_np]) 
        Cocc_ab = GPUMatrix.from_numpy(Cocc_ab_np)

        G = GPUMatrix(
            nrows=self.primary.natom,
            ncols=3,
            dtype=np.double,
            )

        CuestDFIntCompute.coulomb_and_exchange_gradient(
            handle=self.cuest_handle,
            df_int_plan=self.cuest_df_int_plan,
            scaleJ=scaleJ,
            DJptr=DJ.pointer,
            scaleK=scaleK, 
            noccsK=noccsK,
            CoccsKptr=Cocc_ab.pointer,
            Gptr=G.pointer,
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

        CuestXCIntCompute.local_exchange_correlation_gradient(
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

        CuestXCIntCompute.nonlocal_exchange_correlation_gradient(
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

        self.timer.start(key='solve', stream_handle=self.cuest_handle.stream_handle)

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

        # Depends on cuest_primary, cuest_auxiliary, cuest_ao_pair_list, and cuest_xc_int_plan
        self.timer.start(key='df_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_df_int_plan
        self.timer.stop(key='df_int_plan', stream_handle=self.cuest_handle.stream_handle)

        # Depends on cuest_primary and cuest_nlc_grid
        self.timer.start(key='nlc_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_nlc_int_plan
        self.timer.stop(key='nlc_int_plan', stream_handle=self.cuest_handle.stream_handle)

        # Depends on cuest_oe_int_plan
        self.timer.start(key='pcm_int_plan', stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_pcm_int_plan
        self.timer.stop(key='pcm_int_plan', stream_handle=self.cuest_handle.stream_handle)

        self.header()

        if self.benchmark:
            self.timer.start(key="init cuest_ecp_int_plan", stream_handle=self.cuest_handle.stream_handle)
        _ = self.cuest_ecp_int_plan
        if self.benchmark:
            self.timer.stop(key="init cuest_ecp_int_plan", stream_handle=self.cuest_handle.stream_handle)

        if self.benchmark:
            self.timer.start(key="compute_guess", stream_handle=self.cuest_handle.stream_handle)
        self.compute_guess()
        if self.benchmark:
            self.timer.stop(key="compute_guess", stream_handle=self.cuest_handle.stream_handle)

        self.timer.start(key='orthogonalize', stream_handle=self.cuest_handle.stream_handle)
        self.compute_orthogonalization()
        self.timer.stop(key='orthogonalize', stream_handle=self.cuest_handle.stream_handle)

        if self.benchmark:
            self.timer.start(key="occupations", stream_handle=self.cuest_handle.stream_handle)
        self.compute_occupations()
        if self.benchmark:
            self.timer.stop(key="occupations", stream_handle=self.cuest_handle.stream_handle)

        if self.print_level > 0:
            print('DFIntPlan Initialize Time: %11.3E [s]' % (self.timer._times['df_int_plan']))
            print('')

        if self.benchmark:
            self.timer.start(key="solve_diis", stream_handle=self.cuest_handle.stream_handle)
        self.solve_diis()

        self.timer.stop(key='solve', stream_handle=self.cuest_handle.stream_handle)

        self.trailer()

        self.is_solved = True

        
    def header(self):
    
        if self.print_level:
            print('==> cuEST RHF (Python) <==\n')
    
        if self.print_level > 1:

            print('natom  = %4d' % (self.molecule.natom))
            print('charge = %4d' % (self.charge))
            print('')

            print('Primary Basis: %s' % (self.primary_name if self.primary_name is not None else ''))
            print('')
            print(self.primary)

            print('Auxiliary Basis: %s' % (self.auxiliary_name if self.auxiliary_name is not None else ''))
            print('')
            print(self.auxiliary)

            print('MinAO Basis: %s' % (self.minao_name if self.minao_name is not None else ''))
            print('')
            print(self.minao)

            print(self.cuest_xc_int_plan)
        
            print('XC Grid:')
            print('')
            print(self.cuest_xc_grid)

            print('Nonlocal XC Grid:')
            print('')
            print(self.cuest_nlc_grid)

            if self.do_pcm:
                print('PCM:')
                print('')
                print(self.cuest_pcm_int_plan)

    def trailer(self):

        if self.print_level:
            print('cuEST UHF Time: %11.3E [s]\n' % (self.timer.times['solve']))

        if self.print_timings:
            print('cuEST UHF Timing breakdown:')
            print(self.timer.string())

        if self.print_level:
            print('==> End cuEST RHF (Python) <==\n')

    def compute_guess(self):

        if self.print_level > 0:
            print('SAD Guess:\n')

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

        S = self.compute_overlap()

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
            print('threshold_canonical = %11.3E' % (self.threshold_canonical))
            print('nao                 = %11d' % (X.shape[1]))
            print('nmo                 = %11d' % (X.shape[0]))
            print('ndiscard            = %11d' % (X.shape[1] - X.shape[0]))
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
            print('%-5s = %5d' % ('nocca', self.sizes['nocca']))
            print('%-5s = %5d' % ('noccb', self.sizes['noccb']))
            print('%-5s = %5d' % ('nvira', self.sizes['nvira']))
            print('%-5s = %5d' % ('nvirb', self.sizes['nvirb']))
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

        Sab2 = np.sum(Sab.to_numpy()**2)

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

        # Threshold PQ (odd place to report this, but so be it)

        if self.print_level > 0:
            print('threshold_pq = %11.3E' % self.threshold_pq)
            print('')


        # PCM Initialization

        if self.do_pcm:
            self.pcm_q = GPUMatrix(
                nrows=self.cuest_pcm_int_plan.npoint,
                ncols=1,
                dtype=np.double,
                initialize=True,
                )
        # External Nuclear-repulsion energy

        Enuc = self.compute_Enuc()
        self.scalars['Enuc'] = Enuc

        if self.print_level > 0:
            print('Enuc = %24.16E' % (Enuc))
            print('')
        
        S = self.tensors['S']
        X = self.tensors['X']

        # Core Hamiltonian

        if self.benchmark:
            self.timer.start(key="compute_kinetic", stream_handle=self.cuest_handle.stream_handle)
        T = self.compute_kinetic()
        if self.benchmark:
            self.timer.stop(key="compute_kinetic", stream_handle=self.cuest_handle.stream_handle)
        if self.benchmark:
            self.timer.start(key="compute_potential", stream_handle=self.cuest_handle.stream_handle)
        V = self.compute_potential()
        if self.benchmark:
            self.timer.stop(key="compute_potential", stream_handle=self.cuest_handle.stream_handle)

        if self.do_ecp:
            Vecp = self.compute_ecp_potential()

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

        diis = DIIS(max_nvector=self.diis_max_nvector)

        if self.print_level > 1:
            print('DIIS:')
            print('max nvector = %11d' % self.diis_max_nvector)
            print('')

        if self.print_level > 1:
            print('Convergence Options:')
            print('max iterations = %11d' % self.maxiter)
            print('g convergence  = %11.3E' % self.g_convergence)
            print('')

        # Main SCF Iterative Loop 

        start = time.time()
        self.is_converged = False
        Eold = 0.0
        E = 0.0
        dG = 1.0
        if self.print_level > 1:
            print('%-4s: %24s %11s %11s %8s' % ('Iter', 'Energy', 'dE', 'dG', 'Time[s]'))
        if self.benchmark:
            self.timer.start(key="iterations", stream_handle=self.cuest_handle.stream_handle)
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

            # Find slice count
            if self.benchmark:
                self.timer.start(key="find slice counts", stream_handle=self.cuest_handle.stream_handle)
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
            if self.benchmark:
               self.timer.stop(key="find slice counts", stream_handle=self.cuest_handle.stream_handle)

            # Fock Matrix

            if self.benchmark:
                self.timer.start(key="J", stream_handle=self.cuest_handle.stream_handle)
            J = self.compute_coulomb(D=D)
            if self.benchmark:
                self.timer.stop(key="J", stream_handle=self.cuest_handle.stream_handle)

            if self.benchmark:
                self.timer.start(key="Ka", stream_handle=self.cuest_handle.stream_handle)
            Ka = self.compute_exchange(
                Cocc=Cocca,
                dfk_int8_slice_count=dfk_int8_slice_count,
                dfk_int8_modulus_count=dfk_int8_modulus_count,
                )
            if self.benchmark:
                self.timer.stop(key="Ka", stream_handle=self.cuest_handle.stream_handle)

            if self.benchmark:
                self.timer.start(key="Kb", stream_handle=self.cuest_handle.stream_handle)
            Kb = self.compute_exchange(
                Cocc=Coccb,
                dfk_int8_slice_count=dfk_int8_slice_count,
                dfk_int8_modulus_count=dfk_int8_modulus_count,
                )
            if self.benchmark:
                self.timer.stop(key="Kb", stream_handle=self.cuest_handle.stream_handle)

            if self.benchmark:
                self.timer.start(key="Vxc", stream_handle=self.cuest_handle.stream_handle)
            Exc, Vxca, Vxcb = self.compute_local_exchange_correlation(Cocca=Cocca,Coccb=Coccb)
            if self.benchmark:
                self.timer.stop(key="Vxc", stream_handle=self.cuest_handle.stream_handle)

            if self.benchmark:
                self.timer.start(key="NLC", stream_handle=self.cuest_handle.stream_handle)
            Enlc, Vnlc = self.compute_nonlocal_exchange_correlation(Cocca=Cocca,Coccb=Coccb)
            if self.benchmark:
                self.timer.stop(key="NLC", stream_handle=self.cuest_handle.stream_handle)

            # NOTE: This is typically where other electron-dependent Fock
            # matrix terms are added, e.g., PCM, wK.

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

            # SCF Energy

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
            E += -0.5 * GPUMatrixUtility.ddot(
                x=Da,   
                y=Ka,
                )
            E += -0.5 * GPUMatrixUtility.ddot(
                x=Db,
                y=Kb,
                )

            # Include PCM
            if self.benchmark:
                self.timer.start(key="PCM", stream_handle=self.cuest_handle.stream_handle)
            if self.do_pcm:
                Epcm, Vpcm, q_out = self.compute_pcm_energy_and_potential(
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
            if self.benchmark:
                self.timer.stop(key="PCM", stream_handle=self.cuest_handle.stream_handle)

            E += Exc
            E += Enlc
            self.scalars['Escf'] = E

            # Add Vxc to the Fock matrix after calculating the energy

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

            dE = E - Eold
            Eold = E

            # Orbital Gradient (zero at convergence)
            # G = XSDFX' - XFDSX'

            # G = np.dot(np.dot(np.dot(np.dot(X, S), D), F), X.T)
            # Faster in large basis sets

            if self.benchmark:
                self.timer.start(key="Orbital gradient", stream_handle=self.cuest_handle.stream_handle)
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

            Ga = GPUMatrixUtility.matrix_multiply(
                mat1=L,
                mat2=R,
                transpose1=True,
                transpose2=False,
                )
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

            Gb = GPUMatrixUtility.matrix_multiply(
                mat1=L,
                mat2=R,
                transpose1=True,
                transpose2=False,
                )
            Gb = GPUMatrixUtility.dgeam(
                mat1=Gb,
                alpha=1.0,
                transpose1=False,
                mat2=Gb,
                beta=-1.0,
                transpose2=True,
                )
            if self.benchmark:
                self.timer.stop(key="Orbital gradient", stream_handle=self.cuest_handle.stream_handle)

            Ga_np = Ga.to_numpy()
            Gb_np = Gb.to_numpy()

            dGa = np.max(np.abs(Ga_np))
            dGb = np.max(np.abs(Gb_np))
            dG = max(dGa, dGb)

            # Convergence Trace

            stop = time.time()
            if self.print_level:
                print('%-4d: %24.16E %11.3E %11.3E %8.3f (%2d moduli, %2d slices)' % (iteration, E, dE, dG, stop-start, dfk_int8_modulus_count, dfk_int8_slice_count))
            start = stop

            # Convergence Check

            if iteration > 0 and dG < self.g_convergence:
                self.is_converged = True
                break

            # This ensures that we do not extrapolate/diagonalize if maxiter is reached
            if iteration >= self.maxiter - 1:
                break

            # Unconditional DIIS history flush every diis_flush_niter iterations
            if self.diis_flush_niter > 0 and (iteration + 1) % self.diis_flush_niter == 0:
                diis = DIIS(max_nvector=self.diis_max_nvector)

            # DIIS Extrapolation (also updates DIIS history)

            if self.benchmark:
                self.timer.start(key="DIIS extrapolate", stream_handle=self.cuest_handle.stream_handle)

            if iteration > 0:
                Fa, Fb = diis.iterate(
                    state_vectors=[Fa, Fb],
                    error_vectors=[Ga, Gb],
                    )

            if self.benchmark:
                self.timer.stop(key="DIIS extrapolate", stream_handle=self.cuest_handle.stream_handle)

            # Fock Matrix Diagonalization

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

            if self.benchmark:
                self.timer.start(key="eigh alpha", stream_handle=self.cuest_handle.stream_handle)
            epsa, U2a = GPUMatrixUtility.eigh(matrix=F2a)
            if self.benchmark:
                self.timer.stop(key="eigh alpha", stream_handle=self.cuest_handle.stream_handle)

            Ca = GPUMatrixUtility.matrix_multiply(
                mat1=U2a,
                mat2=X,
                transpose1=True,
                transpose2=False,
                )

            # New Occupied Orbitals and Density Matrix

            if self.benchmark:
                self.timer.start(key="Form Da", stream_handle=self.cuest_handle.stream_handle)
            Ca_np = Ca.to_numpy()
            Cocca_np = Ca_np[:self.sizes['nocca'], :]
            Cocca = GPUMatrix.from_numpy(Cocca_np)
            
            Da = GPUMatrixUtility.matrix_multiply(
                mat1=Cocca,
                mat2=Cocca,
                transpose1=True,
                transpose2=False,
                )
            if self.benchmark:
                self.timer.stop(key="Form Da", stream_handle=self.cuest_handle.stream_handle)

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

            if self.benchmark:
                self.timer.start(key="eigh beta", stream_handle=self.cuest_handle.stream_handle)
            epsb, U2b = GPUMatrixUtility.eigh(matrix=F2b)
            if self.benchmark:
                self.timer.stop(key="eigh beta", stream_handle=self.cuest_handle.stream_handle)

            Cb = GPUMatrixUtility.matrix_multiply(
                mat1=U2b,
                mat2=X,
                transpose1=True,
                transpose2=False,
                )

            # New Occupied Orbitals and Density Matrix

            if self.benchmark:
                self.timer.start(key="Form Db", stream_handle=self.cuest_handle.stream_handle)
            Cb_np = Cb.to_numpy()
            Coccb_np = Cb_np[:self.sizes['noccb'], :]
            Coccb = GPUMatrix.from_numpy(Coccb_np)
            
            Db = GPUMatrixUtility.matrix_multiply(
                mat1=Coccb,
                mat2=Coccb,
                transpose1=True,
                transpose2=False,
                )
            if self.benchmark:
                self.timer.stop(key="Form Db", stream_handle=self.cuest_handle.stream_handle)

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
            if self.is_converged:
                self.compute_s2()
                print('SCF Converged\n')
                S = 0.5 * (self.multiplicity - 1)
                S2_exact = S * (S + 1.0)
                print('<Sz>                = %11.3E' % (self.scalars['Sz']))
                print('<S2>                = %11.3E' % (self.scalars['S2']))
                print('Spin contamination  = %11.3E' % (self.scalars['S2'] - S2_exact))
                print('')
            else:
                print('SCF Failed\n')

        # => Print Final Energy <= #

        if self.print_level:
            print('SCF Energy = %24.16E\n' % E)

        if self.print_level > 0:
            print('=> End Solve DIIS <=\n')
        if self.benchmark:
            self.timer.stop(key="iterations", stream_handle=self.cuest_handle.stream_handle)

    def compute_energy(self):

        if not self.is_solved:
            self.solve()

        if not self.is_converged:
            raise RuntimeError('Not converged')
    
        return self.scalars['Escf']

    def compute_gradient(self):

        if not self.is_solved:
            self.solve()

        if not self.is_converged:
            raise RuntimeError('Not converged')
        
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
        # alpha
        epsa_occ = self.tensors['epsa'].to_numpy()[:self.sizes['nocca']].reshape(-1)
        Cocca_np = Cocca.to_numpy()
        Va_np = Cocca_np * epsa_occ[:, None]
        Wa_np = Va_np.T @ Cocca_np
        Wa = GPUMatrix.from_numpy(Wa_np)

        epsb_occ = self.tensors['epsb'].to_numpy()[:self.sizes['noccb']].reshape(-1)
        Coccb_np = Coccb.to_numpy()
        Vb_np = Coccb_np * epsb_occ[:, None]
        Wb_np = Vb_np.T @ Coccb_np
        Wb = GPUMatrix.from_numpy(Wb_np)

        # Total energy-weighted density
        W = Wa.clone()
        GPUMatrixUtility.daxpy(
            alpha=1.0, 
            x=Wb, 
            y=W,
            )

        Dt = GPUMatrixUtility.dgeam(
            mat1=D,
            alpha=0.5,
            transpose1=False,
            mat2=D,
            beta=0.5,
            transpose2=True,
        )
        Wt = GPUMatrixUtility.dgeam(
            mat1=W,
            alpha=0.5,
            transpose1=False,
            mat2=W,
            beta=0.5,
            transpose2=True,
        )
 
        # Nuclear gradient (classical)
 
        Gnuc = self.compute_Enuc_gradient()
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=Gnuc,
            )

        # Overlap gradient

        GS = self.compute_overlap_gradient(Wt)
        GPUMatrixUtility.dscal(
            alpha=-1.0,
            x=GS,
            )

        # Kinetic gradient

        GT = self.compute_kinetic_gradient(Dt)
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=GT,
            )

        # Potential gradient

        GV = self.compute_potential_gradient(Dt)
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=GV,
            )

        # ECP gradient

        Gecp = None
        if self.do_ecp:
            Gecp = self.compute_ecp_gradient(Dt)
            GPUMatrixUtility.dscal(
                alpha=+1.0,
                x=Gecp,
                )

        # United Coulomb + Exchange gradient

        GJK = self.compute_coulomb_and_exchange_gradient(
            scaleJ=+0.5,
            DJ=Dt,
            scaleK=-0.5,
            CoccsK=[Cocca,Coccb],
            )

        # Local exchange-correlation gradient

        GVxc = self.compute_local_exchange_correlation_gradient(Cocca=Cocca,Coccb=Coccb)
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=GVxc,
            )

        # Nonlocal exchange-correlation gradient

        GVnlc = self.compute_nonlocal_exchange_correlation_gradient(Cocca=Cocca,Coccb=Coccb)
        GPUMatrixUtility.dscal(
            alpha=+1.0,
            x=GVnlc,
            )

        # PCM gradient

        Gpcm = None
        if self.do_pcm:
            Gpcm = self.compute_pcm_gradient(
                q=self.pcm_q,
                D=Dt)

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

        return G

