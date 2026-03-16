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

import numpy as np

from .memoized_property import memoized_property

from .molecule import Molecule
from .ao_basis import AOBasis
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

from .gpu_matrix import GPUMatrix
from .gpu_matrix_utility import GPUMatrixUtility

import time

class RHF(object):

    def __init__(
        self,
        *,
        cuest_handle : CuestHandle,
        molecule : Molecule,
        charge : int,
        xc_functional_name : str = None,
        primary : AOBasis,
        auxiliary : AOBasis,
        minao : AOBasis,
        primary_name : str = None,
        auxiliary_name : str = None,
        minao_name : str = None,
        print_level : int = 2,
        threshold_pq : float = 1.0E-12,
        threshold_canonical : float = 1.0E-6,
        diis_max_nvector : int = 6,
        maxiter : int = 100,
        g_convergence : float = 1.0E-6,
        xc_threshold_collocation : float = 1.0e-18,
        xc_grid_level : int = 2,
        xc_grid_family : str = "GRID",
        nlc_threshold_collocation : float = 1.0e-18,
        nlc_grid_level : int = 1,
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

        self.benchmark=benchmark
        if self.benchmark:
            import cuest_core as ce
            self.timer = ce.Timer()

        self.cuest_handle = cuest_handle

        self.molecule = molecule
        self.charge = charge

        self.xc_functional = XCFunctionalInfo.string_to_enum(xc_functional_name)

        self.primary = primary
        self.auxiliary = auxiliary
        self.minao = minao

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

        self.threshold_pq = threshold_pq
        self.threshold_canonical = threshold_canonical
        self.diis_max_nvector = diis_max_nvector
        self.maxiter = maxiter
        self.g_convergence = g_convergence

        self.pcm_num_angular_points_per_heavy_atom = pcm_num_angular_points_per_heavy_atom
        self.pcm_num_angular_points_per_hydrogen_atom = pcm_num_angular_points_per_hydrogen_atom
        self.pcm_epsilon = pcm_epsilon
        self.pcm_x_prefactor = pcm_x_prefactor
        self.pcm_cutoff = pcm_cutoff
        self.pcm_convergence_tol = pcm_convergence_tol
        self.pcm_q = None

        self.dfk_int8_slice_count_start = dfk_int8_slice_count_start
        self.dfk_int8_modulus_count_start = dfk_int8_modulus_count_start
        self.dfk_int8_slice_count_end = dfk_int8_slice_count_end
        self.dfk_int8_modulus_count_end = dfk_int8_modulus_count_end

        if primary.natom != molecule.natom: raise RuntimeError('primary.natom != molecule.natom')
        if auxiliary.natom != molecule.natom: raise RuntimeError('auxiliary.natom != molecule.natom')
        if minao.natom != molecule.natom: raise RuntimeError('minao.natom != molecule.natom')

        self.sizes = {}
        self.scalars = {}
        self.tensors = {}

        self.is_solved = False
        self.is_converged = False

        self.gpu_xyz = GPUMatrix.from_numpy(self.molecule.xyz)
        self.gpu_Z = GPUMatrix.from_numpy(self.molecule.Z.reshape(-1,1))
        GPUMatrixUtility.scale(
            matrix=self.gpu_Z,
            scale=-1.0,
            )

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
                effective_nuclear_charges=self.molecule.Z,
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

        if self.cuest_df_int_plan.exchange_scale == 0.0:
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
        Cocc,
        ):

        if self.xc_functional == 0:
            return 0.0, None

        Vxc = GPUMatrix(
            nrows=self.primary.nao,
            ncols=self.primary.nao,
            dtype=np.double,
            )

        Exc = CuestXCIntCompute.local_exchange_correlation(
            handle=self.cuest_handle,
            xc_int_plan=self.cuest_xc_int_plan,
            Vxcptr=Vxc.pointer,
            nocc=Cocc.shape[0],
            Coccptr=Cocc.pointer,
            )

        return Exc, Vxc

    def compute_nonlocal_exchange_correlation(
        self,   
        Cocc,
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
            nocc=Cocc.shape[0],
            Coccptr=Cocc.pointer,
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
        assert len(CoccsK) == 1

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
            CoccsKptr=CoccsK[0].pointer,
            Gptr=G.pointer,
            )

        return G

    def compute_local_exchange_correlation_gradient(
        self,   
        Cocc,
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
            nocc=Cocc.shape[0],
            Coccptr=Cocc.pointer,
            )

        return G

    def compute_nonlocal_exchange_correlation_gradient(
        self,   
        Cocc,
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
            nocc=Cocc.shape[0],
            Coccptr=Cocc.pointer,
            )

        return G

    # => Main Solve Routines <= #
    
    def solve(self):

        if self.benchmark:
            self.timer.start("SCF Energy")
        self.is_solved = False
        self.is_converged = False

        self.header()

        # Plan initializations (to cleanly separate from iterations).  The orthogonalization
        # calls for S, so most of these are automatically instantiated in that routine.  Build
        # them here to get an idea of how long they take.
        if self.benchmark:
            self.timer.start("init cuest_ao_pair_list")
        _ = self.cuest_ao_pair_list
        if self.benchmark:
            self.timer.stop("init cuest_ao_pair_list")

        if self.benchmark:
            self.timer.start("init cuest_oe_int_plan")
        _ = self.cuest_oe_int_plan
        if self.benchmark:
            self.timer.stop("init cuest_oe_int_plan")

        if self.benchmark:
            self.timer.start("init cuest_xc_int_plan")
        _ = self.cuest_xc_int_plan
        if self.benchmark:
            self.timer.stop("init cuest_xc_int_plan")

        if self.benchmark:
            self.timer.start("init cuest_df_int_plan")
        start = time.time()
        _ = self.cuest_df_int_plan
        stop = time.time()
        if self.benchmark:
            self.timer.stop("init cuest_df_int_plan")
        
        if self.benchmark:
            self.timer.start("init cuest_nlc_int_plan")
        _ = self.cuest_nlc_int_plan
        if self.benchmark:
            self.timer.stop("init cuest_nlc_int_plan")

        if self.benchmark:
            self.timer.start("init cuest_pcm_int_plan")
        _ = self.cuest_pcm_int_plan
        if self.benchmark:
            self.timer.stop("init cuest_pcm_int_plan")

        if self.benchmark:
            self.timer.start("compute_guess")
        self.compute_guess()
        if self.benchmark:
            self.timer.stop("compute_guess")

        if self.benchmark:
            self.timer.start("compute_orthogonalization")
        self.compute_orthogonalization()
        if self.benchmark:
            self.timer.stop("compute_orthogonalization")

        if self.benchmark:
            self.timer.start("occupations")
        self.compute_occupations()
        if self.benchmark:
            self.timer.stop("occupations")

        if self.print_level > 0:
            print('DFIntPlan Initialize Time: %11.3E [s]' % (stop - start))
            print('')

        if self.benchmark:
            self.timer.start("solve_diis")
        self.solve_diis()
        if self.benchmark:
            self.timer.stop("solve_diis")

        self.trailer()

        self.is_solved = True

        if self.benchmark:
            self.timer.stop("SCF Energy")
            print(self.timer)
        
    def header(self):
    
        self.start_time = time.time()
        
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
            print('cuEST RHF Time: %11.3E [s]\n' % (time.time() - self.start_time))
        
        del self.start_time
        
        if self.print_level:
            print('==> End cuEST RHF (Python) <==\n')

    def compute_guess(self):

        if self.print_level > 0:
            print('SAD Guess:\n')

        self.sad_guess = SADGuess.build(
            molecule=self.molecule,
            primary=self.primary,
            minao=self.minao,
            )

        # For now this will be size (nminao, nao), which is usually larger than
        # (nocc, nao) due to fractional occupations in SAD
        self.tensors['Cocc'] = GPUMatrix.from_numpy(self.sad_guess.compute_Cocc())

        # For now this is not a pure state, and corresponds to the neutral SAD
        # guess
        self.tensors['D'] = GPUMatrix.from_numpy(self.sad_guess.compute_Docc())

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

        Q = np.sum(self.molecule.Z) - self.charge
        nocc = int(Q / 2)
        
        if 2 * nocc != Q: raise RuntimeError('molecule is not closed shell')

        nvir = self.tensors['X'].shape[0] - nocc

        self.sizes['nocc'] = nocc
        self.sizes['nvir'] = nvir

        if self.print_level > 0:
            print('%-5s = %5d' % ('nocc', self.sizes['nocc']))
            print('%-5s = %5d' % ('nvir', self.sizes['nvir']))
            print('')

    def compute_Enuc(self):
            
        xyz = self.molecule.xyz
        Z = self.molecule.Z

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
        Z = self.molecule.Z

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
            self.pcm_q.zero()

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
            self.timer.start("compute_kinetic")
        T = self.compute_kinetic()
        if self.benchmark:
            self.timer.stop("compute_kinetic")
        if self.benchmark:
            self.timer.start("compute_potential")
        V = self.compute_potential()
        if self.benchmark:
            self.timer.stop("compute_potential")

        # NOTE: This is typically where other static external potential terms
        # are added, e.g., ECP, QM/MM electrostatics, external perturbations,
        # etc.

        H = T.clone()
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=V,
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
        print('%-4s: %24s %11s %11s %8s' % ('Iter', 'Energy', 'dE', 'dG', 'Time[s]'))
        if self.benchmark:
            self.timer.start("iterations")
        for iteration in range(self.maxiter):

            D = self.tensors['D'] 
            Cocc = self.tensors['Cocc']

            # Find slice count
            if self.benchmark:
                self.timer.start("find slice counts")
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
               self.timer.stop("find slice counts")

            # Fock Matrix

            if self.benchmark:
                self.timer.start("J")
            J = self.compute_coulomb(D=D)
            if self.benchmark:
                self.timer.stop("J")

            if self.benchmark:
                self.timer.start("K")
            K = self.compute_exchange(
                Cocc=Cocc,
                dfk_int8_slice_count=dfk_int8_slice_count,
                dfk_int8_modulus_count=dfk_int8_modulus_count,
                )
            if self.benchmark:
                self.timer.stop("K")

            if self.benchmark:
                self.timer.start("Vxc")
            Exc, Vxc = self.compute_local_exchange_correlation(Cocc=Cocc)
            if self.benchmark:
                self.timer.stop("Vxc")

            if self.benchmark:
                self.timer.start("NLC")
            Enlc, Vnlc = self.compute_nonlocal_exchange_correlation(Cocc=Cocc)
            if self.benchmark:
                self.timer.stop("NLC")

            # NOTE: This is typically where other electron-dependent Fock
            # matrix terms are added, e.g., PCM, wK.

            F = H.clone()
            GPUMatrixUtility.daxpy(
                alpha=2.0,
                x=J,
                y=F,
                )
            GPUMatrixUtility.daxpy(
                alpha=-1.0,
                x=K,
                y=F,
                )

            # SCF Energy

            E = 0.0
            E += Enuc
            E += 1.0 * GPUMatrixUtility.ddot(
                x=D,
                y=H,
                )
            E += 1.0 * GPUMatrixUtility.ddot(
                x=D,
                y=F,
                )

            # Include PCM
            if self.benchmark:
                self.timer.start("PCM")
            if self.do_pcm:
                # PCM expects the total density, so temporarily scale by 2.0
                GPUMatrixUtility.scale(
                    matrix=D,
                    scale=2.0,
                    )
                Epcm, Vpcm, q_out = self.compute_pcm_energy_and_potential(
                    q=self.pcm_q, 
                    D=D)
                GPUMatrixUtility.scale(
                    matrix=D,
                    scale=0.5,
                    )

                self.pcm_q = q_out

                E += Epcm
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vpcm,
                    y=F,
                    )
            if self.benchmark:
                self.timer.stop("PCM")

            E += Exc
            E += Enlc
            self.scalars['Escf'] = E

            # Add Vxc to the Fock matrix after calculating the energy

            if Vxc is not None:
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vxc,
                    y=F,
                    )
            if Vnlc is not None:
                GPUMatrixUtility.daxpy(
                    alpha=1.0,
                    x=Vnlc,
                    y=F,
                    )

            dE = E - Eold
            Eold = E

            # Orbital Gradient (zero at convergence)
            # G = XSDFX' - XFDSX'

            # G = np.dot(np.dot(np.dot(np.dot(X, S), D), F), X.T)
            # Faster in large basis sets

            if self.benchmark:
                self.timer.start("Orbital gradient")
            L1 = GPUMatrixUtility.matrix_multiply(
                mat1=Cocc,
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
                mat1=Cocc,
                mat2=F,
                transpose1=False,
                transpose2=False,
                )
            R = GPUMatrixUtility.matrix_multiply(
                mat1=R1,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )

            G = GPUMatrixUtility.matrix_multiply(
                mat1=L,
                mat2=R,
                transpose1=True,
                transpose2=False,
                )
            G = GPUMatrixUtility.dgeam(
                mat1=G,
                alpha=1.0,
                transpose1=False,
                mat2=G,
                beta=-1.0,
                transpose2=True,
                )
            if self.benchmark:
                self.timer.stop("Orbital gradient")

            G_np = G.to_numpy()

            dG = np.max(np.abs(G_np))

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

            # DIIS Extrapolation (also updates DIIS history)

            if self.benchmark:
                self.timer.start("DIIS extrapolate")

            if iteration > 0:
                F = diis.iterate(
                    state_vector=F,
                    error_vector=G,
                    )
            if self.benchmark:
                self.timer.stop("DIIS extrapolate")

            # Fock Matrix Diagonalization

            tmp = GPUMatrixUtility.matrix_multiply(
                mat1=X,
                mat2=F,
                transpose1=False,
                transpose2=False,
                )
            F2 = GPUMatrixUtility.matrix_multiply(
                mat1=tmp,
                mat2=X,
                transpose1=False,
                transpose2=True,
                )

            if self.benchmark:
                self.timer.start("eigh")
            eps, U2 = GPUMatrixUtility.eigh(matrix=F2)
            if self.benchmark:
                self.timer.stop("eigh")

            C = GPUMatrixUtility.matrix_multiply(
                mat1=U2,
                mat2=X,
                transpose1=True,
                transpose2=False,
                )

            # New Occupied Orbitals and Density Matrix

            if self.benchmark:
                self.timer.start("Form D")
            C_np = C.to_numpy()
            Cocc_np = C_np[:self.sizes['nocc'], :]
            Cocc = GPUMatrix.from_numpy(Cocc_np)
            
            D = GPUMatrixUtility.matrix_multiply(
                mat1=Cocc,
                mat2=Cocc,
                transpose1=True,
                transpose2=False,
                )
            if self.benchmark:
                self.timer.stop("Form D")
            
            self.tensors['D'] = D
            self.tensors['Cocc'] = Cocc

            self.tensors['C'] = C
            self.tensors['eps'] = eps

        # => Print Convergence <= #

        if self.print_level > 0:
            if self.is_converged:
                print('SCF Converged\n')
            else:
                print('SCF Failed\n')

        # => Print Final Energy <= #

        if self.print_level:
            print('SCF Energy = %24.16E\n' % E)

        if self.print_level > 0:
            print('=> End Solve DIIS <=\n')
        if self.benchmark:
            self.timer.stop("iterations")

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
    
        Cocc = self.tensors['Cocc']
        D = self.tensors['D']

        # Energy-weighted density matrix
    
        eps_occ_np = self.tensors['eps'].to_numpy()[:self.sizes['nocc']]
        Cocc_np = Cocc.to_numpy()
        Vocc_np = np.einsum('ip,id->ipd', Cocc_np, eps_occ_np).reshape(Cocc_np.shape[0], Cocc_np.shape[1])
        W_np = np.dot(Vocc_np.T, Cocc_np)
        W = GPUMatrix.from_numpy(W_np)

        # Nuclear gradient (classical)

        Gnuc = self.compute_Enuc_gradient()
        GPUMatrixUtility.scale(
            matrix=Gnuc,
            scale=+1.0,
            )

        # Overlap gradient

        GS = self.compute_overlap_gradient(W)
        GPUMatrixUtility.scale(
            matrix=GS,
            scale=-2.0,
            )

        # Kinetic gradient

        GT = self.compute_kinetic_gradient(D)
        GPUMatrixUtility.scale(
            matrix=GT,
            scale=+2.0,
            )

        # Potential gradient

        GV = self.compute_potential_gradient(D)
        GPUMatrixUtility.scale(
            matrix=GV,
            scale=+2.0,
            )

        # United Coulomb + Exchange gradient

        GJK = self.compute_coulomb_and_exchange_gradient(
            scaleJ=+2.0,
            DJ=D,
            scaleK=-1.0,
            CoccsK=[Cocc],
            )

        # Local exchange-correlation gradient

        GVxc = self.compute_local_exchange_correlation_gradient(Cocc)
        GPUMatrixUtility.scale(
            matrix=GVxc,
            scale=+1.0,
            )

        # Nonlocal exchange-correlation gradient

        GVnlc = self.compute_nonlocal_exchange_correlation_gradient(Cocc)
        GPUMatrixUtility.scale(
            matrix=GVnlc,
            scale=+1.0,
            )

        # PCM gradient

        Gpcm = None
        if self.do_pcm:
            GPUMatrixUtility.scale(
                matrix=D,
                scale=2.0,
                )
            Gpcm = self.compute_pcm_gradient(
                q=self.pcm_q,
                D=D)
            GPUMatrixUtility.scale(
                matrix=D,
                scale=0.5,
                )

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

