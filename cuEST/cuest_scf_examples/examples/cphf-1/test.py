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
#
# This is an example demonstrating how to use the nonsymmetric exchange
# compute routines and multipole integral compute routines to solve the
# CPHF equations to obtain an RHF polarizability. This example is limited
# to gas-phase RHF polarizabilities. 
#
# This implementation uses the cuest_scf.RHF object to access compute
# routines, pre-built data structures and the results of the RHF calculation.
#
#

import cuest.bindings as ce
import cuest_scf
import ctypes
import numpy as np

from kernel_helper import CPHFKernelHelper

kernel_helper = CPHFKernelHelper()

def rhf_dipole_rhs(
    *,
    rhf : cuest_scf.RHF,
    Cocc : cuest_scf.GPUMatrix,
    Cvir : cuest_scf.GPUMatrix,
    ):

    # Array sizing
    nocc = Cocc.shape[0]
    nvir = Cvir.shape[0]
    nao = Cocc.shape[1]

    # Returns the dipole matrices as a list of cuest_scf.GPUMatrix
    mu = rhf.compute_dipole()

    # Allocate an array to store the transformed dipole integrals
    rhs = cuest_scf.GPUMatrix(
        nrows=3,
        ncols=nocc * nvir,
        dtype=np.double,
        )

    # Temporary space for the AO to MO transformation
    Min = cuest_scf.GPUMatrix(
        nrows=nocc,
        ncols=nao,
        dtype=np.double,
        )

    # AO to MO transformation of dipole integrals
    for n in range(3):
        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=False,
            m=nocc,
            n=nao,
            k=nao,
            alpha=1.0,
            a=Cocc,
            offa=0,
            lda=nao,
            b=mu[n],
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Min,
            offc=0,
            ldc=nao,
            )
        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=True,
            m=nocc,
            n=nvir,
            k=nao,
            alpha=1.0,
            a=Min,
            offa=0,
            lda=nao,
            b=Cvir,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=rhs,
            offc=n * nocc * nvir,
            ldc=nvir,
            )

    return rhs


def rhf_mo_hessian_vector_product(
    *,
    rhf : cuest_scf.RHF,
    Cocc : cuest_scf.GPUMatrix,
    Cvir : cuest_scf.GPUMatrix,
    eps_occ : cuest_scf.GPUMatrix,
    eps_vir : cuest_scf.GPUMatrix,
    tia : cuest_scf.GPUMatrix,
    ):

    nocc = Cocc.shape[0]
    nvir = Cvir.shape[0]
    nao = Cocc.shape[1]

    # Factor pseudodensity X_{mn} = \sum_{ia} C_{im} t_{ia} C_{an}
    # As: X_{mn} = \sum_{i} C_{im} T_{in} = \sum_{i} C_{im} ( \sum_{a} t_{ia} C_{an} )
    Tocc = []
    for n in range(3):
        Tocc.append(cuest_scf.GPUMatrix(
            nrows=nocc,
            ncols=nao,
            dtype=np.double,
            ))
        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=False,
            m=nocc,
            n=nao,
            k=nvir,
            alpha=1.0,
            a=tia,
            offa=n * nocc * nvir,
            lda=nvir,
            b=Cvir,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Tocc[n],
            offc=0,
            ldc=nao,
            )

    # Compute all three exchange matrices in one shot
    K = rhf.compute_nonsymmetric_exchange(
        Cocc_left=Cocc,
        Coccs_right=Tocc,
        )

    # Compute Coulomb matrices one at a time
    J = []
    for n in range(3):
        Xmn = cuest_scf.GPUMatrix(
            nrows=nao,
            ncols=nao,
            dtype=np.double,
            )
        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=True,
            transb=False,
            m=nao,
            n=nao,
            k=nocc,
            alpha=1.0,
            a=Cocc,
            offa=0,
            lda=nao,
            b=Tocc[n],
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Xmn,
            offc=0,
            ldc=nao,
            )
        J.append(rhf.compute_coulomb(
            D=Xmn,
            ))

    # Allocate space for the MO Hessian - vector product
    Ria = cuest_scf.GPUMatrix(
        nrows=3,
        ncols=nocc * nvir,
        dtype=np.double,
        )

    # Temporary space for the AO to MO transformation
    Ain = cuest_scf.GPUMatrix(
        nrows=nocc,
        ncols=nao,
        dtype=np.double,
        )

    # Assemble MO Hessian - vector products
    for n in range(3):
        Ax = cuest_scf.GPUMatrixUtility.dgeam(
            transpose1=False,
            alpha=-1.0,
            mat1=K[n],
            transpose2=True,
            beta=-1.0,
            mat2=K[n],
            )
        cuest_scf.GPUMatrixUtility.daxpy(
            n=nao*nao,
            alpha=4.0,
            x=J[n],
            offx=0,
            incx=1,
            y=Ax,
            offy=0,
            incy=1,
            )
        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=False,
            m=nocc,
            n=nao,
            k=nao,
            alpha=1.0,
            a=Cocc,
            offa=0,
            lda=nao,
            b=Ax,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Ain,
            offc=0,
            ldc=nao,
            )
        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=True,
            m=nocc,
            n=nvir,
            k=nao,
            alpha=1.0,
            a=Ain,
            offa=0,
            lda=nao,
            b=Cvir,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Ria,
            offc=n * nocc * nvir,
            ldc=nvir,
            )

    # Add the Fock contributions to the Hessian-vector product
    kernel_helper.ria_update(
        nrhs=3,
        nocc=nocc,
        nvir=nvir,
        ria_ptr=Ria.pointer,
        tia_ptr=tia.pointer,
        eps_occ_ptr=eps_occ.pointer,
        eps_vir_ptr=eps_vir.pointer,
        )

    return Ria

def rhf_polarizability(
    *,
    rhf : cuest_scf.RHF,
    maxiter : int = 100,
    tolerance : float = 1.0e-8,
    ):

    # Grab some array sizes from the cuest_scf.RHF object
    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']
    nao = rhf.tensors['C'].shape[1]
    
    # Make copies of occupied/virtual orbitals
    Cocc = cuest_scf.GPUMatrix(
        nrows=nocc,
        ncols=nao,
        dtype=np.double,
        )
    Cvir = cuest_scf.GPUMatrix(
        nrows=nvir,
        ncols=nao,
        dtype=np.double,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc * nao,
        x=rhf.tensors['C'],
        offx=0,
        incx=1,
        y=Cocc,
        offy=0,
        incy=1,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nvir * nao,
        x=rhf.tensors['C'],
        offx=nocc * nao,
        incx=1,
        y=Cvir,
        offy=0,
        incy=1,
        )
    
    # Make copies of occupied/virtual orbital energies
    eps_occ = cuest_scf.GPUMatrix(
        nrows=1,
        ncols=nocc,
        dtype=np.double,
        )
    eps_vir = cuest_scf.GPUMatrix(
        nrows=1,
        ncols=nvir,
        dtype=np.double,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc,
        x=rhf.tensors['eps'],
        offx=0,
        incx=1,
        y=eps_occ,
        offy=0,
        incy=1,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nvir,
        x=rhf.tensors['eps'],
        offx=nocc,
        incx=1,
        y=eps_vir,
        offy=0,
        incy=1,
        )
   
    # Form the right-hand side of the CPHF equations
    rhs = rhf_dipole_rhs(
        rhf=rhf,
        Cocc=Cocc,
        Cvir=Cvir,
        )
    
    # Allocate space for the x solution vectors
    x = cuest_scf.GPUMatrix(
        nrows=rhs.nrows,
        ncols=rhs.ncols,
    )

    # r = b - A * x
    # Guess used is x = 0
    r = rhs.clone()

    # z = M^{-1} r
    z = cuest_scf.GPUMatrix(
        nrows=rhs.nrows,
        ncols=rhs.ncols,
    )
    kernel_helper.apply_preconditioner(
        nrhs=3,
        nocc=nocc,
        nvir=nvir,
        zia_ptr=z.pointer,
        ria_ptr=r.pointer,
        eps_occ_ptr=eps_occ.pointer,
        eps_vir_ptr=eps_vir.pointer,
        )

    # p = z
    p = z.clone()

    rz_old = cuest_scf.GPUMatrixUtility.ddot(
        x=r,
        y=z,
    )

    r_norm_sq = cuest_scf.GPUMatrixUtility.ddot(
        x=r,
        y=r,
    )
    r_norm = np.sqrt(r_norm_sq)

    converged = False
    iteration = 0

    print('CPHF Iterations:')
    print('')
    print('Iter:       Residual')
    print('%-4d:     %10.4e' % (iteration, r_norm))

    if r_norm <= tolerance:
        converged = True
        maxiter=0

    for iteration in range(1, maxiter + 1):
        Ap = cuest_scf.GPUMatrix(
            nrows=rhs.nrows,
            ncols=rhs.ncols,
        )
        Ap = rhf_mo_hessian_vector_product(
            rhf=rhf,
            Cocc=Cocc,
            Cvir=Cvir,
            eps_occ=eps_occ,
            eps_vir=eps_vir,
            tia=p,
            )

        pAp = cuest_scf.GPUMatrixUtility.ddot(
            x=p,
            y=Ap,
        )

        if pAp == 0.0:
            break

        alpha = rz_old / pAp

        # x = x + alpha p
        cuest_scf.GPUMatrixUtility.daxpy(
            alpha=alpha,
            x=p,
            y=x,
        )

        # r = r - alpha Ap
        cuest_scf.GPUMatrixUtility.daxpy(
            alpha=-alpha,
            x=Ap,
            y=r,
        )

        r_norm_sq = cuest_scf.GPUMatrixUtility.ddot(
            x=r,
            y=r,
        )
        r_norm = np.sqrt(r_norm_sq)

        print('%-4d:     %10.4e' % (iteration, r_norm))

        if r_norm <= tolerance:
            converged = True
            break

        # z = M^{-1} r
        kernel_helper.apply_preconditioner(
            nrhs=3,
            nocc=nocc,
            nvir=nvir,
            zia_ptr=z.pointer,
            ria_ptr=r.pointer,
            eps_occ_ptr=eps_occ.pointer,
            eps_vir_ptr=eps_vir.pointer,
            )

        rz_new = cuest_scf.GPUMatrixUtility.ddot(
            x=r,
            y=z,
        )

        if rz_old == 0.0:
            break

        beta = rz_new / rz_old

        # p = z + beta p
        cuest_scf.GPUMatrixUtility.dscal(
            alpha=beta,
            x=p,
        )
        cuest_scf.GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=z,
            y=p,
        )

        rz_old = rz_new

    if not converged:
        raise RuntimeError('CPHF iterations did not converge')

    print('\nCPHF iterations converged\n')

    # Form polarizability tensor

    alpha = cuest_scf.GPUMatrix(
        nrows=3,
        ncols=3,
    )

    cuest_scf.GPUMatrixUtility.c_dgemm(
       transa=False,
       transb=True,
       m=3,
       n=3,
       k=nocc * nvir,
       alpha=4.0,
       a=rhs,
       offa=0,
       lda=nocc * nvir,
       b=x,
       offb=0,
       ldb=nocc * nvir,
       beta=0.0,
       c=alpha,
       offc=0,
       ldc=3,
       )

    return alpha.to_numpy()


import os
filedir = os.path.dirname(os.path.realpath(__file__))   
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

# Usually you want one CuestHandle for many computations - e.g., do not spin up
# separate handles for every RHF instance
cuest_handle = cuest_scf.CuestHandle()

threshold_pq = 1.0E-18

molecule = cuest_scf.Molecule.parse_from_xyz_file('ethanol.xyz')
charge = 0

primary_name = 'def2-svp'
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
# CPHF example only supports HF
    xc_functional_name='HF',
    primary=primary,
    auxiliary=auxiliary,
    minao=minao,
    primary_name=primary_name,
    auxiliary_name=auxiliary_name,
    minao_name=minao_name,
    threshold_pq=threshold_pq,
    df_fitting_eigenvalue_cutoff=1.0e-12,
# CPHF example does not support PCM
    pcm_epsilon=1.0,
    g_convergence=1.0e-8,
    )

rhf.solve()

print('RHF Energy: %24.16E\n' % (rhf.compute_energy()))

alpha = rhf_polarizability(rhf=rhf)

print('RHF Polarizability Tensor (bohr^3):\n')
print(alpha)

