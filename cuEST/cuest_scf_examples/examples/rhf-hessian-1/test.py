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
# This is an example demonstrating how to use the overlap and kinetic Hessian
# skeleton and right-hand side (RHS) compute routines, together with the
# potential, Coulomb and exchange compute and gradient routines, to form a
# semi-analytic RHF nuclear Hessian by solving the CPHF equations. This
# example is limited to gas-phase, all-electron RHF Hessians with
# density-fitted Coulomb and exchange (fock_method "dfjk").
#
# Skeleton and RHS routines are currently available for the overlap and
# kinetic energy integrals, so those terms are analytic, as is the nuclear
# repulsion term. The potential and two-electron terms are formed by finite
# differences of gradients and Fock matrices at fixed density. Additional
# Hessian terms may appear in later cuEST releases. Any such term can replace
# its finite-difference counterpart, with skeleton and RHS calls like those
# for the overlap and kinetic terms.
#
# The perturbations are processed in blocks of atoms, as described in the
# Blocking Perturbations section of the cuEST documentation.
#
# This example is not intended to be performant. The finite differences
# build a new RHF object, with its basis sets and integral plans, at four
# displaced geometries for each nuclear coordinate. The example is provided
# only to show how to use the Hessian skeleton and RHS routines.
#
# This implementation uses the cuest_scf.RHF object to access compute
# routines, pre-built data structures and the results of the RHF calculation.
#
#

import os

import cuest.bindings as ce
import cuest_scf
import numpy as np

from cuest_hessian_int_helper import CuestHessianIntegralHelper
from kernel_helper import CPHFKernelHelper

filedir = os.path.dirname(os.path.realpath(__file__))
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

kernel_helper = CPHFKernelHelper()

def rhf_hessian_skeleton(
    *,
    rhf : cuest_scf.RHF,
    hessian_helper : CuestHessianIntegralHelper,
    perturbations_a : list,
    perturbations_b : list,
    D : cuest_scf.GPUMatrix,
    Cocc : cuest_scf.GPUMatrix,
    eps_occ_diag : cuest_scf.GPUMatrix,
    ):

    npert_a = 3 * len(perturbations_a)
    npert_b = 3 * len(perturbations_b)

    # Energy-weighted density matrix W_{mn} = \sum_{i} C_{im} \epsilon_{i} C_{in}
    epsCocc = cuest_scf.GPUMatrixUtility.matrix_multiply(
        mat1=eps_occ_diag,
        mat2=Cocc,
        transpose1=False,
        transpose2=False,
        )
    W = cuest_scf.GPUMatrixUtility.matrix_multiply(
        mat1=epsCocc,
        mat2=Cocc,
        transpose1=True,
        transpose2=False,
        )

    HT = cuest_scf.GPUMatrix(
        nrows=npert_a,
        ncols=npert_b,
        dtype=np.double,
        )
    HS = cuest_scf.GPUMatrix(
        nrows=npert_a,
        ncols=npert_b,
        dtype=np.double,
        )

    # One call forms the block for every pair of atoms from the two perturbation lists
    hessian_helper.compute_kinetic_hessian_skeleton(
        perturbations_a=perturbations_a,
        perturbations_b=perturbations_b,
        density=D,
        out_hessian=HT,
        )
    hessian_helper.compute_overlap_hessian_skeleton(
        perturbations_a=perturbations_a,
        perturbations_b=perturbations_b,
        density=W,
        out_hessian=HS,
        )

    # Form the skeleton contribution H = 2 H_T - 2 H_S. The block is rectangular, so
    # scale and add rather than use dgeam, which takes only square matrices
    H = HT.clone()
    cuest_scf.GPUMatrixUtility.dscal(
        alpha=2.0,
        x=H,
        )
    cuest_scf.GPUMatrixUtility.daxpy(
        alpha=-2.0,
        x=HS,
        y=H,
        )

    # Add the nuclear repulsion contribution, H += H_nuc
    Hnuc = rhf_nuclear_repulsion_hessian(
        rhf=rhf,
        perturbations_a=perturbations_a,
        perturbations_b=perturbations_b,
        )
    cuest_scf.GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=Hnuc,
        y=H,
        )

    return H


def rhf_nuclear_repulsion_hessian(
    *,
    rhf : cuest_scf.RHF,
    perturbations_a : list,
    perturbations_b : list,
    ):

    xyz = rhf.molecule.xyz
    Z = -1.0 * rhf.Zeff.reshape((rhf.molecule.natom,))
    natom = rhf.molecule.natom

    # Second derivatives of E_nuc = \sum_{A<B} Z_A Z_B / R_AB, with R = X_A - X_B:
    # d^2 E_nuc / dX_A dX_B = -Z_A Z_B (3 R R^T - R^2 I) / R^5 for A != B
    def pair_block(A, B):
        R = xyz[A, :] - xyz[B, :]
        rAB = np.sqrt(np.sum(R**2))
        return -Z[A] * Z[B] * (3.0 * np.outer(R, R) - rAB**2 * np.eye(3)) / rAB**5

    # Each diagonal block is minus the sum of the off-diagonal blocks over all other
    # atoms, including atoms not in the perturbation lists
    Hnuc = np.zeros((3 * len(perturbations_a), 3 * len(perturbations_b)))
    for a, A in enumerate(perturbations_a):
        for b, B in enumerate(perturbations_b):
            if A != B:
                HAB = pair_block(A, B)
            else:
                HAB = -sum(pair_block(A, C) for C in range(natom) if C != A)
            Hnuc[3 * a:3 * a + 3, 3 * b:3 * b + 3] = HAB
    return cuest_scf.GPUMatrix.from_numpy(Hnuc)


def rhf_hessian_rhs(
    *,
    rhf : cuest_scf.RHF,
    hessian_helper : CuestHessianIntegralHelper,
    perturbations : list,
    C : cuest_scf.GPUMatrix,
    Cocc : cuest_scf.GPUMatrix,
    Cvir : cuest_scf.GPUMatrix,
    eps_occ_diag : cuest_scf.GPUMatrix,
    Foo_fd : cuest_scf.GPUMatrix,
    Fov_fd : cuest_scf.GPUMatrix,
    ):

    # Array sizing
    nocc = Cocc.shape[0]
    nvir = Cvir.shape[0]
    nao = Cocc.shape[1]
    nmo = nocc + nvir
    npert = 3 * len(perturbations)

    # Coefficient matrices for the overlap pairs (Cocc, Cocc), (eps Cocc, Cocc),
    # (eps Cocc, Cvir) and the kinetic pairs (Cocc, Cocc), (Cocc, Cvir)
    epsCocc = cuest_scf.GPUMatrixUtility.matrix_multiply(
        mat1=eps_occ_diag,
        mat2=Cocc,
        transpose1=False,
        transpose2=False,
        )
    S_left = cuest_scf.GPUMatrix(
        nrows=3 * nocc,
        ncols=nao,
        dtype=np.double,
        )
    S_right = cuest_scf.GPUMatrix(
        nrows=nocc + nmo,
        ncols=nao,
        dtype=np.double,
        )
    T_left = cuest_scf.GPUMatrix(
        nrows=2 * nocc,
        ncols=nao,
        dtype=np.double,
        )

    # Overlap left matrices [Cocc; eps Cocc; eps Cocc]
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc * nao,
        x=Cocc,
        offx=0,
        incx=1,
        y=S_left,
        offy=0,
        incy=1,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc * nao,
        x=epsCocc,
        offx=0,
        incx=1,
        y=S_left,
        offy=nocc * nao,
        incy=1,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc * nao,
        x=epsCocc,
        offx=0,
        incx=1,
        y=S_left,
        offy=2 * nocc * nao,
        incy=1,
        )

    # Overlap right matrices [Cocc; C]
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc * nao,
        x=Cocc,
        offx=0,
        incx=1,
        y=S_right,
        offy=0,
        incy=1,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nmo * nao,
        x=C,
        offx=0,
        incx=1,
        y=S_right,
        offy=nocc * nao,
        incy=1,
        )

    # Kinetic left matrices [Cocc; Cocc]
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc * nao,
        x=Cocc,
        offx=0,
        incx=1,
        y=T_left,
        offy=0,
        incy=1,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc * nao,
        x=Cocc,
        offx=0,
        incx=1,
        y=T_left,
        offy=nocc * nao,
        incy=1,
        )

    # Overlap derivative integrals S^x_oo, eps S^x_oo and eps S^x_ov
    Sx = cuest_scf.GPUMatrix(
        nrows=1,
        ncols=npert * (2 * nocc * nocc + nocc * nvir),
        dtype=np.double,
        )
    hessian_helper.compute_overlap_hessian_rhs(
        perturbations=perturbations,
        C_left=S_left,
        nmo_c1=[nocc, nocc, nocc],
        C_right=S_right,
        nmo_c2=[nocc, nocc, nvir],
        out_tensors=Sx,
        )

    # Kinetic derivative integrals T^x_oo and T^x_ov
    Tx = cuest_scf.GPUMatrix(
        nrows=1,
        ncols=npert * (nocc * nocc + nocc * nvir),
        dtype=np.double,
        )
    hessian_helper.compute_kinetic_hessian_rhs(
        perturbations=perturbations,
        C_left=T_left,
        nmo_c1=[nocc, nocc],
        C_right=C,
        nmo_c2=[nocc, nvir],
        out_tensors=Tx,
        )

    # Split the derivative integrals into separate matrices
    Soo = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nocc,
        dtype=np.double,
        )
    epsSoo = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nocc,
        dtype=np.double,
        )
    epsSov = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nvir,
        dtype=np.double,
        )
    Foo = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nocc,
        dtype=np.double,
        )
    Fov = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nvir,
        dtype=np.double,
        )

    offset = 0
    for M in [Soo, epsSoo, epsSov]:
        cuest_scf.GPUMatrixUtility.dcopy(
            n=M.size,
            x=Sx,
            offx=offset,
            incx=1,
            y=M,
            offy=0,
            incy=1,
            )
        offset += M.size

    offset = 0
    for M in [Foo, Fov]:
        cuest_scf.GPUMatrixUtility.dcopy(
            n=M.size,
            x=Tx,
            offx=offset,
            incx=1,
            y=M,
            offy=0,
            incy=1,
            )
        offset += M.size

    # Add the finite-difference potential and two-electron terms to the derivative Fock matrix
    cuest_scf.GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=Foo_fd,
        y=Foo,
        )
    cuest_scf.GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=Fov_fd,
        y=Fov,
        )

    # Factor pseudodensity P_{mn} = \sum_{ij} C_{im} S_{ij} C_{jn}
    # As: P_{mn} = \sum_{i} C_{im} R_{in} = \sum_{i} C_{im} ( \sum_{j} S_{ij} C_{jn} )
    R = []
    for n in range(npert):
        R.append(cuest_scf.GPUMatrix(
            nrows=nocc,
            ncols=nao,
            dtype=np.double,
            ))
        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=False,
            m=nocc,
            n=nao,
            k=nocc,
            alpha=1.0,
            a=Soo,
            offa=n * nocc * nocc,
            lda=nocc,
            b=Cocc,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=R[n],
            offc=0,
            ldc=nao,
            )

    # Compute all exchange matrices in one shot
    K = rhf.compute_nonsymmetric_exchange(
        Cocc_left=Cocc,
        Coccs_right=R,
        )

    # Allocate space for the transformed Fock response
    Qov = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nvir,
        dtype=np.double,
        )
    Goo = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nocc,
        dtype=np.double,
        )

    # Temporary space for the AO to MO transformation
    Gin = cuest_scf.GPUMatrix(
        nrows=nocc,
        ncols=nao,
        dtype=np.double,
        )

    # Temporary space for the pseudodensity
    Pmn = cuest_scf.GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )

    for n in range(npert):

        # Compute Coulomb matrices one at a time
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
            b=R[n],
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Pmn,
            offc=0,
            ldc=nao,
            )
        J = rhf.compute_coulomb(
            D=Pmn,
            )

        # G = 2 J - K
        G = cuest_scf.GPUMatrixUtility.dgeam(
            transpose1=False,
            alpha=2.0,
            mat1=J,
            transpose2=False,
            beta=-1.0,
            mat2=K[n],
            )

        # AO to MO transformation of the Fock response
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
            b=G,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Gin,
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
            a=Gin,
            offa=0,
            lda=nao,
            b=Cvir,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Qov,
            offc=n * nocc * nvir,
            ldc=nvir,
            )
        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=True,
            m=nocc,
            n=nocc,
            k=nao,
            alpha=1.0,
            a=Gin,
            offa=0,
            lda=nao,
            b=Cocc,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=Goo,
            offc=n * nocc * nocc,
            ldc=nocc,
            )

    # Form the right-hand side of the CPHF equations, B = F - eps S - Q, which is -R^A in
    # the cuEST documentation, so the CPHF solution is x = -U^A
    B = Fov.clone()
    cuest_scf.GPUMatrixUtility.daxpy(
        alpha=-1.0,
        x=epsSov,
        y=B,
        )
    cuest_scf.GPUMatrixUtility.daxpy(
        alpha=-1.0,
        x=Qov,
        y=B,
        )

    return B, Soo, epsSoo, Foo, Goo


def rhf_mo_hessian_vector_product(
    *,
    rhf : cuest_scf.RHF,
    Cocc : cuest_scf.GPUMatrix,
    Cvir : cuest_scf.GPUMatrix,
    eps_occ : cuest_scf.GPUMatrix,
    eps_vir : cuest_scf.GPUMatrix,
    tia : cuest_scf.GPUMatrix,
    ):

    nrhs = tia.shape[0]
    nocc = Cocc.shape[0]
    nvir = Cvir.shape[0]
    nao = Cocc.shape[1]

    # Factor pseudodensity X_{mn} = \sum_{ia} C_{im} t_{ia} C_{an}
    # As: X_{mn} = \sum_{i} C_{im} T_{in} = \sum_{i} C_{im} ( \sum_{a} t_{ia} C_{an} )
    Tocc = []
    for n in range(nrhs):
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

    # Compute all exchange matrices in one shot
    K = rhf.compute_nonsymmetric_exchange(
        Cocc_left=Cocc,
        Coccs_right=Tocc,
        )

    # Allocate space for the MO Hessian - vector product
    Ria = cuest_scf.GPUMatrix(
        nrows=nrhs,
        ncols=nocc * nvir,
        dtype=np.double,
        )

    # Temporary space for the AO to MO transformation
    Ain = cuest_scf.GPUMatrix(
        nrows=nocc,
        ncols=nao,
        dtype=np.double,
        )

    # Temporary space for the pseudodensity
    Xmn = cuest_scf.GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )

    # Assemble MO Hessian - vector products, computing Coulomb matrices one at a time
    for n in range(nrhs):
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
        J = rhf.compute_coulomb(
            D=Xmn,
            )
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
            x=J,
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
        nrhs=nrhs,
        nocc=nocc,
        nvir=nvir,
        ria_ptr=Ria.pointer,
        tia_ptr=tia.pointer,
        eps_occ_ptr=eps_occ.pointer,
        eps_vir_ptr=eps_vir.pointer,
        )

    return Ria


def rhf_solve_cphf(
    *,
    rhf : cuest_scf.RHF,
    Cocc : cuest_scf.GPUMatrix,
    Cvir : cuest_scf.GPUMatrix,
    eps_occ : cuest_scf.GPUMatrix,
    eps_vir : cuest_scf.GPUMatrix,
    rhs : cuest_scf.GPUMatrix,
    maxiter : int = 100,
    tolerance : float = 1.0e-8,
    ):

    nrhs = rhs.nrows
    nocc = Cocc.shape[0]
    nvir = Cvir.shape[0]

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
        nrhs=nrhs,
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
            nrhs=nrhs,
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

    return x


def build_displaced_rhf(
    *,
    rhf : cuest_scf.RHF,
    xyz : np.ndarray,
    ):

    # Displaced molecule, coordinates in bohr
    molecule = cuest_scf.Molecule(
        N=rhf.molecule.N,
        Z=rhf.molecule.Z,
        xyz=xyz,
        )

    def gbs(name):
        return cuest_scf.AOBasis.parse_from_gbs_file(
            os.path.join(basisdir, '%s.gbs' % name),
            molecule=molecule,
            )

    # Same basis sets and thresholds as the reference. The displaced object is never solved.
    return cuest_scf.RHF(
        cuest_handle=rhf.cuest_handle,
        molecule=molecule,
        charge=rhf.charge,
        xc_functional_name='HF',
        primary=gbs(rhf.primary_name),
        auxiliary=gbs(rhf.auxiliary_name),
        minao=gbs(rhf.minao_name),
        primary_name=rhf.primary_name,
        auxiliary_name=rhf.auxiliary_name,
        minao_name=rhf.minao_name,
        threshold_pq=rhf.threshold_pq,
        df_fitting_eigenvalue_cutoff=rhf.df_fitting_eigenvalue_cutoff,
        pcm_epsilon=1.0,
        )


def rhf_finite_difference_terms(
    *,
    rhf : cuest_scf.RHF,
    perturbations_a : list,
    perturbations_b : list,
    D : cuest_scf.GPUMatrix,
    Cocc : cuest_scf.GPUMatrix,
    Cvir : cuest_scf.GPUMatrix,
    delta : float = 1.0e-3,
    ):

    # Array sizing
    nocc = Cocc.shape[0]
    nvir = Cvir.shape[0]
    nao = Cocc.shape[1]
    npert_a = 3 * len(perturbations_a)
    npert_b = 3 * len(perturbations_b)

    # Allocate space for the finite-difference Hessian block of the displaced atoms in
    # perturbations_a and the gradient components of perturbations_b, and for the
    # derivative Fock matrix terms of perturbations_a
    Hfd = cuest_scf.GPUMatrix(
        nrows=npert_a,
        ncols=npert_b,
        dtype=np.double,
        )
    Foo_fd = cuest_scf.GPUMatrix(
        nrows=npert_a,
        ncols=nocc * nocc,
        dtype=np.double,
        )
    Fov_fd = cuest_scf.GPUMatrix(
        nrows=npert_a,
        ncols=nocc * nvir,
        dtype=np.double,
        )

    # Temporary space for the derivative Fock matrix and its AO to MO transformation
    Fx = cuest_scf.GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )
    Fin = cuest_scf.GPUMatrix(
        nrows=nocc,
        ncols=nao,
        dtype=np.double,
        )

    xyz0 = np.array(rhf.molecule.xyz)

    stencil = list(zip(
        np.array([1.0, -8.0, 8.0, -1.0]) / (12.0 * delta),
        delta * np.array([-2.0, -1.0, 1.0, 2.0]),
        ))

    print('Finite difference terms (4 point central, %.1e bohr step)' % delta)
    print('')
    for a, A in enumerate(perturbations_a):
        for direction in range(3):
            n = 3 * a + direction
            Fx.zero()
            for weight, displacement in stencil:
                xyz = xyz0.copy()
                xyz[A, direction] += displacement
                rhf_displaced = build_displaced_rhf(
                    rhf=rhf,
                    xyz=xyz,
                    )

                # Potential and two-electron Fock matrix at fixed D, V + 2 J - K
                V = rhf_displaced.compute_potential()
                J = rhf_displaced.compute_coulomb(
                    D=D,
                    )
                # DF-K int8 slice and modulus counts (used only by fock_method "dfjk")
                K, = rhf_displaced.compute_exchange(
                    Coccs=[Cocc],
                    dfk_int8_slice_count=6,
                    dfk_int8_modulus_count=9,
                    )
                cuest_scf.GPUMatrixUtility.daxpy(
                    alpha=weight,
                    x=V,
                    y=Fx,
                    )
                cuest_scf.GPUMatrixUtility.daxpy(
                    alpha=2.0 * weight,
                    x=J,
                    y=Fx,
                    )
                cuest_scf.GPUMatrixUtility.daxpy(
                    alpha=-weight,
                    x=K,
                    y=Fx,
                    )

                # Gradient at fixed D, the last call since it overwrites the DF integral plan
                GV = rhf_displaced.compute_potential_gradient(D)
                Gfd = rhf_displaced.compute_coulomb_and_exchange_gradient(
                    scaleJ=2.0,
                    DJ=D,
                    scaleK=-1.0,
                    CoccsK=[Cocc],
                    )
                cuest_scf.GPUMatrixUtility.daxpy(
                    alpha=2.0,
                    x=GV,
                    y=Gfd,
                    )
                # Keep the gradient components of the atoms in perturbations_b
                for b, B in enumerate(perturbations_b):
                    cuest_scf.GPUMatrixUtility.daxpy(
                        n=3,
                        alpha=weight,
                        x=Gfd,
                        offx=3 * B,
                        incx=1,
                        y=Hfd,
                        offy=n * npert_b + 3 * b,
                        incy=1,
                        )

            # AO to MO transformation of the derivative Fock matrix
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
                b=Fx,
                offb=0,
                ldb=nao,
                beta=0.0,
                c=Fin,
                offc=0,
                ldc=nao,
                )
            cuest_scf.GPUMatrixUtility.c_dgemm(
                transa=False,
                transb=True,
                m=nocc,
                n=nocc,
                k=nao,
                alpha=1.0,
                a=Fin,
                offa=0,
                lda=nao,
                b=Cocc,
                offb=0,
                ldb=nao,
                beta=0.0,
                c=Foo_fd,
                offc=n * nocc * nocc,
                ldc=nocc,
                )
            cuest_scf.GPUMatrixUtility.c_dgemm(
                transa=False,
                transb=True,
                m=nocc,
                n=nvir,
                k=nao,
                alpha=1.0,
                a=Fin,
                offa=0,
                lda=nao,
                b=Cvir,
                offb=0,
                ldb=nao,
                beta=0.0,
                c=Fov_fd,
                offc=n * nocc * nvir,
                ldc=nvir,
                )

    return Hfd, Foo_fd, Fov_fd


def rhf_hessian(
    *,
    rhf : cuest_scf.RHF,
    block_size : int,
    maxiter : int = 100,
    tolerance : float = 1.0e-8,
    ):

    # Grab some array sizes from the cuest_scf.RHF object
    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']
    nao = rhf.tensors['C'].shape[1]

    # Perturb every atom, in blocks of block_size consecutive atoms. Each block is a
    # sorted list without duplicates, as the Hessian routines require, and its rows
    # are contiguous in the Hessian
    atoms = list(range(rhf.molecule.natom))
    blocks = [atoms[start:start + block_size] for start in range(0, len(atoms), block_size)]
    npert = 3 * len(atoms)

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

    # Diagonal matrix of occupied orbital energies
    eps_occ_diag = cuest_scf.GPUMatrix(
        nrows=nocc,
        ncols=nocc,
        dtype=np.double,
        )
    cuest_scf.GPUMatrixUtility.dcopy(
        n=nocc,
        x=rhf.tensors['eps'],
        offx=0,
        incx=1,
        y=eps_occ_diag,
        offy=0,
        incy=nocc + 1,
        )

    # Hessian integrals use the one-electron integral plan of the RHF object
    hessian_helper = CuestHessianIntegralHelper(
        cuest_handle=rhf.cuest_handle,
        oe_int_plan=rhf.cuest_oe_int_plan,
        )

    # Allocate space for the Hessian, and for the right-hand sides, CPHF solutions and
    # MO basis matrices of every block, which the response contribution needs for
    # every pair of blocks
    H = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=npert,
        dtype=np.double,
        )
    B = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nvir,
        dtype=np.double,
        )
    x = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nvir,
        dtype=np.double,
        )
    Soo = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nocc,
        dtype=np.double,
        )
    epsSoo = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nocc,
        dtype=np.double,
        )
    Foo = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nocc,
        dtype=np.double,
        )
    Goo = cuest_scf.GPUMatrix(
        nrows=npert,
        ncols=nocc * nocc,
        dtype=np.double,
        )

    # Process the perturbations one block of atoms at a time
    for block in blocks:

        print('Perturbation block: atoms %d to %d' % (block[0], block[-1]))
        print('')

        # First row of this block in the Hessian and in the matrices kept for the
        # response contribution
        row = 3 * block[0]

        # Form the potential and two-electron terms by finite differences, for the
        # rows of this block and the columns of every atom
        Hfd, Foo_fd, Fov_fd = rhf_finite_difference_terms(
            rhf=rhf,
            perturbations_a=block,
            perturbations_b=atoms,
            D=rhf.tensors['D'],
            Cocc=Cocc,
            Cvir=Cvir,
            )

        # Form the skeleton contribution, for the rows of this block and the columns
        # of every atom
        Hskel = rhf_hessian_skeleton(
            rhf=rhf,
            hessian_helper=hessian_helper,
            perturbations_a=block,
            perturbations_b=atoms,
            D=rhf.tensors['D'],
            Cocc=Cocc,
            eps_occ_diag=eps_occ_diag,
            )

        # Add the skeleton and finite-difference terms to the rows of this block
        for M in [Hskel, Hfd]:
            cuest_scf.GPUMatrixUtility.daxpy(
                n=M.size,
                alpha=1.0,
                x=M,
                offx=0,
                incx=1,
                y=H,
                offy=row * npert,
                incy=1,
                )

        # Form the right-hand sides of the CPHF equations for this block
        B_block, Soo_block, epsSoo_block, Foo_block, Goo_block = rhf_hessian_rhs(
            rhf=rhf,
            hessian_helper=hessian_helper,
            perturbations=block,
            C=rhf.tensors['C'],
            Cocc=Cocc,
            Cvir=Cvir,
            eps_occ_diag=eps_occ_diag,
            Foo_fd=Foo_fd,
            Fov_fd=Fov_fd,
            )

        # Solve the CPHF equations for this block
        x_block = rhf_solve_cphf(
            rhf=rhf,
            Cocc=Cocc,
            Cvir=Cvir,
            eps_occ=eps_occ,
            eps_vir=eps_vir,
            rhs=B_block,
            maxiter=maxiter,
            tolerance=tolerance,
            )

        # Keep the matrices of this block for the response contribution
        for M_block, M in [
            (B_block, B),
            (x_block, x),
            (Soo_block, Soo),
            (epsSoo_block, epsSoo),
            (Foo_block, Foo),
            (Goo_block, Goo),
            ]:
            cuest_scf.GPUMatrixUtility.dcopy(
                n=M_block.size,
                x=M_block,
                offx=0,
                incx=1,
                y=M,
                offy=row * M.ncols,
                incy=1,
                )

    # Add the response contribution for every pair of blocks at once, H += -4 B x^T
    cuest_scf.GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=True,
        m=npert,
        n=npert,
        k=nocc * nvir,
        alpha=-4.0,
        a=B,
        offa=0,
        lda=nocc * nvir,
        b=x,
        offb=0,
        ldb=nocc * nvir,
        beta=1.0,
        c=H,
        offc=0,
        ldc=npert,
        )

    # H += -2 S_oo F_oo^T
    cuest_scf.GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=True,
        m=npert,
        n=npert,
        k=nocc * nocc,
        alpha=-2.0,
        a=Soo,
        offa=0,
        lda=nocc * nocc,
        b=Foo,
        offb=0,
        ldb=nocc * nocc,
        beta=1.0,
        c=H,
        offc=0,
        ldc=npert,
        )

    # H += -2 F_oo S_oo^T
    cuest_scf.GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=True,
        m=npert,
        n=npert,
        k=nocc * nocc,
        alpha=-2.0,
        a=Foo,
        offa=0,
        lda=nocc * nocc,
        b=Soo,
        offb=0,
        ldb=nocc * nocc,
        beta=1.0,
        c=H,
        offc=0,
        ldc=npert,
        )

    # H += 4 (eps S_oo) S_oo^T
    cuest_scf.GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=True,
        m=npert,
        n=npert,
        k=nocc * nocc,
        alpha=4.0,
        a=epsSoo,
        offa=0,
        lda=nocc * nocc,
        b=Soo,
        offb=0,
        ldb=nocc * nocc,
        beta=1.0,
        c=H,
        offc=0,
        ldc=npert,
        )

    # H += 2 S_oo G_oo^T
    cuest_scf.GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=True,
        m=npert,
        n=npert,
        k=nocc * nocc,
        alpha=2.0,
        a=Soo,
        offa=0,
        lda=nocc * nocc,
        b=Goo,
        offb=0,
        ldb=nocc * nocc,
        beta=1.0,
        c=H,
        offc=0,
        ldc=npert,
        )

    return H.to_numpy()


# Usually you want one CuestHandle for many computations - e.g., do not spin up
# separate handles for every RHF instance
cuest_handle = cuest_scf.CuestHandle()

# This will set the cuEST handle so all operations are performed with FP64 arithmetic
# cuest_handle.set_math_mode(mode=ce.CuestMathMode.CUEST_NATIVE_FP64_MATH_MODE)

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
# Hessian example only supports HF
    xc_functional_name='HF',
    primary=primary,
    auxiliary=auxiliary,
    minao=minao,
    primary_name=primary_name,
    auxiliary_name=auxiliary_name,
    minao_name=minao_name,
    threshold_pq=threshold_pq,
    df_fitting_eigenvalue_cutoff=1.0e-12,
# Hessian example does not support PCM
    pcm_epsilon=1.0,
    g_convergence=1.0e-8,
    )

rhf.solve()

print('RHF Energy: %24.16E\n' % (rhf.compute_energy()))

# rhf.compute_gradient() overwrites the DF integral plan, so call it after the Hessian
# Blocks of 4 atoms show the blocking on this small molecule
hessian = rhf_hessian(
    rhf=rhf,
    block_size=4,
    )

# Rows and columns are ordered 3 * atom + {x, y, z}
print('Semi-Analytic RHF Hessian (Hartree/Bohr^2):\n')
print(hessian)
