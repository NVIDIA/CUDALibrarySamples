#!/usr/bin/env python
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
# Analytic density-fitted MP2 nuclear gradient built on cuest_scf.  Equation
# numbers correspond to the implementation described in dx.doi.org/10.1007/s00214-003-0453-3.
# This implementation generally follows those equations, but loops over occupied indices
# one-at-a-time to avoid having to build and store the full ijab tensors in DRAM.
#

import os
import sys

import numpy as np

import cuest_scf
from cuest_scf import GPUMatrix, GPUMatrixUtility
from cuest_scf import CuestDFIntCompute, CuestOEIntCompute, CuestOEIntPlan
from cuest_scf import CuestECPIntCompute, CuestECPIntPlan
from cuest_scf.cuest_ecp_basis import CuestECPBasis
from cuest_scf import Timer

from cuest_mo_deriv_int_helper import CuestMOIntegralHelper
from kernel_helper import MP2KernelHelper

filedir = os.path.dirname(os.path.realpath(__file__))
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

# => Reference wavefunction <= #

def build_rhf(
    cuest_handle,
    molecule,
    *,
    primary_name,
    auxiliary_name,
    minao_name,
    threshold_pq,
    ecp_name=None,
    print_level=2,
    ):

    def gbs(name):
        return cuest_scf.AOBasis.parse_from_gbs_file(
            os.path.join(basisdir, '%s.gbs' % name),
            molecule=molecule,
            )

    ecp_basis = None
    if ecp_name is not None:
        ecp_basis = cuest_scf.ECPBasis.parse_from_gbs_file(
            os.path.join(basisdir, '%s.gbs' % ecp_name),
            molecule=molecule,
            )

    rhf = cuest_scf.RHF(
        print_level=print_level,
        cuest_handle=cuest_handle,
        molecule=molecule,
        charge=0,
        xc_functional_name='HF',
        primary=gbs(primary_name),
        ecp_basis=ecp_basis,
        ecp_basis_name=ecp_name,
        auxiliary=gbs(auxiliary_name),
        minao=gbs(minao_name),
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        print_timings=False,
        threshold_pq=threshold_pq,
        g_convergence=1e-8,
        dfk_int8_modulus_count_start=8,
        dfk_int8_modulus_count_end=10,
        dfk_int8_slice_count_start=5,
        dfk_int8_slice_count_end=7,
        )
    rhf.solve()
    return rhf


# => Fitted MO integrals over the full MO space <= #

def compute_B(
    helper,
    naux,
    C_left,
    nmo_c1,
    c1_offset,
    C_right,
    nmo_c2,
    c2_offset,
    ):
    """B[A,p,q] = [J^-1/2]_AB (B|pq) over one or more pairs of MO subspaces."""

    num_left = [nmo_c1] if isinstance(nmo_c1, int) else list(nmo_c1)
    num_right = [nmo_c2] if isinstance(nmo_c2, int) else list(nmo_c2)

    B = GPUMatrix(
        nrows=naux * sum(l * r for l, r in zip(num_left, num_right)),
        ncols=1,
        dtype=np.double,
        )
    helper.compute_3c_integral(
        C_left=C_left,
        nmo_c1=nmo_c1,
        c1_offset=c1_offset,
        C_right=C_right,
        nmo_c2=nmo_c2,
        c2_offset=c2_offset,
        out_mo_integrals=B,
        )
    return B


def concatenate_orbitals(
    C,
    blocks,
    nao,
    ):
    """Stack MO subspaces of C, each given as (offset, count), into one buffer."""

    out = GPUMatrix(
        nrows=sum(count for _, count in blocks),
        ncols=nao,
        dtype=np.double,
        )
    offset = 0
    for start, count in blocks:
        GPUMatrixUtility.dcopy(
            n=count * nao,
            x=C,
            offx=start * nao,
            y=out,
            offy=offset,
            )
        offset += count * nao
    return out


def get_kernels():
    """The JIT-compiled kernels, built once and shared by every geometry."""

    if get_kernels.kernels is None:
        get_kernels.kernels = MP2KernelHelper()
    return get_kernels.kernels

get_kernels.kernels = None


def offset_pointer(matrix, offset):
    """Device address of element `offset` of a GPUMatrix, for kernels taking raw pointers."""

    return int(matrix.pointer) + offset * matrix.type_size




# => One-electron integrals owned by the example <= #

class OneElectronHelper(object):
    """The one-electron plan and nuclear charges the gradient needs once the RHF object is gone.
    We avoid using the RHF object's equivalent plan directly here, because we want to free the
    relatively large DFIntPlan held by RHF.
    """

    def __init__(
        self,
        helper,
        molecule,
        ecp_basis=None,
        ):

        self.cuest_handle = helper.cuest_handle
        self.natom = molecule.natom
        self.xyz = molecule.xyz

        # The nuclei the electrons see: the bare charges less any electrons an ECP replaces.
        self.do_ecp = ecp_basis is not None and ecp_basis.is_active
        nelectron = np.zeros(molecule.natom)
        if self.do_ecp:
            nelectron = np.array(
                [atom.nelectron for atom in ecp_basis.atoms],
                dtype=np.float64,
                )
        self.Z = molecule.Z - nelectron
        self.oe_int_plan = CuestOEIntPlan(
            handle=helper.cuest_handle,
            basis=helper.cuest_primary,
            ao_pair_list=helper.cuest_ao_pair_list,
            )

        self.gpu_xyz = GPUMatrix.from_numpy(molecule.xyz)
        # The potential integrals want the charges negated.
        self.gpu_Z = GPUMatrix.from_numpy(
            self.Z.reshape(-1, 1).astype(np.float64),
            )
        GPUMatrixUtility.dscal(
            alpha=-1.0,
            x=self.gpu_Z,
            )

        if self.do_ecp:
            self.cuest_ecp_basis = CuestECPBasis(
                handle=helper.cuest_handle,
                ecp_basis=ecp_basis,
                )
            self.ecp_int_plan = CuestECPIntPlan(
                handle=helper.cuest_handle,
                basis=helper.cuest_primary,
                xyzs=molecule.xyz,
                numECPAtoms=self.cuest_ecp_basis.num_active_ecp,
                activeIndices=self.cuest_ecp_basis.ecp_indices,
                activeAtoms=self.cuest_ecp_basis.ecp_atoms,
                )

    def new_gradient(self):
        return GPUMatrix(
            nrows=self.natom,
            ncols=3,
            dtype=np.double,
            )

    def nuclear_repulsion_gradient(self):
        """dEnuc/dR. Classical, natom^2, and cheap enough to carry out directly in numpy.
        Uses the effective nuclear charges, so ECP-replaced electrons are left out.
        """

        G = np.zeros((self.natom, 3))
        for a in range(self.natom):
            for b in range(self.natom):
                if a == b:
                    continue
                r_ab = self.xyz[a, :] - self.xyz[b, :]
                G[a, :] -= self.Z[a] * self.Z[b] * r_ab / np.sum(r_ab ** 2) ** 1.5
        return GPUMatrix.from_numpy(G)

    def kinetic_gradient(
        self,
        D,
        ):

        G = self.new_gradient()
        CuestOEIntCompute.kinetic_gradient(
            handle=self.cuest_handle,
            oe_int_plan=self.oe_int_plan,
            Gptr=G.pointer,
            Dptr=D.pointer,
            )
        return G

    def potential_gradient(
        self,
        D,
        ):

        # The basis and charge derivatives come back separately and are simply summed.
        G_basis = self.new_gradient()
        G_charge = self.new_gradient()
        CuestOEIntCompute.potential_gradient(
            handle=self.cuest_handle,
            oe_int_plan=self.oe_int_plan,
            ncharge=self.natom,
            xyz=self.gpu_xyz.pointer,
            q=self.gpu_Z.pointer,
            Dptr=D.pointer,
            GptrBasis=G_basis.pointer,
            GptrCharge=G_charge.pointer,
            )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=G_charge,
            y=G_basis,
            )
        return G_basis

    def ecp_gradient(
        self,
        D,
        ):

        G = self.new_gradient()
        CuestECPIntCompute.potential_gradient(
            handle=self.cuest_handle,
            ecp_int_plan=self.ecp_int_plan,
            Dptr=D.pointer,
            Gptr=G.pointer,
            )
        return G

    def overlap_gradient(
        self,
        W,
        ):

        G = self.new_gradient()
        CuestOEIntCompute.overlap_gradient(
            handle=self.cuest_handle,
            oe_int_plan=self.oe_int_plan,
            Gptr=G.pointer,
            Wptr=W.pointer,
            )
        return G


# => Amplitudes and the correlation density <= #

def compute_correlation_intermediates(
    kernels,
    B,
    ov_offset,
    eps,
    naux,
    nocc,
    nvir,
    *,
    nfrz=0,
    energy_only=False,
    ):
    """Everything the gradient needs from the MP2 amplitudes, in a single pass,

    The amplitudes t and T = 2t - t' are o^2v^2, by far the largest tensors
    here, so they are never held in core: only the nact pair blocks sharing the
    current occupied index i are stored, as a (nact * nvir, nvir) buffer
    indexed [j, a, b].

    The first nfrz occupied orbitals are frozen: i and j run over the active
    orbitals only and t and T are stored active-relative. B keeps its
    full-occupied layout, so its offsets use the absolute index. g3_ov and P_ij
    are active-only, like t and T.
    """

    nact = nocc - nfrz
    pair_size = nvir * nvir

    t = GPUMatrix(
        nrows=nact * nvir,
        ncols=nvir,
        dtype=np.double,
        )
    T = GPUMatrix(
        nrows=nact * nvir,
        ncols=nvir,
        dtype=np.double,
        )
    pair_mo_ints = GPUMatrix(
        nrows=nvir,
        ncols=nvir,
        dtype=np.double,
        )

    P_ij = None
    P_ab = None
    g3_ov = None
    if not energy_only:
        P_ij = GPUMatrix(
            nrows=nact,
            ncols=nact,
            dtype=np.double,
            )
        P_ab = GPUMatrix(
            nrows=nvir,
            ncols=nvir,
            dtype=np.double,
            )
        g3_ov = GPUMatrix(
            nrows=naux * nact,
            ncols=nvir,
            dtype=np.double,
            )

    e_corr = 0.0
    for i in range(nfrz, nocc):
        for j in range(nfrz, nocc):
            offset = (j - nfrz) * pair_size

            # eq. (144)
            # (ia|jb) = sum_A B[A,i,a] B[A,j,b]
            GPUMatrixUtility.c_dgemm(
                transa=True,
                transb=False,
                m=nvir,
                n=nvir,
                k=naux,
                alpha=1.0,
                a=B,
                offa=ov_offset + i * nvir,
                lda=nocc * nvir,
                b=B,
                offb=ov_offset + j * nvir,
                ldb=nocc * nvir,
                beta=0.0,
                c=pair_mo_ints,
                offc=0,
                ldc=nvir,
                )

            # eq. (144)
            kernels.make_pair_amplitudes(
                i=i,
                j=j,
                nocc=nocc,
                nvir=nvir,
                eps_ptr=eps.pointer,
                pair_mo_ints_ptr=pair_mo_ints.pointer,
                pair_amplitudes_ptr=offset_pointer(t, offset),
                )
            # eq. (144)
            kernels.make_pair_tau(
                nvir=nvir,
                pair_amplitudes_ptr=offset_pointer(t, offset),
                pair_tau_ptr=offset_pointer(T, offset),
                )

            # eq. (1)
            e_corr += GPUMatrixUtility.ddot(
                n=pair_size,
                x=pair_mo_ints,
                offx=0,
                incx=1,
                y=T,
                offy=offset,
                incy=1,
                )

            if energy_only:
                continue

            # eq. (171)
            # gamma3[A,i,a] = sum_jb B[A,j,b] T_ij[a,b]
            GPUMatrixUtility.c_dgemm(
                transa=False,
                transb=True,
                m=naux,
                n=nvir,
                k=nvir,
                alpha=1.0,
                a=B,
                offa=ov_offset + j * nvir,
                lda=nocc * nvir,
                b=T,
                offb=offset,
                ldb=nvir,
                beta=1.0,
                c=g3_ov,
                offc=(i - nfrz) * nvir,
                ldc=nact * nvir,
                )

        if energy_only:
            continue

        # eq. (177)
        GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=True,
            m=nact,
            n=nact,
            k=pair_size,
            alpha=-1.0,
            a=t,
            offa=0,
            lda=pair_size,
            b=T,
            offb=0,
            ldb=pair_size,
            beta=1.0,
            c=P_ij,
            offc=0,
            ldc=nact,
            )
        # eq. (178)
        GPUMatrixUtility.c_dgemm(
            transa=True,
            transb=False,
            m=nvir,
            n=nvir,
            k=nact * nvir,
            alpha=1.0,
            a=t,
            offa=0,
            lda=nvir,
            b=T,
            offb=0,
            ldb=nvir,
            beta=1.0,
            c=P_ab,
            offc=0,
            ldc=nvir,
            )

    return e_corr, P_ij, P_ab, g3_ov


def compute_lagrangian(
    B,
    oo_offset,
    ov_offset,
    vv_offset,
    g3_ov,
    naux,
    nocc,
    nvir,
    nfrz=0,
    ):
    """Z-vector right-hand side and the generalized Fock building blocks.

    The non-separable gamma3 is accumulated alongside the amplitudes that produce it, in
    compute_correlation_intermediates, and arrives here already formed. Every remaining
    piece is then a contraction of two fitted quantities, because the amplitude side of W[I]
    enters only through gamma3:
        W_I_ij[i,k] = -2 sum_Aa B[A,i,a] gamma3[A,k,a]
        W_I_ab[a,b] = -2 sum_Ai B[A,i,a] gamma3[A,i,b]
    which avoids ever touching the o^2v^2 amplitudes again.

    gamma3 has active rows only, so the second index of these contractions runs over the
    active orbitals; the outputs keep all nocc columns, with the frozen ones left at zero,
    by using an offset of nfrz.
    """

    nact = nocc - nfrz

    W_I_ij = GPUMatrix(
        nrows=nocc,
        ncols=nocc,
        dtype=np.double,
        )
    W_I_ab = GPUMatrix(
        nrows=nvir,
        ncols=nvir,
        dtype=np.double,
        )
    half_L_ai_4 = GPUMatrix(
        nrows=nvir,
        ncols=nocc,
        dtype=np.double,
        )
    half_W_I_ai = GPUMatrix(
        nrows=nvir,
        ncols=nocc,
        dtype=np.double,
        )
    for A in range(naux):
        # eq. (181)
        GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=True,
            m=nocc,
            n=nact,
            k=nvir,
            alpha=-2.0,
            a=B,
            offa=ov_offset + A * nocc * nvir,
            lda=nvir,
            b=g3_ov,
            offb=A * nact * nvir,
            ldb=nvir,
            beta=1.0,
            c=W_I_ij,
            offc=nfrz,
            ldc=nocc,
            )
        # eq. (159)
        GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=True,
            m=nvir,
            n=nact,
            k=nvir,
            alpha=2.0,
            a=B,
            offa=vv_offset + A * nvir * nvir,
            lda=nvir,
            b=g3_ov,
            offb=A * nact * nvir,
            ldb=nvir,
            beta=1.0,
            c=half_L_ai_4,
            offc=nfrz,
            ldc=nocc,
            )
        # eq. (183)
        GPUMatrixUtility.c_dgemm(
            transa=True,
            transb=False,
            m=nvir,
            n=nocc,
            k=nact,
            alpha=-2.0,
            a=g3_ov,
            offa=A * nact * nvir,
            lda=nvir,
            b=B,
            offb=oo_offset + A * nocc * nocc + nfrz * nocc,
            ldb=nocc,
            beta=1.0,
            c=half_W_I_ai,
            offc=0,
            ldc=nocc,
            )
        # eq. (182)
        GPUMatrixUtility.c_dgemm(
            transa=True,
            transb=False,
            m=nvir,
            n=nvir,
            k=nact,
            alpha=-2.0,
            a=B,
            offa=ov_offset + A * nocc * nvir + nfrz * nvir,
            lda=nvir,
            b=g3_ov,
            offb=A * nact * nvir,
            ldb=nvir,
            beta=1.0,
            c=W_I_ab,
            offc=0,
            ldc=nvir,
            )

    # eq. (159)
    half_L_ai_NS = half_L_ai_4.clone()
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=half_W_I_ai,
        y=half_L_ai_NS,
        )

    return half_L_ai_NS, W_I_ij, W_I_ab, half_W_I_ai


# => Fock response and the z-vector equations <= #

def compute_veff(
    helper,
    P_ao,
    ):
    """
    veff(P) = J(P) - 1/2 K(P) for a symmetric AO density

    Built on helper's own density-fitting plan rather than the RHF's. The two are the same
    JK-fit auxiliary over the same molecule, so driving the response off this one lets the
    reference wavefunction be released before the z-vector solve.
    """

    nao = P_ao.nrows

    P_sym = GPUMatrixUtility.dgeam(
        transpose1=False,
        alpha=0.5,
        mat1=P_ao,
        transpose2=True,
        beta=0.5,
        mat2=P_ao,
        )

    J = P_sym.clone()
    CuestDFIntCompute.coulomb(
        handle=helper.cuest_handle,
        df_int_plan=helper.df_int_plan,
        Jptr=J.pointer,
        Dptr=P_sym.pointer,
        )

    evals, evecs = GPUMatrixUtility.eigh(matrix=P_sym)
    right = GPUMatrixUtility.dgeam(
        transpose1=True,
        alpha=1.0,
        mat1=evecs,
        transpose2=False,
        beta=0.0,
        mat2=evecs,
        )
    left = right.clone()
    get_kernels().scale_rows(
        nrows=nao,
        ncols=nao,
        scales_ptr=evals.pointer,
        matrix_ptr=left.pointer,
        )

    K = GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )
    CuestDFIntCompute.nonsymmetric_exchange(
        handle=helper.cuest_handle,
        df_int_plan=helper.df_int_plan,
        nocc=nao,
        Cocc_left_ptr=left.pointer,
        Cocc_right_ptrs=[right.pointer],
        Kptrs=[K.pointer],
        )

    GPUMatrixUtility.daxpy(
        alpha=-0.5,
        x=K,
        y=J,
        )
    return J


def fock_response(
    helper,
    C,
    nocc,
    nvir,
    P_ao,
    out,
    ):
    """
    The occupied-virtual Fock response.

    out[a,i] = 2 [Cv veff(P_ao) Co^T]_ai, 

    This is the coupling that appears both in the z-vector right-hand side and in every
    application of the orbital Hessian, differing only in the density it is formed from.
    """

    nao = C.ncols
    scratch = GPUMatrix(
        nrows=nvir,
        ncols=nao,
        dtype=np.double,
        )

    # eq. (158)
    v = compute_veff(helper, P_ao)
    GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=False,
        m=nvir,
        n=nao,
        k=nao,
        alpha=1.0,
        a=C,
        offa=nocc * nao,
        lda=nao,
        b=v,
        offb=0,
        ldb=nao,
        beta=0.0,
        c=scratch,
        offc=0,
        ldc=nao,
        )
    GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=True,
        m=nvir,
        n=nocc,
        k=nao,
        alpha=2.0,
        a=scratch,
        offa=0,
        lda=nao,
        b=C,
        offb=0,
        ldb=nao,
        beta=0.0,
        c=out,
        offc=0,
        ldc=nocc,
        )


def solve_z_vector(
    kernels,
    helper,
    C,
    eps,
    nocc,
    nvir,
    nao,
    minus_L_ai,
    *,
    tolerance=1.0e-9,
    maxiter=100,
    ):
    """
    Preconditioned conjugate-gradient solution of the RHF z-vector equations.

    The orbital Hessian is applied as
        (A P_ai)_ai = (eps_a - eps_i) P_ai + [Cv (4J - K - K^T) Co^T]_ai
    and every vector, including the CG bookkeeping, stays on the device; only three
    scalars per iteration come back to the host.
    """

    eps_occ = eps.pointer
    eps_vir = offset_pointer(eps, nocc)

    def new_vo_vector():
        return GPUMatrix(
            nrows=nvir,
            ncols=nocc,
            dtype=np.double,
            )

    def precondition(x, y):
        kernels.apply_denominator(
            nvir=nvir,
            nocc=nocc,
            invert=True,
            alpha=1.0,
            beta=0.0,
            eps_occ_ptr=eps_occ,
            eps_vir_ptr=eps_vir,
            x_ptr=x.pointer,
            y_ptr=y.pointer,
            )

    P_ao = GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )
    scratch = GPUMatrix(
        nrows=nao,
        ncols=nocc,
        dtype=np.double,
        )

    # eq. (163)
    def hessian_vector(P_ai, out):
        # The response is wanted at the density P + P^T with P = Cv^T P_ai Co; veff averages
        # its argument with its own transpose, so 2P is what gets passed into fock_response.
        GPUMatrixUtility.c_dgemm(
            transa=True,
            transb=False,
            m=nao,
            n=nocc,
            k=nvir,
            alpha=1.0,
            a=C,
            offa=nocc * nao,
            lda=nao,
            b=P_ai,
            offb=0,
            ldb=nocc,
            beta=0.0,
            c=scratch,
            offc=0,
            ldc=nocc,
            )
        GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=False,
            m=nao,
            n=nao,
            k=nocc,
            alpha=2.0,
            a=scratch,
            offa=0,
            lda=nocc,
            b=C,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=P_ao,
            offc=0,
            ldc=nao,
            )
        fock_response(
            helper,
            C,
            nocc,
            nvir,
            P_ao,
            out,
            )
        # eq. (163)
        kernels.apply_denominator(
            nvir=nvir,
            nocc=nocc,
            invert=False,
            alpha=1.0,
            beta=1.0,
            eps_occ_ptr=eps_occ,
            eps_vir_ptr=eps_vir,
            x_ptr=P_ai.pointer,
            y_ptr=out.pointer,
            )

    P_ai, r, p, zr, Ap = (new_vo_vector() for _ in range(5))

    precondition(
        minus_L_ai,
        P_ai,
        )
    hessian_vector(
        P_ai,
        r,
        )
    GPUMatrixUtility.dscal(
        alpha=-1.0,
        x=r,
        )
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=minus_L_ai,
        y=r,
        )
    precondition(
        r,
        p,
        )
    rz_old = GPUMatrixUtility.ddot(
        n=r.size,
        x=r,
        offx=0,
        incx=1,
        y=p,
        offy=0,
        incy=1,
        )

    print('Z-vector iterations:')
    print('')
    print('Iter:       Residual')
    for iteration in range(1, maxiter + 1):
        r_norm = np.sqrt(GPUMatrixUtility.ddot(
            n=r.size,
            x=r,
            offx=0,
            incx=1,
            y=r,
            offy=0,
            incy=1,
            ))
        print('%-4d:     %10.4e' % (iteration - 1, r_norm))
        if r_norm <= tolerance:
            break
        hessian_vector(p, Ap)
        pAp = GPUMatrixUtility.ddot(
            n=p.size,
            x=p,
            offx=0,
            incx=1,
            y=Ap,
            offy=0,
            incy=1,
            )
        alpha = rz_old / pAp
        GPUMatrixUtility.daxpy(
            alpha=alpha,
            x=p,
            y=P_ai,
            )
        GPUMatrixUtility.daxpy(
            alpha=-alpha,
            x=Ap,
            y=r,
            )
        precondition(r, zr)
        rz_new = GPUMatrixUtility.ddot(
            n=r.size,
            x=r,
            offx=0,
            incx=1,
            y=zr,
            offy=0,
            incy=1,
            )
        GPUMatrixUtility.dscal(
            alpha=rz_new / rz_old,
            x=p,
            )
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=zr,
            y=p,
            )
        rz_old = rz_new
    else:
        raise RuntimeError('z-vector iterations did not converge')

    print('')
    print('Z-vector iterations converged')
    print('')
    return P_ai


# => Three-center TPDMs <= #

def build_gamma3_separable(
    B,
    P1,
    P2,
    naux,
    nmo,
    ):
    """
    gamma3 = 0.5 dE/dB for the bilinear two-electron energy of densities P1 and P2,
       E(P1,P2) = sum_A [ g1[A] g2[A] - 0.5 tr(P1 B[A] P2 B[A]) ],
       g[A] = sum_pq P_pq B[A,pq]

    Written in place: B is consumed and returned as gamma3, which has the same
    shape. These are the two largest tensors in the JK-fit basis and holding
    both would nearly double the memory requirements, so B is destroyed. The
    fitting charges g1 and g2 have to be formed first because they read all of
    B; after that block A depends only on B[A], so each block is assembled
    complete in a scratch buffer and written back only once B[A] has been read
    for the last time.
    """

    # eq. (174)
    g1 = GPUMatrix(
        nrows=1,
        ncols=naux,
        dtype=np.double,
        )
    GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=True,
        m=1,
        n=naux,
        k=nmo * nmo,
        alpha=1.0,
        a=P1,
        offa=0,
        lda=nmo * nmo,
        b=B,
        offb=0,
        ldb=nmo * nmo,
        beta=0.0,
        c=g1,
        offc=0,
        ldc=naux,
        )
    # eq. (174)
    g2 = GPUMatrix(
        nrows=1,
        ncols=naux,
        dtype=np.double,
        )
    GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=True,
        m=1,
        n=naux,
        k=nmo * nmo,
        alpha=1.0,
        a=P2,
        offa=0,
        lda=nmo * nmo,
        b=B,
        offb=0,
        ldb=nmo * nmo,
        beta=0.0,
        c=g2,
        offc=0,
        ldc=naux,
        )

    scratch = GPUMatrix(
        nrows=nmo,
        ncols=nmo,
        dtype=np.double,
        )
    block = GPUMatrix(
        nrows=nmo,
        ncols=nmo,
        dtype=np.double,
        )
    for A in range(naux):
        # eq. (174)
        for left, right, beta in ((P1, P2, 0.0), (P2, P1, 1.0)):
            GPUMatrixUtility.c_dgemm(
                transa=False,
                transb=False,
                m=nmo,
                n=nmo,
                k=nmo,
                alpha=1.0,
                a=left,
                offa=0,
                lda=nmo,
                b=B,
                offb=A * nmo * nmo,
                ldb=nmo,
                beta=0.0,
                c=scratch,
                offc=0,
                ldc=nmo,
                )
            GPUMatrixUtility.c_dgemm(
                transa=False,
                transb=False,
                m=nmo,
                n=nmo,
                k=nmo,
                alpha=-0.25,
                a=scratch,
                offa=0,
                lda=nmo,
                b=right,
                offb=0,
                ldb=nmo,
                beta=beta,
                c=block,
                offc=0,
                ldc=nmo,
                )

        # The Coulomb halves are rank one in the fitting index, so each is a single row of
        # charges times a density; taking them one A at a time keeps g1 and g2 on the device.
        # eq. (174)
        for charges, density in ((g2, P1), (g1, P2)):
            GPUMatrixUtility.c_dgemm(
                transa=False,
                transb=False,
                m=1,
                n=nmo * nmo,
                k=1,
                alpha=0.5,
                a=charges,
                offa=A,
                lda=1,
                b=density,
                offb=0,
                ldb=nmo * nmo,
                beta=1.0,
                c=block,
                offc=0,
                ldc=nmo * nmo,
                )

        # B[A] has now been read for the last time.
        GPUMatrixUtility.dcopy(
            n=nmo * nmo,
            x=block,
            offx=0,
            y=B,
            offy=A * nmo * nmo,
            )

    return B


# => Gradient assembly <= #

def mp2_gradient(
    *,
    oe,
    C,
    eps,
    nocc,
    nvir,
    nfrz=0,
    natom,
    helper_ri,
    helper_jk,
    naux_ri,
    naux_jk,
    timer,
    ):

    # Nothing here touches the RHF object: C and eps are handed in as copies, the J/K builds
    # run on helper_jk's plan and every gradient term (including the reference) is assembled
    # from the example's own plans, so the wavefunction has already been released.
    nmo = nocc + nvir
    nao = C.ncols

    # Correlation quantities are fitted in the RI basis; the separable term from orbital
    # relaxation needs the SCF's own JK fitting, and is transformed much later so that the
    # two fitting bases are never resident at once.
    #
    # The three correlation blocks go in as one batched transform so the AO three-center
    # integrals are traversed once. Batching concatenates the coefficient matrices, and
    # the occ-occ, occ-vir and vir-vir tensors come back packed back to back.
    timer.start(key="MO Integrals")

    C_left = concatenate_orbitals(
        C,
        ((0, nocc), (0, nocc), (nocc, nvir)),
        nao,
        )
    C_right = concatenate_orbitals(
        C,
        ((0, nocc), (nocc, nvir), (nocc, nvir)),
        nao,
        )

    B_ri = compute_B(
        helper_ri,
        naux_ri,
        C_left,
        (nocc, nocc, nvir),
        0,
        C_right,
        (nocc, nvir, nvir),
        0,
        )
    oo_offset = 0
    ov_offset = naux_ri * nocc * nocc
    vv_offset = ov_offset + naux_ri * nocc * nvir
    timer.stop(key="MO Integrals")

    kernels = get_kernels()

    timer.start(key="Amplitudes")
    e_corr, P_ij, P_ab, g3_ov = compute_correlation_intermediates(
        kernels,
        B_ri,
        ov_offset,
        eps,
        naux_ri,
        nocc,
        nvir,
        nfrz=nfrz,
        )
    timer.stop(key="Amplitudes")

    timer.start(key="Lagrangian")
    half_L_ai_NS, W_I_ij, W_I_ab, half_W_I_ai = compute_lagrangian(
        B_ri,
        oo_offset,
        ov_offset,
        vv_offset,
        g3_ov,
        naux_ri,
        nocc,
        nvir,
        nfrz=nfrz,
        )
    timer.stop(key="Lagrangian")

    # => Non-separable derivative, taken early so the RI side can be released <= #

    # Everything fitted in the RI basis is finished with at this point: B_ri's last reader
    # was compute_lagrangian, and g3_ov was final the moment the amplitudes were. Contracting
    # now (before the first J/K build below instantiates helper_jk's plan) is what keeps
    # the two fitting bases from ever being resident together. Safe with respect to
    # MEMORY_POLICY_OVERWRITE because the RI plan never serves a Coulomb or exchange build.
    #
    # The non-separable TPDM lives entirely in the occ-vir block, so it is contracted on
    # its own subspaces rather than padded out to nmo x nmo. B_ia and B_ai are the same
    # integrals, so the (vo) ordering of the TPDM contributes exactly what the (ov) ordering
    # does. A single (ov) term with the TPDM doubled is therefore equivalent to contracting
    # both orderings, and avoids building the transposed TPDM and the swapped coefficients.
    #
    # B_ri goes first: it is by far the largest thing alive and its last reader was
    # compute_lagrangian.
    del B_ri, C_left, C_right

    timer.start(key="MO Derivative Integrals")
    nact = nocc - nfrz
    # g3_ov is not needed after this contraction, so it is doubled in place.
    GPUMatrixUtility.dscal(
        alpha=2.0,
        x=g3_ov,
        )

    # The derivative driver zeroes its output, so each call needs its own buffer.
    G_ri = GPUMatrix(
        nrows=natom,
        ncols=3,
        dtype=np.double,
        )
    # eq. (171)
    helper_ri.compute_3c_derivative(
        C_left=C,
        nmo_c1=nact,
        c1_offset=nfrz,
        C_right=C,
        nmo_c2=nvir,
        c2_offset=nocc,
        three_center_tpdm=g3_ov,
        out_gradient=G_ri,
        )
    timer.stop(key="MO Derivative Integrals")

    del g3_ov
    # The memoization in helper_ir stuffs the int plan in an underscored variable
    del helper_ri._df_int_plan

    timer.start(key="Lagrangian")

    def symmetrize(matrix, scale=1.0):
        """scale * (M + M^T) for a square matrix."""

        return GPUMatrixUtility.dgeam(
            transpose1=False,
            alpha=scale,
            mat1=matrix,
            transpose2=True,
            beta=scale,
            mat2=matrix,
            )

    def transpose(matrix):
        out = GPUMatrix(
            nrows=matrix.ncols,
            ncols=matrix.nrows,
            dtype=np.double,
            )
        kernels.transpose_blocks(
            nblocks=1,
            nrows=matrix.nrows,
            ncols=matrix.ncols,
            in_ptr=matrix.pointer,
            out_ptr=out.pointer,
            )
        return out

    def scatter(block, destination, row, column, scale=1.0):
        """Write scale * block into the (row, column) corner of an (nmo, nmo) matrix."""

        kernels.scatter_block(
            nrows=block.nrows,
            ncols=block.ncols,
            ld_dst=nmo,
            alpha=scale,
            src_ptr=block.pointer,
            dst_ptr=offset_pointer(destination, row * nmo + column),
            )

    def mo_to_ao(matrix):
        """C^T M C, taking an (nmo, nmo) MO-basis matrix to the AO basis."""

        scratch = GPUMatrix(
            nrows=nao,
            ncols=nmo,
            dtype=np.double,
            )
        GPUMatrixUtility.c_dgemm(
            transa=True,
            transb=False,
            m=nao,
            n=nmo,
            k=nmo,
            alpha=1.0,
            a=C,
            offa=0,
            lda=nao,
            b=matrix,
            offb=0,
            ldb=nmo,
            beta=0.0,
            c=scratch,
            offc=0,
            ldc=nmo,
            )
        out = GPUMatrix(
            nrows=nao,
            ncols=nao,
            dtype=np.double,
            )
        GPUMatrixUtility.c_dgemm(
            transa=False,
            transb=False,
            m=nao,
            n=nao,
            k=nmo,
            alpha=1.0,
            a=scratch,
            offa=0,
            lda=nmo,
            b=C,
            offb=0,
            ldb=nao,
            beta=0.0,
            c=out,
            offc=0,
            ldc=nao,
            )
        return out

    # => Orbital relaxation <= #

    P_unrelaxed = GPUMatrix(
        nrows=nmo,
        ncols=nmo,
        dtype=np.double,
        )
    # eq. (177)
    scatter(
        symmetrize(P_ij),
        P_unrelaxed,
        nfrz,
        nfrz,
        )
    # eq. (178)
    scatter(
        symmetrize(P_ab),
        P_unrelaxed,
        nocc,
        nocc,
        )

    if nfrz:
        # Core-active block of the correlation density, P_Ki = 2 / (eps_i - eps_K) sum_jab
        # T_ij^ab (Ka|jb). The sum is the K row of -W_I_ij, so no iteration is needed.
        eps_np = eps.to_numpy().ravel()
        W_I_ij_np = W_I_ij.to_numpy()
        # eq. (176)
        P_Ki_np = -W_I_ij_np[:nfrz, nfrz:] / (eps_np[None, nfrz:nocc] - eps_np[:nfrz, None])
        P_Ki = GPUMatrix.from_numpy(np.ascontiguousarray(P_Ki_np))
        scatter(
            P_Ki,
            P_unrelaxed,
            0,
            nfrz,
            )
        scatter(
            transpose(P_Ki),
            P_unrelaxed,
            nfrz,
            0,
            )

    half_L_ai = GPUMatrix(
        nrows=nvir,
        ncols=nocc,
        dtype=np.double,
        )
    # eq. (159)
    fock_response(
        helper_jk,
        C,
        nocc,
        nvir,
        mo_to_ao(P_unrelaxed),
        half_L_ai,
        )
    # eq. (159)
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=half_L_ai_NS,
        y=half_L_ai,
        )
    timer.stop(key="Lagrangian")

    # The z-vector equations are A P_ai = -L_ai
    timer.start(key="Z Vector Solve")
    minus_half_L_ai = half_L_ai.clone()
    GPUMatrixUtility.dscal(
        alpha=-1.0,
        x=minus_half_L_ai,
        )
    # eq. (163)
    half_P_ai = solve_z_vector(
        kernels,
        helper_jk,
        C,
        eps,
        nocc,
        nvir,
        nao,
        minus_half_L_ai,
        )
    timer.stop(key="Z Vector Solve")

    timer.start(key="Fock and Overlap Terms")
    P_mo = P_unrelaxed.clone()
    # eq. (180)
    scatter(
        half_P_ai,
        P_mo,
        nocc,
        0,
        )
    # eq. (180)
    scatter(
        transpose(half_P_ai),
        P_mo,
        0,
        nocc,
        )

    # => Generalized Fock and energy-weighted densities (MO) <= #

    W_I_mo = GPUMatrix(
        nrows=nmo,
        ncols=nmo,
        dtype=np.double,
        )
    # eq. (181)
    scatter(
        W_I_ij,
        W_I_mo,
        0,
        0,
        )
    # eq. (182)
    scatter(
        W_I_ab,
        W_I_mo,
        nocc,
        nocc,
        )
    # eq. (183)
    scatter(
        transpose(half_W_I_ai),
        W_I_mo,
        0,
        nocc,
        )
    # eq. (183)
    scatter(
        half_W_I_ai,
        W_I_mo,
        nocc,
        0,
        )

    W_II_mo = GPUMatrix(
        nrows=nmo,
        ncols=nmo,
        dtype=np.double,
        )
    # eq. (184), (185), (186)
    kernels.make_W(
        nmo=nmo,
        nocc=nocc,
        eps_ptr=eps.pointer,
        dm1_ptr=P_mo.pointer,
        W_ptr=W_II_mo.pointer,
        )
    GPUMatrixUtility.dscal(
        alpha=-1.0,
        x=W_II_mo,
        )

    W_I_ao = mo_to_ao(W_I_mo)
    W_II_ao = mo_to_ao(W_II_mo)
    P_ao = mo_to_ao(P_mo)

    p1 = GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )
    GPUMatrixUtility.c_dgemm(
        transa=True,
        transb=False,
        m=nao,
        n=nao,
        k=nocc,
        alpha=1.0,
        a=C,
        offa=0,
        lda=nao,
        b=C,
        offb=0,
        ldb=nao,
        beta=0.0,
        c=p1,
        offc=0,
        ldc=nao,
        )

    # The reference contributions to the one-electron gradients
    W_hf = GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )
    C_occ_scaled = GPUMatrix(
        nrows=nocc,
        ncols=nao,
        dtype=np.double,
        )
    GPUMatrixUtility.dcopy(
        n=nocc * nao,
        x=C,
        offx=0,
        y=C_occ_scaled,
        offy=0,
        )
    kernels.scale_rows(
        nrows=nocc,
        ncols=nao,
        scales_ptr=eps.pointer,
        matrix_ptr=C_occ_scaled.pointer,
        )
    GPUMatrixUtility.c_dgemm(
        transa=True,
        transb=False,
        m=nao,
        n=nao,
        k=nocc,
        alpha=1.0,
        a=C,
        offa=0,
        lda=nao,
        b=C_occ_scaled,
        offb=0,
        ldb=nao,
        beta=0.0,
        c=W_hf,
        offc=0,
        ldc=nao,
        )

    scratch = GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )
    GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=False,
        m=nao,
        n=nao,
        k=nao,
        alpha=1.0,
        a=p1,
        offa=0,
        lda=nao,
        b=compute_veff(helper_jk, symmetrize(P_ao)),
        offb=0,
        ldb=nao,
        beta=0.0,
        c=scratch,
        offc=0,
        ldc=nao,
        )
    W_III_ao = GPUMatrix(
        nrows=nao,
        ncols=nao,
        dtype=np.double,
        )
    # eq. (187)
    GPUMatrixUtility.c_dgemm(
        transa=False,
        transb=False,
        m=nao,
        n=nao,
        k=nao,
        alpha=-1.0,
        a=scratch,
        offa=0,
        lda=nao,
        b=p1,
        offb=0,
        ldb=nao,
        beta=0.0,
        c=W_III_ao,
        offc=0,
        ldc=nao,
        )
    timer.stop(key="Fock and Overlap Terms")

    # => Two-electron TPDM: non-separable correlation plus the relaxed-density correction <= #

    timer.start(key="MO Integrals")
    B_jk = compute_B(
        helper_jk,
        naux_jk,
        C,
        nmo,
        0,
        C,
        nmo,
        0,
        )
    timer.stop(key="MO Integrals")

    timer.start(key="Gamma 3")
    P_hf_mo = GPUMatrix(
        nrows=nmo,
        ncols=nmo,
        dtype=np.double,
        )
    kernels.set_diagonal(
        count=nocc,
        ld=nmo,
        value=2.0,
        matrix_ptr=P_hf_mo.pointer,
        )

    P_total_mo = P_mo.clone()
    GPUMatrixUtility.daxpy(
        alpha=0.5,
        x=P_hf_mo,
        y=P_total_mo,
        )
    # eq. (174)
    gamma3_jk = build_gamma3_separable(
        B_jk,
        P_total_mo,
        P_hf_mo,
        naux_jk,
        nmo,
        )
    del B_jk
    timer.stop(key="Gamma 3")

    # => Contract every piece with the derivative integrals <= #

    G = GPUMatrix(
        nrows=natom,
        ncols=3,
        dtype=np.double,
        )
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=oe.nuclear_repulsion_gradient(),
        y=G,
        )

    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=G_ri,
        y=G,
        )

    timer.start(key="MO Derivative Integrals")
    # The separable term (reference plus relaxed correction) is fitted in the SCF's
    # JK-fit basis, while the non-seperable part above generally uses a different RI fit basis
    G_jk = GPUMatrix(
        nrows=natom,
        ncols=3,
        dtype=np.double,
        )
    # eq. (174)
    helper_jk.compute_3c_derivative(
        C_left=C,
        nmo_c1=nmo,
        c1_offset=0,
        C_right=C,
        nmo_c2=nmo,
        c2_offset=0,
        three_center_tpdm=gamma3_jk,
        out_gradient=G_jk,
        )
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=G_jk,
        y=G,
        )
    timer.stop(key="MO Derivative Integrals")

    timer.start(key="1e Derivative Integrals")
    D_total_ao = P_ao.clone()
    GPUMatrixUtility.daxpy(
        alpha=2.0,
        x=p1,
        y=D_total_ao,
        )
    # eq. (24)
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=oe.kinetic_gradient(D_total_ao),
        y=G,
        )
    # eq. (24)
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=oe.potential_gradient(D_total_ao),
        y=G,
        )
    if oe.do_ecp:
        # eq. (24)
        GPUMatrixUtility.daxpy(
            alpha=1.0,
            x=oe.ecp_gradient(D_total_ao),
            y=G,
            )

    W = W_I_ao
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=W_II_ao,
        y=W,
        )
    GPUMatrixUtility.daxpy(
        alpha=-2.0,
        x=W_hf,
        y=W,
        )
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=W_III_ao,
        y=W,
        )
    # eq. (24)
    GPUMatrixUtility.daxpy(
        alpha=1.0,
        x=oe.overlap_gradient(symmetrize(W, scale=0.5)),
        y=G,
        )

    G = G.to_numpy()
    timer.stop(key="1e Derivative Integrals")

    return e_corr, G


# => Finite difference validation machinery <= #

def compute_total_energy(
    cuest_handle,
    reference,
    xyz,
    **kwargs,
    ):
    """
    E_HF (JK fitting) + E_corr (RI fitting) at one geometry, coordinates in bohr.
    """

    print_timings = kwargs.get('print_timings', False)

    mytimer = Timer(detailed=True)

    mytimer.start(key="RHF Energy",)
    ri_auxiliary_name = kwargs.pop('ri_auxiliary_name')
    freeze_core = kwargs.pop('freeze_core', False)

    molecule = cuest_scf.Molecule(
        N=reference.N,
        Z=reference.Z,
        xyz=xyz,
        )

    rhf = build_rhf(
        cuest_handle,
        molecule,
        print_level=0,
        **kwargs,
        )
    mytimer.stop(key="RHF Energy")

    mytimer.start(key="MO Integrals")
    auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % ri_auxiliary_name),
        molecule=molecule,
        )
    helper = CuestMOIntegralHelper(
        cuest_handle=rhf.cuest_handle,
        molecule=rhf.molecule,
        primary=rhf.primary,
        ri_auxiliary=auxiliary,
        primary_name=rhf.primary_name,
        ri_auxiliary_name=ri_auxiliary_name,
        threshold_pq=rhf.threshold_pq,
        df_fitting_eigenvalue_cutoff=rhf.df_fitting_eigenvalue_cutoff,
        ecp_basis=rhf.ecp_basis,
        )

    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']
    nfrz = helper.compute_frozen_core() if freeze_core else 0

    C = rhf.tensors['C']

    B_ov = compute_B(
        helper,
        auxiliary.nao,
        C,
        nocc,
        0,
        C,
        nvir,
        nocc,
        )
    mytimer.stop(key="MO Integrals")

    mytimer.start(key="MP2 Energy")
    e_corr, _, _, _ = compute_correlation_intermediates(
        get_kernels(),
        B_ov,
        0,
        rhf.tensors['eps'],
        auxiliary.nao,
        nocc,
        nvir,
        nfrz=nfrz,
        energy_only=True,
        )
    mytimer.stop(key="MP2 Energy")

    if print_timings:
        mytimer.print_out(header='Energy timings:')

    return rhf.compute_energy() + e_corr


def finite_difference_gradient(
    cuest_handle,
    molecule,
    *,
    delta=1.0e-3,
    **kwargs,
    ):

    xyz0 = np.array(molecule.xyz)
    G = np.zeros((molecule.natom, 3))

    stencil = list(zip(
        np.array([1.0, -8.0, 8.0, -1.0]) / (12.0 * delta),
        delta * np.array([-2.0, -1.0, 1.0, 2.0]),
        ))

    print('Finite difference gradient (4 point central, %.1e bohr step):' % delta)
    print('')
    for atom in range(molecule.natom):
        for direction in range(3):
            for weight, displacement in stencil:
                xyz = xyz0.copy()
                xyz[atom, direction] += displacement
                G[atom, direction] += weight * compute_total_energy(
                    cuest_handle,
                    molecule,
                    xyz,
                    **kwargs,
                    )
        print('  %-2s  %15.9f %15.9f %15.9f' % (
            molecule.N[atom], G[atom, 0], G[atom, 1], G[atom, 2]))
    print('')
    return G


# => Driver <= #

def main():

    cuest_handle = cuest_scf.CuestHandle()

    molecule = cuest_scf.Molecule.parse_from_xyz_file(f'{filedir}/ethanol.xyz')

    primary_name = 'def2-qzvp'
    ri_auxiliary_name = 'def2-qzvp-rifit'
    auxiliary_name = 'def2-universal-jkfit'
    minao_name = 'minao-1'
    ecp_name = None

    # Set this to True to freeze the core electrons.  If False, a larger basis set,
    # e.g. cc-pCVDZ vs. cc-pVDZ, is usually required to correctly describe the core electrons
    freeze_core = True

    timer = Timer(detailed=True)
    timer.start(key="Total")
    timer.start(key="RHF Energy")

    rhf = build_rhf(
        cuest_handle,
        molecule,
        primary_name=primary_name,
        auxiliary_name=auxiliary_name,
        minao_name=minao_name,
        ecp_name=ecp_name,
        threshold_pq=1.0E-14,
        )


    primary = rhf.primary
    ecp_basis = rhf.ecp_basis
    threshold_pq = rhf.threshold_pq
    df_fitting_eigenvalue_cutoff = rhf.df_fitting_eigenvalue_cutoff

    auxiliary_jk = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % auxiliary_name),
        molecule=molecule,
        )
    helper_jk = CuestMOIntegralHelper(
        cuest_handle=cuest_handle,
        molecule=molecule,
        primary=primary,
        ri_auxiliary=auxiliary_jk,
        primary_name=primary_name,
        ri_auxiliary_name=auxiliary_name,
        threshold_pq=threshold_pq,
        df_fitting_eigenvalue_cutoff=df_fitting_eigenvalue_cutoff,
        integral_direct=False,
        ecp_basis=ecp_basis,
        )

    naux_jk = auxiliary_jk.nao

    nfrz = helper_jk.compute_frozen_core() if freeze_core else 0
    print(f'Frozen core orbitals: {nfrz}')

    ri_auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(
        os.path.join(basisdir, '%s.gbs' % ri_auxiliary_name),
        molecule=molecule,
        )
    helper_ri = CuestMOIntegralHelper(
        cuest_handle=cuest_handle,
        molecule=molecule,
        primary=primary,
        ri_auxiliary=ri_auxiliary,
        primary_name=primary_name,
        ri_auxiliary_name=ri_auxiliary_name,
        threshold_pq=threshold_pq,
        df_fitting_eigenvalue_cutoff=df_fitting_eigenvalue_cutoff,
        integral_direct=False,
        ecp_basis=ecp_basis,
        )
    naux_ri = ri_auxiliary.nao


    e_scf = rhf.compute_energy()

    # Everything the correlation treatment needs from the reference is taken now, as
    # copies, so that the RHF object is not the owner of any of it.
    C = rhf.tensors['C'].clone()
    eps = rhf.tensors['eps'].clone()
    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']
    natom = rhf.primary.natom
    timer.stop(key="RHF Energy")

    # The reference gradient is assembled inside mp2_gradient from the example's own plans,
    # so nothing needs the wavefunction past this point. Deleting it here frees a
    # density-fitting plan that would otherwise sit alongside helper_jk's duplicate of it
    # for the whole correlation treatment.
    del rhf

    oe = OneElectronHelper(
        helper_jk,
        molecule,
        ecp_basis,
        )

    e_corr, G = mp2_gradient(
        oe=oe,
        C=C,
        eps=eps,
        nocc=nocc,
        nvir=nvir,
        nfrz=nfrz,
        natom=natom,
        helper_ri=helper_ri,
        helper_jk=helper_jk,
        naux_ri=naux_ri,
        naux_jk=naux_jk,
        timer=timer,
        )

    print(f'RHF Energy:             {e_scf:24.16E}')
    print(f'MP2 Correlation Energy: {e_corr:24.16E}')
    print(f'MP2 Energy:             {e_scf + e_corr:24.16E}')
    print()
    print('MP2 Gradient (Hartree/Bohr):')
    print()
    for atom in range(molecule.natom):
        print('  %-2s  %15.9f %15.9f %15.9f' % (
            molecule.N[atom], G[atom, 0], G[atom, 1], G[atom, 2]))
    print()

    if '--fd' in sys.argv:
        G_fd = finite_difference_gradient(
            cuest_handle,
            molecule,
            primary_name=primary_name,
            auxiliary_name=auxiliary_name,
            ri_auxiliary_name=ri_auxiliary_name,
            minao_name=minao_name,
            ecp_name=ecp_name,
            threshold_pq=1.0E-14,
            freeze_core=freeze_core,
            )
        print('  max|analytic - finite difference| = %10.4e' % np.max(np.abs(G - G_fd)))
        print()

    timer.stop(key="Total")
    timer.print_out(header='Gradient timings:')

    return e_scf + e_corr, G


if __name__ == '__main__':
    main()
