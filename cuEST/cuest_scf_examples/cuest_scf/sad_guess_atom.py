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

from .ao_basis import AOBasis

from .sad_atom_structure import SADAtomStructure

from .sad_overlap import SADOverlap

from .memoized_property import memoized_property

class SADGuessAtom(object):

    def __init__(
        self,
        primary : AOBasis,
        minao : AOBasis,
        structure : SADAtomStructure,
        ):

        self.primary = primary
        self.minao = minao
        self.structure = structure

        if self.primary.natom != 1: raise RuntimeError('primary.natom != 1')
        if self.minao.natom != 1: raise RuntimeError('minao.natom != 1')

        if [shell.L for shell in minao.shells[0]] != structure.Ls:
            raise RuntimeError('minao L structure is not correct for this atom. N = %d' % (structure.N))

        if not minao.is_pure: 
            raise RuntimeError('minao is not pure')

        if minao.nao > primary.nao: 
            raise RuntimeError('minao.nao > primary.nao')

    @memoized_property
    def Smm(self):
        return SADOverlap.compute_atomic_overlap(
            basis1=self.minao,
            basis2=self.minao,
            )

    @memoized_property
    def Smp(self):
        return SADOverlap.compute_atomic_overlap(
            basis1=self.minao,
            basis2=self.primary,
            )

    @memoized_property
    def Spp(self):
        return SADOverlap.compute_atomic_overlap(
            basis1=self.primary,
            basis2=self.primary,
            )

    @memoized_property
    def Xm(self):
        s, U = np.linalg.eigh(self.Smm)
        X = np.einsum('pi,i->pi', U, s**(-0.5))
        return np.dot(X, U.T)

    @memoized_property
    def Xp(self):
        s, U = np.linalg.eigh(self.Spp)
        ind = s > 1.0E-12
        s = s[ind]
        U = U[:, ind]
        X = np.einsum('pi,i->ip', U, s**(-0.5))
        return X

    @memoized_property
    def M(self):
        return np.dot(np.dot(self.Xm, self.Smp), self.Xp.T)

    @memoized_property
    def U(self):
        return self.UsV[0] 

    @memoized_property
    def s(self):
        return self.UsV[1] 
        
    @memoized_property
    def V(self):
        return self.UsV[2] 
        
    @memoized_property
    def UsV(self):
        return np.linalg.svd(self.M, full_matrices=False)
        
    @memoized_property
    def P(self):
        return np.dot(self.U, self.V)

    @memoized_property
    def C(self):
        return np.dot(self.P, self.Xp)

    @property
    def minao_health(self):
        return np.max(np.abs(self.Smm - np.eye(self.Smm.shape[0])))

    @property
    def overlap_health(self):
        return np.max(np.abs(self.s - 1.0))
    
