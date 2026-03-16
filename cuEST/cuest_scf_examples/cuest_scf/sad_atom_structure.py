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
    
class SADAtomStructure(object):

    def __init__(
        self,
        *,
        N : int,
        Ls : list,
        actives : list,
        ):

        self.N = N
        self.Ls = Ls
        self.actives = actives

        if not all(isinstance(_, int) for _ in Ls): raise RuntimeError('Ls is not list of int')
        if not all(isinstance(_, bool) for _ in actives): raise RuntimeError('actives is not list of bool')
        if list(sorted(actives)) != actives: raise RuntimeError('actives is not sorted')

        if self.N > 0 and self.nao_active == 0:
            raise RuntimeError('N > 0 and nao_active == 0')

    @property
    def nshell(self):
        return len(self.Ls)

    @property
    def max_L(self):
        return max(self.Ls) if self.nshell else 0
        
    @property
    def nao_inactive(self):
        return sum(2*L + 1 for L, active in zip(self.Ls, self.actives) if not active)

    @property
    def nao_active(self):
        return sum(2*L + 1 for L, active in zip(self.Ls, self.actives) if active)

    @property
    def nao(self):
        return self.nao_inactive + self.nao_active

    """
    Works in terms of RHF docc
    """
    @property
    def nocc(self):
        if self.nao == 0:
            return np.array([])
        nocc_total = self.N / 2
        if self.nao_inactive > nocc_total:
            raise RuntimeError('nao_inactive > nocc_total. N = %d' % self.N)
        if self.nao < nocc_total:
            raise RuntimeError('nao < nocc_total. N = %d' % self.N)
        nocc_active = nocc_total - self.nao_inactive
        n = nocc_active / self.nao_active
        return np.array(
            [1.0] * self.nao_inactive +
            [n] * self.nao_active)

    @property
    def string(self):
        s = 'SADAtomStructure: N = %d\n' % (self.N)
        for P in range(self.nshell):
            s += '%1d : %s\n' % (self.Ls[P], 'active' if self.actives[P] else 'inactive')
        return s

    @property
    def aufbau_string(self):
        symbols = ['s', 'p', 'd', 'f']
        if self.max_L >= len(symbols): raise RuntimeError('max_L too high for symbol table')

        s = ''
        s += '['
        for P in range(self.nshell):
            if P > 0 and self.actives[P] and not self.actives[P-1]:
                s += ']['
            L = self.Ls[P]        
            s += str(sum(1 for L2 in self.Ls[:P] if L2 == L) + 1 + L)
            s += symbols[L]
        s += ']'

        return s

    def __str__(self):
        return self.aufbau_string

    """
    Technique used here:
      - In the S block, only the highest s shell is open.
      - In the P block, both the highest s and highest p shells are open.
      - In the D block, only the highest d shell is open.
      - In the F block, only the highest f shell is open.
    
    Specific choice:
      - The closed and open blocks are strictly canonicalized according to
        (n, l) ordering.
    """

    @staticmethod
    def build(N):
        if N == 0:
            # Gh [][]
            return SADAtomStructure(
                N=N,
                Ls=[],
                actives=[], 
                )
        elif N >= 1 and N <= 2:
            # H  - He: [][1s]
            return SADAtomStructure(
                N=N,
                Ls=[0],
                actives=[True],
                )
        elif N >= 3 and N <= 4:
            # Li - Be: [1s][2s]
            return SADAtomStructure(
                N=N,
                Ls=[0, 0],
                actives=[False]*1 + [True]*1,
                )
        elif N >= 5 and N <= 10:
            # B  - Ne: [1s][2s2p]
            return SADAtomStructure(
                N=N,
                Ls=[0, 0, 1],
                actives=[False]*1 + [True]*2,
                )
        elif N >= 11 and N <= 12:
            # Na - Mg: [1s2s2p][3s]
            return SADAtomStructure(
                N=N,
                Ls=[0, 0, 1, 0],
                actives=[False]*3 + [True]*1,
                )
        elif N >= 13 and N <= 18:
            # Al - Ar: [1s2s2p][3s3p]
            return SADAtomStructure(
                N=N,
                Ls=[0, 0, 1, 0, 1],
                actives=[False]*3 + [True]*2,
                )
        elif N >= 19 and N <= 20:
            # K  - Ca: [1s2s2p3s3p][4s]
            return SADAtomStructure(
                N=N,
                Ls=[0, 0, 1, 0, 1, 0],
                actives=[False]*5 + [True]*1,
                )
        elif N >= 21 and N <= 30:
            # Sc - Zn: [1s2s2p3s3p4s][3d]
            return SADAtomStructure(
                N=N,
                Ls=[0, 0, 1, 0, 1, 0, 2],
                actives=[False]*6 + [True]*1,
                )
        elif N >= 31 and N <= 36:
            # Ga - Kr: [1s2s2p3s3p3d][4s4p]
            return SADAtomStructure(
                N=N,
                Ls=[0, 0, 1, 0, 1, 2, 0, 1],
                actives=[False]*6 + [True]*2,
                )
        # NOTE: We have the definitions for these down to Z=118, but have not
        # had time to format them - please reach out to cuEST product team if
        # you need these defined (note that you likely also need ECPs if this
        # deep)
        else:
            raise RuntimeError('Unknown SADAtomStructure for atomic number N = %d' % N)
