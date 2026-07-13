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
    
class SADAtomStructure(object):

    def __init__(
        self,
        *,
        N : int,
        Ls : list,
        valence_Ls : list,
        actives : list,
        core : list,
        ):

        self.N = N
        self.Ls = Ls
        self.valence_Ls = valence_Ls
        self.actives = actives
        self.core = core

        if not all(isinstance(_, int) for _ in Ls): raise RuntimeError('Ls is not list of int')
        if not all(isinstance(_, int) for _ in valence_Ls): raise RuntimeError('valence_Ls is not list of int')
        if not all(isinstance(_, bool) for _ in actives): raise RuntimeError('actives is not list of bool')
        if list(sorted(actives)) != actives: raise RuntimeError('actives is not sorted')
        if not all(isinstance(_, bool) for _ in core): raise RuntimeError('core is not list of bool')
        if list(sorted(core, reverse=True)) != core: raise RuntimeError('core is not sorted')

        if self.N > 0 and self.nao_active == 0:
            raise RuntimeError('N > 0 and nao_active == 0')

    @property
    def nshell(self):
        return len(self.Ls)

    @property
    def nshell_valence(self):
        return len(self.valence_Ls)

    @property
    def max_L(self):
        return max(self.Ls) if self.nshell else 0
        
    @property
    def max_L_valence(self):
        return max(self.valence_Ls) if self.nshell_valence else 0
        
    @property
    def nao_inactive(self):
        return sum(2*L + 1 for L, active in zip(self.valence_Ls, self.actives) if not active)

    @property
    def nao_active(self):
        return sum(2*L + 1 for L, active in zip(self.valence_Ls, self.actives) if active)

    @property
    def nao_core(self):
        return sum(2*L + 1 for L, is_core in zip(self.Ls, self.core) if is_core)

    @property
    def nao_valence(self):
        return self.nao_inactive + self.nao_active

    @property
    def nao(self):
        return self.nao_core + self.nao_inactive + self.nao_active

    """
    Works in terms of RHF docc
    """
    @property
    def nocc(self):
        if self.nao_valence == 0:
            return np.array([])
        nocc_total = (self.N / 2) - self.nao_core
        if self.nao_inactive > nocc_total:
            raise RuntimeError('nao_inactive > nocc_total. N = %d' % self.N)
        if self.nao_valence < nocc_total:
            raise RuntimeError('nao_valence < nocc_total. N = %d' % self.N)
        nocc_active = nocc_total - self.nao_inactive
        n = nocc_active / self.nao_active
        return np.array(
            [1.0] * self.nao_inactive +
            [n] * self.nao_active)

    @property
    def string(self):
        s = 'SADAtomStructure: N = %d\n' % (self.N)
        idx = 0
        for P in range(self.nshell):
            if self.core[P]:
                s += '%1d : %s\n' % (self.Ls[P], 'core')
            else:
                s += '%1d : %s\n' % (self.Ls[P], 'active' if self.actives[idx] else 'inactive')
                idx = idx+1
        return s

    @property
    def aufbau_string(self):
        symbols = ['s', 'p', 'd', 'f']
        counts  = [ 1,   2,   3,   4 ]
        if self.max_L >= len(symbols): raise RuntimeError('max_L too high for symbol table')

        s = ''
        s += '['
        if any(self.core):
            for P in range(self.nshell):
                if self.core[P]:
                    L = self.Ls[P]        
                    s += str(counts[L])
                    s += symbols[L]
                    counts[L] = counts[L] + 1
        s += ']['
        if self.nao_inactive > 0:
            idx = 0
            for P in range(self.nshell):
                if not self.core[P]:
                    if not self.actives[idx]:
                        L = self.Ls[P]        
                        s += str(counts[L])
                        s += symbols[L]
                        counts[L] = counts[L] + 1
                    idx = idx + 1
        s += ']['
        if self.nao_active > 0:
            idx = 0
            for P in range(self.nshell):
                if not self.core[P]:
                    if self.actives[idx]:
                        L = self.Ls[P]        
                        s += str(counts[L])
                        s += symbols[L]
                        counts[L] = counts[L] + 1
                    idx = idx + 1
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
    def find_ecp_shells(Ls, core_Ls, actives):
        core = [False] * len(Ls)

        idx = 0
        for core_L in core_Ls:
            matched = False
            for j, L in enumerate(Ls[idx:]):
                if core_L == L:
                    core[idx + j] = True
                    idx = idx + j + 1
                    matched = True
                    break
            if not matched:
                raise RuntimeError(f'Could not find core shell L={core_L} in remaining shells {Ls[idx:]}')

        valence_Ls = [L for L, is_core in zip(Ls, core) if not is_core]
        valence_actives = [active for active, is_core in zip(actives, core) if not is_core]

        return core, valence_Ls, valence_actives

    @staticmethod
    def build(N, Necp=0):

        # This is mostly correct. There might be occasional edge cases
        # where the ECPs model a different configuration. In a production
        # code, it might be better to define these separately for each of 
        # the ECP definitions you wish to support.
        if Necp == 0:
            core_Ls = []
        elif Necp == 2:
            core_Ls = [0]
        elif Necp == 10:
            core_Ls = [0, 0, 1]
        elif Necp == 18:
            core_Ls = [0, 0, 1, 0, 1]
        elif Necp == 28:
            core_Ls = [0, 0, 1, 0, 1, 2]
        elif Necp == 36:
            core_Ls = [0, 0, 1, 0, 1, 2, 0, 1]
        elif Necp == 46:
            core_Ls = [0, 0, 1, 0, 1, 2, 0, 1, 2]
        elif Necp == 54:
            core_Ls = [0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1]
        elif Necp == 60:
            core_Ls = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
        elif Necp == 68:
            core_Ls = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1]
        else:
            raise RuntimeError('Unknown ECP configuration')

        if N == 0:
            # Gh [][]
            Ls=[]
            actives=[]
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 1 and N <= 2:
            # H  - He: [][1s]
            Ls=[0]
            actives=[True]
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 3 and N <= 4:
            # Li - Be: [1s][2s]
            Ls=[0, 0]
            actives=[False]*1 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 5 and N <= 10:
            # B  - Ne: [1s][2s2p]
            Ls=[0, 0, 1]
            actives=[False]*1 + [True]*2
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 11 and N <= 12:
            # Na - Mg: [1s2s2p][3s]
            Ls=[0, 0, 1, 0]
            actives=[False]*3 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 13 and N <= 18:
            # Al - Ar: [1s2s2p][3s3p]
            Ls=[0, 0, 1, 0, 1]
            actives=[False]*3 + [True]*2
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 19 and N <= 20:
            # K  - Ca: [1s2s2p3s3p][4s]
            Ls=[0, 0, 1, 0, 1, 0]
            actives=[False]*5 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 21 and N <= 30:
            # Sc - Zn: [1s2s2p3s3p4s][3d]
            Ls=[0, 0, 1, 0, 1, 0, 2]
            actives=[False]*6 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 31 and N <= 36:
            # Ga - Kr: [1s2s2p3s3p3d][4s4p]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1]
            actives=[False]*6 + [True]*2
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 37 and N <= 38:
            # Rb - Sr: [1s2s2p3s3p3d4s4p][5s]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 0]
            actives=[False]*8 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 39 and N <= 48:
            # Y  - Cd: [1s2s2p3s3p3d4s4p5s][4d]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 0, 2]
            actives=[False]*9 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 49 and N <= 54:
            # In - Xe: [1s2s2p3s3p3d4s4p4d][5s5p]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1]
            actives=[False]*9 + [True]*2
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 55 and N <= 56:
            # Cs - Ba: [1s2s2p3s3p3d4s4p4d5s5p][6s]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0]
            actives=[False]*11 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 57 and N <= 70:
            # La - Yb: [1s2s2p3s3p3d4s4p4d5s5p6s][4f]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 0, 3]
            actives=[False]*12 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 71 and N <= 80:
            # Lu - Hg: [1s2s2p3s3p3d4s4p4d4f5s5p6s][5d]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 2]
            actives=[False]*13 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 81 and N <= 86:
            # Tl - Rn: [1s2s2p3s3p3d4s4p4d4f5s5p5d][6s6p]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1]
            actives=[False]*13 + [True]*2
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 87 and N <= 88:
            # Fr - Ra: [1s2s2p3s3p3d4s4p4d4f5s5p5d6s6p][7s]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0]
            actives=[False]*15 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 89 and N <= 102:
            # Ac - No: [1s2s2p3s3p3d4s4p4d4f5s5p5d6s6p7s][5f]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 3]
            actives=[False]*16 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 103 and N <= 112:
            # Lr - Cn: [1s2s2p3s3p3d4s4p4d4f5s5p5d5f6s6p7s][6d]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 2]
            actives=[False]*17 + [True]*1
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N >= 113 and N <= 118:
            # Nh - Og: [1s2s2p3s3p3d4s4p4d4f5s5p5d5f6s6p6d][7s7p]
            Ls=[0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1]
            actives=[False]*17 + [True]*2
            core, valence_Ls, actives = SADAtomStructure.find_ecp_shells(Ls, core_Ls, actives)
            return SADAtomStructure(
                N=N,
                Ls=Ls,
                valence_Ls=valence_Ls,
                actives=actives,
                core=core,
                )
        elif N == 119: 
            raise RuntimeError('You have discovered Ununennium! Let cuest product know to add element %d' % N)
        else:
            raise RuntimeError('Unknown SADAtomStructure for atomic number N = %d' % N)
