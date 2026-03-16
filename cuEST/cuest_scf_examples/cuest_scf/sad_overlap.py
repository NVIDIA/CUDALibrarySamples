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

from .sad_solid_harmonics import SADSolidHarmonics

class SADOverlap(object):

    @staticmethod
    def compute_atomic_overlap(
        *,
        basis1 : AOBasis,
        basis2 : AOBasis,
        ): 

        if basis1.natom != 1: raise RuntimeError('basis1.natom != 1')
        if basis2.natom != 1: raise RuntimeError('basis2.natom != 1')

        if basis1.is_mixed: raise RuntimeError('basis1 is mixed')
        if basis2.is_mixed: raise RuntimeError('basis2 is mixed')

        df = [1.0]
        for L in range(2, basis1.max_L + basis2.max_L + 1, 2):
            df.append(df[L // 2 - 1] * (L - 1.0))

        Scc = np.zeros((basis1.ncart, basis2.ncart))

        for P1, shell1 in enumerate(basis1.shells[0]):
            L1 = shell1.L
            offset1 = basis1.shell_cart_starts[P1]

            for P2, shell2 in enumerate(basis2.shells[0]):
                L2 = shell2.L
                offset2 = basis2.shell_cart_starts[P2]

                for e1, c1 in zip(shell1.exponents, shell1.coefficients):
                    for e2, c2 in zip(shell2.exponents, shell2.coefficients):

                        c12 = c1 * c2
                        e12 = e1 + e2

                        S0 = c12 * (np.pi / e12)**(1.5) * (2 * e12)**(-0.5 * (L1 + L2))

                        index1 = 0
                        for i1 in range(L1+1):
                            l1 = L1 - i1
                            for j1 in range(i1+1):
                                m1 = i1 - j1
                                n1 = j1 

                                index2 = 0
                                for i2 in range(L2+1):
                                    l2 = L2 - i2
                                    for j2 in range(i2+1):
                                        m2 = i2 - j2
                                        n2 = j2 

                                        l12 = l1 + l2                                        
                                        m12 = m1 + m2                                        
                                        n12 = n1 + n2                                        

                                        if l12 % 2 == 1: 
                                            index2 += 1
                                            continue
                                        if m12 % 2 == 1: 
                                            index2 += 1
                                            continue
                                        if n12 % 2 == 1: 
                                            index2 += 1
                                            continue

                                        S = S0 * df[l12 // 2] * df[m12 // 2] * df[n12 // 2]

                                        cart1 = offset1 + index1
                                        cart2 = offset2 + index2
                    
                                        Scc[cart1, cart2] += S

                                        index2 += 1
                                
                                index1 += 1

        if basis1.is_cart and basis2.is_cart:
            return Scc

        max_L = 0
        if basis1.is_pure: max_L = max(max_L, basis1.max_L)
        if basis2.is_pure: max_L = max(max_L, basis2.max_L)
        sh = SADSolidHarmonics(max_L=max_L)

        if basis2.is_pure:
            Scs = np.zeros((basis1.ncart, basis2.nao))
            for P2, shell2 in enumerate(basis2.shells[0]):
                L2 = shell2.L
                ncart2 = (L2 + 1) * (L2 + 2) // 2
                npure2 = 2 * L2 + 1
                cart_offset2 = basis2.shell_cart_starts[P2]
                pure_offset2 = basis2.shell_ao_starts[P2]
                T = sh.T[L2]
                Scs[:, pure_offset2:pure_offset2+npure2] = np.dot(Scc[:, cart_offset2:cart_offset2+ncart2], T)
        else:
            Scs = Scc

        if basis1.is_pure:
            Sss = np.zeros((basis1.nao, basis2.nao))
            for P1, shell1 in enumerate(basis1.shells[0]):
                L1 = shell1.L
                ncart1 = (L1 + 1) * (L1 + 2) // 2
                npure1 = 2 * L1 + 1
                cart_offset1 = basis1.shell_cart_starts[P1]
                pure_offset1 = basis1.shell_ao_starts[P1]
                T = sh.T[L1]
                Sss[pure_offset1:pure_offset1+npure1, :] = np.dot(T.T, Scs[cart_offset1:cart_offset1+ncart1, :])
        else:
            Sss = Scs

        return Sss
