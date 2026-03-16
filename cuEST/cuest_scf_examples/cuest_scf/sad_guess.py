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

from .molecule import Molecule    
from .ao_basis import AOBasis
from .sad_atom_structure import SADAtomStructure
from .sad_guess_atom import SADGuessAtom

from .memoized_property import memoized_property

class SADGuess(object):

    def __init__(
        self,
        *,
        sad_guess_atoms : list,
        ):

        self.sad_guess_atoms = sad_guess_atoms

        if not all(isinstance(_, SADGuessAtom) for _ in sad_guess_atoms): raise RuntimeError('sad_guess_atoms is not list of SADGuessAtom')

    @property
    def natom(self):
        return len(self.sad_guess_atoms)

    @memoized_property
    def primary(self):
        return AOBasis.concatenate(_.primary for _ in self.sad_guess_atoms)

    @memoized_property
    def minao(self):
        return AOBasis.concatenate(_.minao for _ in self.sad_guess_atoms)

    def compute_Cocc(self):
        Cocc = np.zeros((self.minao.nao, self.primary.nao))
        primary_start = 0
        minao_start = 0
        for A, sad_guess_atom in enumerate(self.sad_guess_atoms):
            nocc2 = sad_guess_atom.structure.nocc
            Cocc2 = sad_guess_atom.C
            Focc2 = np.einsum('i,ip->ip', np.sqrt(nocc2), Cocc2)
            nprimary = sad_guess_atom.primary.nao
            nminao = sad_guess_atom.minao.nao
            Cocc[minao_start:minao_start+nminao, primary_start:primary_start+nprimary] = Focc2
            primary_start += nprimary
            minao_start += nminao
        return Cocc

    def compute_Docc(self):
        Cocc = self.compute_Cocc()
        return np.dot(Cocc.T, Cocc)

    @staticmethod
    def build(
        *,
        molecule : Molecule,
        primary : AOBasis,
        minao : AOBasis,
        ):

        if primary.natom != molecule.natom: raise RuntimeError('primary.natom != molecule.natom')
        if minao.natom != molecule.natom: raise RuntimeError('minao.natom != molecule.natom')

        unique_sad_guess_atoms = {}

        structures = [SADAtomStructure.build(N=N) for N in molecule.N]
        primary_atoms = primary.atomize()
        minao_atoms = minao.atomize()

        for N, structure, primary_atom, minao_atom in zip(molecule.N, structures, primary_atoms, minao_atoms):
            if N not in unique_sad_guess_atoms:
                minao_atom2 = SADGuess.reorder_minao(
                    minao=minao_atom,
                    structure=structure,
                    )
                unique_sad_guess_atoms[N] = SADGuessAtom(
                    primary=primary_atom,
                    minao=minao_atom2,
                    structure=structure,
                    )

        return SADGuess(
            sad_guess_atoms=[unique_sad_guess_atoms[N] for N in molecule.N],
            )
             
    """
    Sometimes a GBS file is ordered in its own way, which does not correspond
    to the order of inactive and active shells in SADAtomStructure. E.g., an
    element like S might have its MINAO basis file ordered as 1s2s3s2p3p but
    the SADAtomStructure order is [1s2s2p][3s3p]. This function provides a sort
    of sheep and goats shuffle to transform a 1-atom AOBasis object to another
    1-atom AOBasis object with the order of structure, if this is topologically
    possible.
    """
    @staticmethod
    def reorder_minao(
        *,
        minao : AOBasis,
        structure : SADAtomStructure,
        ):

        if minao.natom != 1: raise RuntimeError('minao.natom != 1')

        if list(sorted(shell.L for shell in minao.shells[0])) != list(sorted(structure.Ls)):
            raise RuntimeError('minao and structure do not have topological agreement on L structure. N = %d' % (structure.N))

        back_map = {}
        Ns = [0 for L in range(minao.max_L+1)]      
        index = 0
        for shell in minao.shells[0]:
            L = shell.L
            N = Ns[L]
            back_map[N, L] = index
            Ns[L] += 1
            index += 1

        shells = []
        Ns = [0 for L in range(minao.max_L+1)]      
        for L in structure.Ls:
            N = Ns[L]
            index = back_map[N, L]
            shells.append(minao.shells[0][index].clone())
            Ns[L] += 1

        return AOBasis(shells=[shells])
