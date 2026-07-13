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

from .periodic_table import PeriodicTable
from .ecp_atom import ECPAtom

from .memoized_property import memoized_property
    
class ECPBasis(object):

    def __init__(   
        self,
        *,
        atoms : list,
        ):

        self.atoms = atoms

    @memoized_property
    def natom(self):
        return len(self.atoms)

    @memoized_property
    def nelectron(self):
        return sum(atom.nelectron for atom in self.atoms)

    @memoized_property
    def is_active(self):
        return any(atom.is_active for atom in self.atoms)

    def __str__(self):
        return self.string

    @property
    def string(self):
        s = ''
        s += 'ECPBasis:\n'
        s += '%-10s = %6d\n' % ('natom', self.natom)
        s += '%-10s = %6d\n' % ('ecp_natom', sum(atom.is_active for atom in self.atoms))
        s += '%-10s = %6d\n' % ('nelectron', self.nelectron)
        return s

    # => GBS File ECPBasis Parsing <= #

    @staticmethod
    def parse_from_gbs_string(
        string,
        *,
        molecule,
        ):

        import re

        lines = string.split('\n')

        lines2 = []
        for line in lines:
            if re.match(r'^\s*$', line):  # Remove blank lines
                continue
            if re.match(r'^\s*!', line):  # Remove comment lines
                continue
            lines2.append(line)
    

        # Find ECP entries matching this kind of pattern
        # SI     0
        # SI-ECP     2     10
        # But joined onto a single line
        ecp_re = re.compile(r'^\s*(\S+)\s+(\d+)\s*(\S+)-ECP\s+(\d+)\s+(\d+)\s*$')
        ecp_inds = []
        for ind in range(len(lines2) - 1):
            line = lines2[ind] + lines2[ind + 1]  # Join into a single line
            if ecp_re.match(line):
                ecp_inds.append(ind)
        ecp_inds.append(len(lines2))

        # Extract the lines pertaining to each atom symbol
        atom_ecp = {}
        for k in range(len(ecp_inds) - 1):
            ind1 = ecp_inds[k]
            ind2 = ecp_inds[k + 1]
            if (ind2 - ind1) <= 0:  # Guard against empty entries
                continue
            mobj = re.match(
                r'^\s*(\S+)\s+(\d+)\s*$', lines2[ind1]
            )  # Check if the line is a regular basis entry or an ECP entry
            if mobj is None:
                raise RuntimeError("Malformed ECP entry: expected ID line")
            if mobj.group(2) != '0':
                continue  # Regular basis entry, not an ECP entry
            # Try to match the next line to something of the form
            # SR-ECP     3     28
            mobj = re.match(r'^\s*(\S+)-ECP\s+(\d+)\s+(\d+)\s*$', lines2[ind1 + 1])
            if mobj is None:
                # This is a regular atom entry, not an ECP entry
                continue
            atom_ecp[mobj.group(1).upper()] = ECPAtom.parse_from_ecp_lines(lines2[ind1 + 1 : ind2])

        atoms = []
        for N, Z in zip(molecule.N, molecule.Z, strict=True):
            N_int = int(N)
            Z_float = float(Z)
            sym = PeriodicTable.N_to_symbol_upper_table.get(N_int, 'X')
            atoms.append(
                atom_ecp.get(sym, ECPAtom.create_inactive_atom())
                if Z_float != 0.0
                else ECPAtom.create_inactive_atom()
            )

        return ECPBasis(atoms=atoms)
    
    @staticmethod
    def parse_from_gbs_file(
        filename,
        *,
        molecule):
    
        with open(filename, 'r') as fh:
            string = fh.read()
    
        return ECPBasis.parse_from_gbs_string(
            string=string,
            molecule=molecule)
