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

from .periodic_table import PeriodicTable

from .ao_shell import AOShell

from .memoized_property import memoized_property
    
class AOBasis(object):

    def __init__(   
        self,
        *,
        shells : list,
        ):

        self.shells = shells

        if not all(isinstance(_, list) for _ in shells): raise RuntimeError('shells is not list of lists')
        if not all(isinstance(_, AOShell) for _ in self.shells_unrolled): raise RuntimeError('shells is not list of list of AOShell')

    @memoized_property
    def is_pure(self):
        return all(_.is_pure for _ in self.shells_unrolled)

    @memoized_property
    def is_cart(self):
        return all(not _.is_pure for _ in self.shells_unrolled)

    @memoized_property
    def is_mixed(self):
        return (not self.is_pure) and (not self.is_cart)

    @memoized_property
    def natom(self):
        return len(self.shells)

    @memoized_property
    def nshell(self):
        return len(self.shells_unrolled)

    @memoized_property
    def nao(self):
        return sum(_.nao for _ in self.shells_unrolled)

    @memoized_property
    def ncart(self):
        return sum(_.ncart for _ in self.shells_unrolled)

    @memoized_property
    def nprimitive(self):
        return sum(_.nprimitive for _ in self.shells_unrolled)

    @memoized_property
    def max_L(self):
        return max(_.L for _ in self.shells_unrolled) if self.nshell else 0

    @memoized_property
    def shells_unrolled(self):
        return [item for sublist in self.shells for item in sublist]

    # => Indexing <= #

    @memoized_property
    def shell_ao_starts(self):
        if self.nshell == 0: return []
        return [0] + list(np.cumsum([_.nao for _ in self.shells_unrolled]))[:-1]

    @memoized_property
    def shell_cart_starts(self):
        if self.nshell == 0: return []
        return [0] + list(np.cumsum([_.ncart for _ in self.shells_unrolled]))[:-1]

    @memoized_property
    def shell_primitive_starts(self):
        if self.nshell == 0: return []
        return [0] + list(np.cumsum([_.nprimitive for _ in self.shells_unrolled]))[:-1]

    @memoized_property
    def shell_atoms(self):
        return [item for sublist in [[A]*len(shells2) for A, shells2 in enumerate(self.shells)] for item in sublist]

    @memoized_property
    def shell_atom_shells(self):
        return [item for sublist in [list(range(len(shells2))) for shells2 in self.shells] for item in sublist]

    def update_to_pure(self):
        return AOBasis(shells=[[shell.update_to_pure() for shell in shells2] for shells2 in self.shells])

    def update_to_cart(self):
        return AOBasis(shells=[[shell.update_to_cart() for shell in shells2] for shells2 in self.shells])

    def clone(self):
        shells = []    
        for shells2 in self.shells:
            shells.append([_.clone() for _ in shells2])
        return AOBasis(shells=shells)

    @staticmethod
    def concatenate(bases):
        shells = []
        for basis in bases:
            shells += basis.clone().shells
        return AOBasis(shells=shells)

    def subset(self, indices):
        return AOBasis(shells=[[shell.clone() for shell in self.shells[A]] for A in indices])

    def atomize(self):
        return [self.subset(indices=[A]) for A in range(self.natom)]

    def __eq__(self, other):
        if not isinstance(other, AOBasis):
            return NotImplemented
        return self.shells_unrolled == other.shells_unrolled 

    def __ne__(self, other):
        if not isinstance(other, AOBasis):
            return NotImplemented
        return self.shells_unrolled != other.shells_unrolled 

    # => String Details <= #

    def __str__(self):
        return self.string

    @property
    def string(self):
        s = ''
        s += 'AOBasis:\n';
        s += '%-10s = %6d\n' % ('natom', self.natom)
        s += '%-10s = %6d\n' % ('nshell', self.nshell)
        s += '%-10s = %6d\n' % ('nao', self.nao)
        s += '%-10s = %6d\n' % ('ncart', self.ncart)
        s += '%-10s = %6d\n' % ('nprimitive', self.nprimitive)
        s += '%-10s = %6d\n' % ('max_L', self.max_L)
        s += '%-10s = %6s\n' % ('is_pure', self.is_pure)
        s += '%-10s = %6s\n' % ('is_cart', self.is_cart)
        s += '%-10s = %6s\n' % ('is_mixed', self.is_mixed)
        return s;

    @property
    def detail_string(self):
        s = ''
        s += '%-5s : %7s %2s %5s %5s %5s %10s %10s %10s %10s %10s\n' % (
            'shell',
            'is_pure',
            'L',
            'nao',
            'ncart',
            'nprim',   
            'ao_start',
            'cart_start',
            'prim_start',
            'atom',
            'atom_shell',
            )
        P = 0
        for A in range(self.natom):
            for P2 in range(len(self.shells[A])):
                shell = self.shells[A][P2]
                s += '%-5d : %7s %2d %5d %5d %5d %10d %10d %10d %10d %10d\n' % (
                    P,
                    shell.is_pure,
                    shell.L,
                    shell.nao,
                    shell.ncart,
                    shell.nprimitive,
                    self.shell_ao_starts[P],
                    self.shell_cart_starts[P],
                    self.shell_primitive_starts[P],
                    self.shell_atoms[P],
                    self.shell_atom_shells[P],
                    )
                P += 1
        return s

    # => GBS File AOBasis Parsing <= #
    
    angular_momentum_table = {
        'S' :  0, 
        'P' :  1, 
        'D' :  2, 
        'F' :  3, 
        'G' :  4, 
        'H' :  5, 
        'I' :  6, 
        'K' :  7, 
        'L' :  8, 
        'M' :  9, 
        'N' : 10,
        'O' : 11,
        'Q' : 12,
        'R' : 13,
        'T' : 14,
        'U' : 15,
        'V' : 16,
        'W' : 17,
        'X' : 18,
        'Y' : 19,
        'Z' : 20,
        }

    @staticmethod
    def parse_from_gbs_lines(
        *,
        lines,
        molecule,
        ):
    
        import re
    
        # => Cleaning <= #
    
        # Strip blank lines out
        lines = [_ for _ in lines if len(_.strip())]
        # Strip comment lines out
        re_comment = re.compile(r'\s*!')
        lines = [_ for _ in lines if not re.match(re_comment, _)]
        # Lines must be nonzero 
        if len(lines) == 0: raise RuntimeError('GBS lines are blank')
    
        # => Spherical / Cartesian Tag Line <= #
    
        mobj = re.match(r'^\s*(spherical|cartesian)\s*$', lines[0])
        if mobj is None:
            raise RuntimeError("First line of GBS file must be 'spherical' or 'cartesian', instead is: %s" % lines[0])
    
        if mobj.group(1) == 'spherical':
            is_pure = True
        elif mobj.group(1) == 'cartesian':
            is_pure = False
        else:
            raise RuntimeError('Invalid cartesian/spherical label: %s' % mobj.group(1))
    
        lines = lines[1:]
    
        # => Atom Block Location <= #
    
        re_separator = re.compile(r'\s*\*\*\*\*\s*$')
        separator_indices = [k for k, line in enumerate(lines) if re.match(re_separator, line)]
        if len(separator_indices) == 0: 
            raise RuntimeError('No **** separators present')
        if separator_indices[-1] + 1 != len(lines):
            raise RuntimeError('Last line must be ****, instead is: %s' % lines[-1])
        lines = lines[:-1]
        separator_indices = separator_indices[:-1]
    
        if len(lines) == 0: raise RuntimeError('GBS lines are blank')
        if len(separator_indices) == 0: raise RuntimeError('No **** separators present')
    
        # => Atom IDs and corresponding block index <= #
    
        re_atom_id = re.compile(r'^\s*(\S+)\s+(\d+)\s*$')
        atom_id_lines = [lines[_ + 1] for _ in separator_indices]
        N_index_map = {}
        for index, atom_id_line in enumerate(atom_id_lines):
            mobj = re.match(re_atom_id, atom_id_line)
            if mobj is None:
                raise RuntimeError('Malformed atom ID line: %s' % (atom_id_line))
            symbol = mobj.group(1)
            atom_index = int(mobj.group(2))
            if atom_index != 0: 
                raise RuntimeError('"Symbol 0" is only allowed atom line - multiple basis sets per atom type in GBS files is not supported by this library')
            symbol_upper = symbol.upper()
            if symbol_upper not in PeriodicTable.symbol_upper_to_N_table:
                raise RuntimeError('Unknown atomic in GBS file symbol: %s' % symbol)
            N = PeriodicTable.symbol_upper_to_N_table[symbol_upper]
            if N in N_index_map:
                raise RuntimeError('Duplicate atomic symbol in GBS file: %s' % symbol)
            N_index_map[N] = index
    
        # => Unique atoms in molecule (will parse only these) <= #
    
        unique_N = list(set(molecule.N[...]))
    
        re_shell_type = re.compile(r'^\s*(\S+)\s+(\d+)\s+(\S+)\s*$')
        re_primitive = re.compile(r'^\s*(\S+)\s+(\S+)\s*$')
    
        N_to_shells_map = {}
        for N in unique_N:
            if N not in N_index_map:
                raise RuntimeError('N not in GBS file: %d' % (N))
            block_index = N_index_map[N]
            index1 = separator_indices[block_index + 0]
            index2 = separator_indices[block_index + 1] if block_index + 1 < len(separator_indices) else len(lines)
            
            block_lines = lines[index1+2:index2]
            shell_type_indices = [k for k, line in enumerate(block_lines) if re.match(re_shell_type, line)]
    
            shells = []
            for index in shell_type_indices:
                shell_type_line = block_lines[index]
                mobj = re.match(re_shell_type, shell_type_line)
                am_symbol_upper = mobj.group(1).upper()
                if am_symbol_upper not in AOBasis.angular_momentum_table:
                    raise RuntimeError('Unknown angular momentum symbol: %s' % am_symbol_upper)
                L = AOBasis.angular_momentum_table[am_symbol_upper]
                nprimitive = int(mobj.group(2))
                normalization = float(mobj.group(3))
                exponents = []
                coefficients_raw = []
                for K in range(nprimitive):
                    primitive_line = block_lines[index + 1 + K]
                    # Hack to replace D with E for FORTRAN notation
                    primitive_line = primitive_line.replace('D', 'E')
                    primitive_line = primitive_line.replace('d', 'e')
                    mobj = re.match(re_primitive, primitive_line)
                    exponents += [float(mobj.group(1))]
                    coefficients_raw += [float(mobj.group(2))]
                exponents = np.array(exponents)
                coefficients_raw = np.array(coefficients_raw)
                shells.append(AOShell.build_from_raw_data(
                    is_pure=is_pure,
                    L=L,
                    exponents=exponents,
                    coefficients_raw=coefficients_raw,
                    normalization=normalization,
                    ))
            N_to_shells_map[N] = shells
    
        # => AOBasis <= #
    
        shells = []
        for N in molecule.N:
            shells.append(N_to_shells_map[N])
    
        return AOBasis(
            shells=shells,
            )
    
    @staticmethod
    def parse_from_gbs_string(
        string,
        *,
        molecule,
        **kwargs):
        
        return AOBasis.parse_from_gbs_lines(lines=string.split('\n'), molecule=molecule, **kwargs) 
    
    @staticmethod
    def parse_from_gbs_file(
        filename,
        *,
        molecule,
        **kwargs):
    
        with open(filename, 'r') as fh:
            lines = fh.read().splitlines()
    
        return AOBasis.parse_from_gbs_lines(
            lines=lines,
            molecule=molecule,
            **kwargs)
