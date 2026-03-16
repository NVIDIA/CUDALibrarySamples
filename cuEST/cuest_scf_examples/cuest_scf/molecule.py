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

from .unit_conversions import UnitConversions
from .periodic_table import PeriodicTable

class Molecule(object):

    def __init__(
        self,
        *,
        N : np.ndarray,
        Z : np.ndarray,
        xyz : np.ndarray,
        ):

        self.N = N
        self.Z = Z
        self.xyz = xyz

        if self.N.shape != (self.natom,): raise RuntimeError('N.shape is not (natom,)')
        if self.Z.shape != (self.natom,): raise RuntimeError('Z.shape is not (natom,)')
        if self.xyz.shape != (self.natom, 3): raise RuntimeError('xyz.shape is not (natom, 3)')

        if self.N.dtype != np.uint64: raise RuntimeError('N.dtype is not np.uint64')
        if self.Z.dtype != np.float64: raise RuntimeError('Z.dtype is not np.float64')
        if self.xyz.dtype != np.float64: raise RuntimeError('xyz.dtype is not np.float64')

    @property
    def natom(self):
        return len(self.N)

    @staticmethod
    def parse_from_xyz_lines(
        lines,
        *,
        bohr_per_ang=UnitConversions.conversions['bohr_per_ang'],
        ):
    
        # Strip fringe blank lines (but not internal blank lines)
        lines = '\n'.join(lines).strip().split('\n')
    
        if len(lines) == 0:
            return Molecule(
                N=np.zeros((0,), dtype=np.uint64),
                Z=np.zeros((0,)),
                xyz=np.zeros((0, 3)),
                )
    
        import re
            
        # Natom line 
        # (optional, but we enforce natom and length of file if specified)
        mobj = re.match(r'^\s*(\d+)\s*$', lines[0])
        if mobj:
            natom = int(mobj.group(1))
            lines = lines[2:]
            if len(lines) != natom:
                raise RuntimeError('natom line specified %d atoms, but %d lines remain in XYZ file' % (natom, len(lines)))
    
        re_atom_line = re.compile(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$')

        Ns = []
        Zs = []
        xyzs = []
    
        for lindex, line in enumerate(lines):
            mobj = re.match(re_atom_line, line)
            if mobj is None:
                raise RuntimeError('Malformed xyz atom line #%d: %s' % (lindex, line))
            
            symbol = mobj.group(1)
            x = float(mobj.group(2))
            y = float(mobj.group(3))
            z = float(mobj.group(4))
            
            x *= bohr_per_ang
            y *= bohr_per_ang
            z *= bohr_per_ang
    
            # NOTE: This does not handle extravagant symbol names like "Gh(He)" or "OXT" or "H23"
            symbol_upper = symbol.upper()
    
            if symbol_upper not in PeriodicTable.symbol_upper_to_N_table:
                raise RuntimeError('Symbol %s is not in periodic table: %s' % (symbol_upper, line))
    
            N = PeriodicTable.symbol_upper_to_N_table[symbol_upper] 
            Z = float(N)
            
            Ns.append(N)
            Zs.append(Z)
            xyzs.append((x, y, z))
        
        Ns = np.array(Ns, dtype=np.uint64)
        Zs = np.array(Zs)
        xyzs = np.array(xyzs)

        return Molecule(
            N=Ns,
            Z=Zs,
            xyz=xyzs,
            )
        
    @staticmethod
    def parse_from_xyz_string(
        string,
        **kwargs):
    
        return Molecule.parse_from_xyz_lines(lines=string.split('\n'), **kwargs)
        
    @staticmethod
    def parse_from_xyz_file(
        filename,
        **kwargs):
    
        with open(filename, 'r') as fh:
            lines = fh.read().splitlines()
    
        return Molecule.parse_from_xyz_lines(lines=lines, **kwargs)
