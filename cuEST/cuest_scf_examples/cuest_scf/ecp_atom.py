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
from .ecp_shell import ECPShell
import numpy as np
    
class ECPAtom(object):

    def __init__(
        self,
        *,
        nelectron : int, 
        shells : list,
        top_shell : ECPShell,
        is_active : bool,
        ):

        self.nelectron = nelectron
        self.shells = shells
        self.top_shell = top_shell
        self.is_active = is_active

    # Shell name to L
    shell_data = {
        'S': 0,
        'P': 1,
        'D': 2,
        'F': 3,
        'G': 4,
        'H': 5,
        'I': 6,
        'K': 7,
        'L': 8,
        'M': 9,
        'N': 10,
        'O': 11,
        'Q': 12,
        'R': 13,
        'T': 14,
        'U': 15,
        'V': 16,
        'W': 17,
        'X': 18,
        'Y': 19,
        'Z': 20,
    }

    @staticmethod
    def create_inactive_atom():
        return ECPAtom(
            nelectron=0,
            shells=[],
            top_shell=ECPShell.create_inactive_shell(),
            is_active=False,
        )


    @staticmethod
    def parse_from_ecp_lines(
        lines,
        ):

        import re

        # Remove comment lines
        lines2 = []
        for line in lines:
            if re.match(r'^\s*$', line):  # Remove blank lines
                continue
            if re.match(r'^\s*!', line):  # Remove comment lines
                continue
            lines2.append(line)

        mobj = re.match(r'^\s*(\S+)-ECP\s+(\d+)\s+(\d+)\s*$', lines2[0], re.IGNORECASE)
        if mobj is None:
            raise RuntimeError('Where is the "ATOM-ECP max_L nelectron" line?')
        max_L = int(mobj.group(2))
        nelectron = int(mobj.group(3))

        lines2 = lines2[1:]

        potential_indices = [
            index
            for index, line in enumerate(lines2)
            if re.match(r'^\s*(\S+)\s+potential\s*$', line, re.IGNORECASE)
        ]

        if len(potential_indices) < 1:
            raise RuntimeError('len(potential_indices) < 1')

        potential_indices.append(len(lines2))

        # Top shell

        index = 0

        potential_start = potential_indices[index]
        potential_stop = potential_indices[index + 1]

        mobj = re.match(r'^\s*(\S)\s+potential\s*$', lines2[potential_start], re.IGNORECASE)
        if mobj is None:
            raise RuntimeError('Where is the "max_L potential" line?')

        shell_char = mobj.group(1).upper()
        if shell_char not in ECPAtom.shell_data:
            raise RuntimeError(f'Unknown shell type: {shell_char}')

        if ECPAtom.shell_data[shell_char] != max_L:
            raise RuntimeError('Invalid max_L')

        mobj = re.match(r'^\s*(\d+)\s*$', lines2[potential_start + 1])
        if mobj is None:
            raise RuntimeError('Where is the "nprimitive" line?')

        nprimitive = int(mobj.group(1))

        lines3 = lines2[potential_start + 2 : potential_stop]

        if len(lines3) != nprimitive:
            raise RuntimeError('nprimitive is incorrect')

        ns = []
        ws = []
        es = []
        for line in lines3:
            mobj = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s*$', line)
            if mobj is None:
                raise RuntimeError('where is the "n e w" line?')
            ns.append(int(mobj.group(1)))
            es.append(float(mobj.group(2)))
            ws.append(float(mobj.group(3)))

        top_shell = ECPShell(L=max_L, ns=ns, ws=ws, es=es, is_active=True)

        # Other shells

        shells = []
        for index in range(1, len(potential_indices) - 1):
            potential_start = potential_indices[index]
            potential_stop = potential_indices[index + 1]

            mobj = re.match(r'^\s*(\S)-(\S)\s+potential\s*$', lines2[potential_start], re.IGNORECASE)
            if mobj is None:
                raise RuntimeError('Where is the "L-max_L potential" line?')

            if ECPAtom.shell_data[mobj.group(2).upper()] != max_L:
                raise RuntimeError('Invalid max_L')

            L = ECPAtom.shell_data[mobj.group(1).upper()]

            mobj = re.match(r'^\s*(\d+)\s*$', lines2[potential_start + 1])
            if mobj is None:
                raise RuntimeError('Where is the "nprimitive" line?')

            nprimitive = int(mobj.group(1))

            lines3 = lines2[potential_start + 2 : potential_stop]

            if len(lines3) != nprimitive:
                raise RuntimeError('nprimitive is incorrect')

            ns = []
            ws = []
            es = []
            for line in lines3:
                mobj = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s*$', line)
                if mobj is None:
                    raise RuntimeError('where is the "n e w" line?')
                ns.append(int(mobj.group(1)))
                es.append(float(mobj.group(2)))
                ws.append(float(mobj.group(3)))

            shells.append(ECPShell(L=L, ns=ns, ws=ws, es=es, is_active=True))

        return ECPAtom(
            nelectron=nelectron, 
            shells=shells, 
            top_shell=top_shell, 
            is_active=True)

