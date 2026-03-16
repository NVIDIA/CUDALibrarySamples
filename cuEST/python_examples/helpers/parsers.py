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

import re
from collections import namedtuple

def simple_xyz_parser(
    *,
    filename,
    to_bohr_scale_factor,
    ):

    elements = [
        "X",  "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE",
        "NA", "MG", "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA",
        "SC", "TI", "V",  "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y",  "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I",  "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W",  "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U",  "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS",
        "RG", "CN", "NH", "FL", "MC", "LV", "TS", "OG"
        ]
    symbols = []
    xyzs = []
    Zs = []
    with open(filename) as fp:
        atom_count = int(fp.readline())
        comment = fp.readline()
        for _ in range(atom_count):
            parts = fp.readline().split()
            symbol = parts[0].upper()
            # The potential integrals routine compute electron-electron interaction
            # integrals; to make this an electron-nucleus interaction we scale Z here
            Z = -1.0 * elements.index(symbol)
            x = float(parts[1]) * to_bohr_scale_factor
            y = float(parts[2]) * to_bohr_scale_factor
            z = float(parts[3]) * to_bohr_scale_factor
            xyzs.extend([x, y, z])
            symbols.append(symbol)
            Zs.append(Z)
    return symbols, xyzs, Zs


ECPShellInfo = namedtuple('ECPShellInfo', ['L', 'ns', 'ws', 'es'])

def _ecp_shell_info_from_ecp_lines(
    *,
    lines,
    ):
    _shell_data_ = {
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
    mobj = re.match(r'^\s*(\S+)-ECP\s+(\d+)\s+(\d+)\s*$', lines[0], re.IGNORECASE)
    if mobj is None:
        raise RuntimeError('Where is the "ATOM-ECP max_L nelectron" line?')
    max_L = int(mobj.group(2))
    nelectron = int(mobj.group(3))

    lines2 = lines[1:]

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
    if shell_char not in _shell_data_:
        raise RuntimeError(f'Unknown shell type: {shell_char}')

    if _shell_data_[shell_char] != max_L:
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

    top_shell = ECPShellInfo(L=max_L, ns=ns, ws=ws, es=es)

    # Other shells

    shells = []
    for index in range(1, len(potential_indices) - 1):
        potential_start = potential_indices[index]
        potential_stop = potential_indices[index + 1]

        mobj = re.match(r'^\s*(\S)-(\S)\s+potential\s*$', lines2[potential_start], re.IGNORECASE)
        if mobj is None:
            raise RuntimeError('Where is the "L-max_L potential" line?')

        if _shell_data_[mobj.group(2).upper()] != max_L:
            raise RuntimeError('Invalid max_L')

        L = _shell_data_[mobj.group(1).upper()]

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

        shells.append(ECPShellInfo(L=L, ns=ns, ws=ws, es=es))

    return {
        'nelectron': nelectron,
        'shells': shells,
        'top_shell': top_shell,
        }


def simple_ecp_parser(
    *,
    filename,
    symbols,
    ):

    with open(filename) as fp:
        lines = fp.readlines()

    # => Cleaning <= #

    # Strip blank lines out
    lines = [_ for _ in lines if len(_.strip())]
    # Strip comment lines out
    re_comment = re.compile(r'\s*!')
    lines = [_ for _ in lines if not re.match(re_comment, _)]
    # Lines must be nonzero 
    if len(lines) == 0: raise RuntimeError('GBS lines are blank')

    # Find ECP entries matching this kind of pattern
    # SI     0
    # SI-ECP     2     10
    # But joined onto a single line
    ecp_re = re.compile(r'^\s*(\S+)\s+(\d+)\s*(\S+)-ECP\s+(\d+)\s+(\d+)\s*$')
    ecp_inds = []
    for ind in range(len(lines) - 1):
        line = lines[ind] + lines[ind + 1]  # Join into a single line
        if ecp_re.match(line):
            ecp_inds.append(ind)
    ecp_inds.append(len(lines))

    # Extract the lines pertaining to each atom symbol
    symbol_ecp_info = {}
    for k in range(len(ecp_inds) - 1):
        ind1 = ecp_inds[k]
        ind2 = ecp_inds[k + 1]
        if (ind2 - ind1) <= 0:  # Guard against empty entries
            continue
        mobj = re.match(
            r'^\s*(\S+)\s+(\d+)\s*$', lines[ind1]
        )  # Check if the line is a regular basis entry or an ECP entry
        if mobj == None:
            raise Exception("Where is the ID V line?")
        if mobj.group(2) != '0':
            continue  # Regular basis entry, not an ECP entry
        # Try to match the next line to something of the form
        # SR-ECP     3     28
        mobj = re.match(r'^\s*(\S+)-ECP\s+(\d+)\s+(\d+)\s*$', lines[ind1 + 1])
        if mobj is None:
            # This is a regular atom entry, not an ECP entry
            continue
        symbol_ecp_info[mobj.group(1).upper()] = _ecp_shell_info_from_ecp_lines(lines=lines[ind1 + 1 : ind2])

    ecp_metadata = []
    for symbol in symbols:
        ecp_metadata.append(symbol_ecp_info.get(symbol.upper(), None))

    return ecp_metadata


def simple_gbs_parser(
    *,
    filename,
    symbols,
    ):

    shell_types = [
        'S', 'P', 'D', 'F', 'G', 'H', 'I',
        'K', 'L', 'M', 'N', 'O', 'Q', 'R',
        'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        ]
    with open(filename) as fp:
        lines = fp.readlines()

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
    symbol_index_map = {}
    for index, atom_id_line in enumerate(atom_id_lines):
        mobj = re.match(re_atom_id, atom_id_line)
        if mobj is None:
            raise RuntimeError('Malformed atom ID line: %s' % (atom_id_line))
        symbol = mobj.group(1)
        atom_index = int(mobj.group(2))
        if atom_index != 0: 
            raise RuntimeError('"Symbol 0" is only allowed atom line - multiple basis sets per atom type in GBS files is not supported by this library')
        symbol_upper = symbol.upper()
        if symbol in symbol_index_map:
            raise RuntimeError(f'Duplicate atomic symbol in GBS file: {symbol}')
        symbol_index_map[symbol_upper] = index

    re_shell_type = re.compile(r'^\s*(\S+)\s+(\d+)\s+(\S+)\s*$')
    re_primitive = re.compile(r'^\s*(\S+)\s+(\S+)\s*$')

    shellinfo = []
    for symbol in symbols:
        symbol = symbol.upper()
        if symbol not in symbol_index_map:
            raise RuntimeError(f'{symbol} not defined in GBS file')
        block_index = symbol_index_map[symbol]
        index1 = separator_indices[block_index + 0]
        index2 = separator_indices[block_index + 1] if block_index + 1 < len(separator_indices) else len(lines)

        block_lines = lines[index1+2:index2]
        shell_type_indices = [k for k, line in enumerate(block_lines) if re.match(re_shell_type, line)]

        atom_shells = []
        for index in shell_type_indices:
            shell_type_line = block_lines[index]
            mobj = re.match(re_shell_type, shell_type_line)
            am_symbol_upper = mobj.group(1).upper()
            if am_symbol_upper not in shell_types:
                raise RuntimeError(f'Unknown angular momentum symbol: {am_symbol_upper}')
            L = shell_types.index(am_symbol_upper)
            nprimitive = int(mobj.group(2))
            normalization = float(mobj.group(3))
            exponents = []
            coefficients_raw = []
            for K in range(nprimitive):
                primitive_line = block_lines[index + 1 + K]
                # Replace D with E for FORTRAN notation
                primitive_line = primitive_line.replace('D', 'E')
                primitive_line = primitive_line.replace('d', 'e')
                mobj = re.match(re_primitive, primitive_line)
                exponents.append(float(mobj.group(1)))
                coefficients_raw.append(float(mobj.group(2)))
            atom_shells.append({
                'is_pure' : is_pure,
                'L' : L,
                'exponents' : exponents,
                'coefficients' : coefficients_raw,
                'normalization' : normalization,
                })
        shellinfo.append(atom_shells)
    return shellinfo
