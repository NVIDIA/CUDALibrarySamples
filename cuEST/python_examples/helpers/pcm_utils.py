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

# York-Karplus zeta exponents indexed by Lebedev angular-point count.
# J. Phys. Chem. A, 103, 11060-11079 (1999), Table 1.
_zeta_map = {
      14: 4.865,  26: 4.855,  50: 4.893, 110: 4.901,  194: 4.903,
     302: 4.905, 434: 4.906, 590: 4.905, 770: 4.899,  974: 4.907,
    1202: 4.907,
}

# Scaled Bondi radii (Angstrom) for PCM cavity construction.
# Truhlar et al., J. Phys. Chem. A, 113, 5806-5812 (2009), Table 12;
# CRC Handbook, 95th Ed., pp. 9-49.  Multiply by 1.2 × bohr_per_ang to use.
_bondi_radii_ang = {
    'H':  1.10, 'HE': 1.40, 'LI': 1.81, 'BE': 1.53, 'B':  1.92,
    'C':  1.70, 'N':  1.55, 'O':  1.52, 'F':  1.47, 'NE': 1.54,
    'NA': 2.27, 'MG': 1.73, 'AL': 1.84, 'SI': 2.10, 'P':  1.80,
    'S':  1.80, 'CL': 1.75, 'AR': 1.88, 'K':  2.75, 'CA': 2.31,
    'SC': 2.15, 'TI': 2.11, 'V':  2.07, 'CR': 2.06, 'MN': 2.05,
    'FE': 2.04, 'CO': 2.00, 'NI': 1.97, 'CU': 1.96, 'ZN': 2.01,
    'GA': 1.87, 'GE': 2.11, 'AS': 1.85, 'SE': 1.90, 'BR': 1.83,
    'KR': 2.02, 'RB': 3.03, 'SR': 2.49, 'Y':  2.32, 'ZR': 2.23,
    'NB': 2.18, 'MO': 2.17, 'TC': 2.16, 'RU': 2.13, 'RH': 2.10,
    'PD': 2.10, 'AG': 2.11, 'CD': 2.18, 'IN': 1.93, 'SN': 2.17,
    'SB': 2.06, 'TE': 2.06, 'I':  1.98, 'XE': 2.16, 'CS': 3.43,
    'BA': 2.68, 'LA': 2.43, 'CE': 2.42, 'PR': 2.40, 'ND': 2.39,
    'PM': 2.38, 'SM': 2.36, 'EU': 2.35, 'GD': 2.34, 'TB': 2.33,
    'DY': 2.31, 'HO': 2.30, 'ER': 2.29, 'TM': 2.27, 'YB': 2.26,
    'LU': 2.24, 'HF': 2.23, 'TA': 2.22, 'W':  2.18, 'RE': 2.16,
    'OS': 2.16, 'IR': 2.13, 'PT': 2.13, 'AU': 2.14, 'HG': 2.23,
    'TL': 1.96, 'PB': 2.02, 'BI': 2.07, 'PO': 1.97, 'AT': 2.02,
    'RN': 2.20, 'FR': 3.48, 'RA': 2.83, 'AC': 2.47, 'TH': 2.45,
    'PA': 2.43, 'U':  2.41, 'NP': 2.39, 'PU': 2.43, 'AM': 2.44,
    'CM': 2.45, 'BK': 2.44, 'CF': 2.45, 'ES': 2.45, 'FM': 2.45,
    'MD': 2.46, 'NO': 2.46, 'LR': 2.46,
}

_bohr_per_ang = 1.0 / 0.52917720859


def pcm_cavity_parameters(*, symbols, Zs):
    """
    Derive per-atom PCM cavity parameters from parsed molecular geometry.

    Angular grid sizes follow the convention of 110 points for hydrogen and
    194 for all heavier atoms.  York-Karplus zeta values are looked up by
    angular-point count.  Bondi radii are scaled 1.2x and converted to bohr.
    Effective nuclear charges are positive Z values (parsers store -Z).

    Returns (num_angular_points_per_atom, zetas, atomic_radii,
             effective_nuclear_charges) as lists.
    """
    num_angular_points_per_atom = [110 if s == 'H' else 194 for s in symbols]
    zetas = [_zeta_map[n] for n in num_angular_points_per_atom]
    atomic_radii = [1.2 * _bondi_radii_ang[s] * _bohr_per_ang for s in symbols]
    effective_nuclear_charges = [-Z for Z in Zs]
    return num_angular_points_per_atom, zetas, atomic_radii, effective_nuclear_charges
