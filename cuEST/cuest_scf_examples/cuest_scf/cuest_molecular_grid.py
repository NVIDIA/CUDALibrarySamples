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

from .memoized_property import memoized_property

import cuest.bindings as ce    

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace

from .cuest_handle import CuestHandle

from .cuest_parameters import CuestParameters

from .periodic_table import PeriodicTable

from .unit_conversions import UnitConversions


import numpy as np


# => Helpers to generate the SGn family of grids <= #


# Table of Bragg-Slater Atomic Radii (Angstroms)
#
# J.C. Slater, Symmetry and Energy Bands in Crystals,
# Dover, N.Y. 1972, page 55.
# The radii of noble gas atoms are set to be equal 
# to the radii of the corresponding halogen atoms.
# The radius of At is set to be equal to the radius of Po. 
# The radius of Fr is set to be equal to the radius of Cs. 
_bragg_slater_radii_in_ang = [
1.00, # Dummy
0.35,                                                                                0.35, # He 
1.45,1.05,                                                  0.85,0.70,0.65,0.60,0.50,0.50, # Ne 
1.80,1.50,                                                  1.25,1.10,1.00,1.00,1.00,1.00, # Ar 
2.20,1.80,1.60,1.40,1.35,1.40,1.40,1.40,1.35,1.35,1.35,1.35,1.30,1.25,1.15,1.15,1.15,1.15, # Kr 
2.35,2.00,1.80,1.55,1.45,1.45,1.35,1.30,1.35,1.40,1.60,1.55,1.55,1.45,1.45,1.40,1.40,1.40, # Xe 
2.60,2.15,
          1.95,1.85,1.85,1.85,1.85,1.85,1.85,1.80,1.75,1.75,1.75,1.75,1.75,1.75,           # Lanthanide Series
          1.75,1.55,1.45,1.35,1.35,1.30,1.35,1.35,1.35,1.50,1.90,1.80,1.60,1.90,1.90,1.90, # Rn 
2.60,2.15,
          1.95,1.80,1.80,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,           # Actinide Series
          1.75,1.55,1.55] # Last Element is Db = 105


def _symbol_to_handy_radius(
    symbol 
    ):

    N = PeriodicTable.symbol_upper_to_N_table[symbol.upper()]
    if N < 0 or N > 105:
        raise RuntimeError(f"Element number {N} is not supported in quadrature rules")

    return UnitConversions.conversions['bohr_per_ang'] * _bragg_slater_radii_in_ang[N]


_multiexp_grids = {}
_multiexp_grids[23] = [
    1.5058924745841485E-03,   1.9397997519812427E-01,
    9.9491128468613462E-03,   2.6236396365964215E-01,
    2.6212787562513891E-02,   2.6742812676396960E-01,
    5.0215014094684117E-02,   2.4866202472840934E-01,
    8.1660512821456047E-02,   2.1982849560935092E-01,
    1.2009269427713717E-01,   1.8749582392684871E-01,
    1.6491579716600493E-01,   1.5523427291530381E-01,
    2.1541139722601380E-01,   1.2506617470764020E-01,
    2.7075407329850221E-01,   9.8100418872550485E-02,
    3.3002759271630688E-01,   7.4860364849398689E-02,
    3.9224199352412348E-01,   5.5477247414684391E-02,
    4.5635157223202683E-01,   3.9817159733692153E-02,
    5.2127362052638904E-01,   2.7571576765107419E-02,
    5.8590767062942084E-01,   1.8325604666401547E-02,
    6.4915496391179928E-01,   1.1611019656801417E-02,
    7.0993783394310539E-01,   6.9477499303844600E-03,
    7.6721868628476320E-01,   3.8757714610574293E-03,
    8.2001826057974259E-01,   1.9785635068497289E-03,
    8.6743287798275903E-01,   8.9889253062980484E-04,
    9.0865042304171706E-01,   3.4755788975769891E-04,
    9.4296495577173689E-01,   1.0572868056731358E-04,
    9.6979055758591737E-01,   2.1565953939261409E-05,
    9.8868121412417898E-01,   1.9205788879728225E-06
    ]
_multiexp_grids[26] = [
    1.2118959531441655E-03,   1.6590959790074028E-01,
    7.9508350834212021E-03,   2.2996905094874989E-01,
    2.0911970701014439E-02,   2.4044079379163869E-01,
    4.0062890303616018E-02,   2.2975862025927751E-01,
    6.5227756094948006E-02,   2.0930163443099781E-01,
    9.6124873966380614E-02,   1.8456387321242523E-01,
    1.3238114512422217E-01,   1.5860582735470496E-01,
    1.7354177117918737E-01,   1.3324603342450497E-01,
    2.1907886291165971E-01,   1.0957766175447055E-01,
    2.6840004931968464E-01,   8.8230076743000874E-02,
    3.2085745807052080E-01,   6.9518256800206554E-02,
    3.7575717144832965E-01,   5.3537150852983870E-02,
    4.3236914470191778E-01,   4.0226455838504871E-02,
    4.8993751506621985E-01,   2.9418139625084539E-02,
    5.4769119749545303E-01,   2.0873023457481800E-02,
    6.0485464446878279E-01,   1.4309796296644042E-02,
    6.6065863649024748E-01,   9.4283196275204324E-03,
    7.1435096447486734E-01,   5.9282775235989377E-03,
    7.6520686385149628E-01,   3.5237959667717032E-03,
    8.1253906250257690E-01,   1.9544295871423150E-03,
    8.5570731112497822E-01,   9.9280471150256034E-04,
    8.9412727813918980E-01,   4.4916603235275061E-04,
    9.2727872393893851E-01,   1.7307202304493732E-04,
    9.5471297844715652E-01,   5.2504259615682031E-05,
    9.7606029628496505E-01,   1.0687185195322842E-05,
    9.9104255389177465E-01,   9.5039183893269103E-07
    ]


def _build_multiexp_radial_quadrature(
    *,
    npoint,
    R,
    ):
    """
    P.M.W. Gill, S.H. Chien, J. Comp. Chem., 24, 732 (2003)
    R. M. Parrish https://arxiv.org/abs/2305.01621
    """
    xu = np.array(_multiexp_grids[npoint])

    x = np.flip(xu[::2])
    u = np.flip(xu[1::2])
    radial_nodes = -R * np.log(x)
    radial_weights = R**3 * u / x

    return radial_nodes, radial_weights


def _build_handy_radial_quadrature(
    *,
    npoint,
    R,
    ):
    """ Euler-Macluarin rules, per C.W. Murray, N.C. Handy, G.J. Laming, Mol. Phys., 78, 997 (1993) """

    x = np.arange(1.0, npoint+1.0) / (npoint + 1.0)
    radial_nodes = R * x**2 / (1.0 - x)**2
    radial_weights = 2 * R**3 * x**5 / ((npoint + 1.0) * (1.0 - x)**7)

    return radial_nodes, radial_weights


def _build_de2_radial_quadrature(
    *,
    npoint,
    R,
    r_start,
    r_end,
    alpha
    ):
    """ Build a double exponential radial grid:  M. Mitani and Y. Yoshioka, Theor Chem Acc (2012) 131:1169 """

    # Use Newton's method to convert the endpoints of the quadrature from the (input) r (taken as
    # input) to x.  For SG2 and SG3 Newton's method always converges in fewer than 10 iterations.

    # Find x_start from r_start using Newton's method
    x_start = -2.3
    converged = False
    for i in range(100):
        x_start_old = x_start
        exp_neg_x = np.exp(-x_start)
        f = np.exp(alpha * x_start - exp_neg_x) - r_start
        df = (alpha + exp_neg_x) * np.exp(alpha * x_start - exp_neg_x)
        x_start = x_start - f / df
        if np.abs(x_start_old - x_start) < 1.0e-14:
            converged = True
            break
    if not converged:
        raise RuntimeError("Unable to find x_start in DE2 setup")

    # Find x_end from r_end using Newton's method
    x_end = 1.25
    converged = False
    for i in range(100):
        x_end_old = x_end
        exp_neg_x = np.exp(-x_end)
        f = np.exp(alpha * x_end - exp_neg_x) - r_end
        df = (alpha + exp_neg_x) * np.exp(alpha * x_end - exp_neg_x)
        x_end = x_end - f / df
        if np.abs(x_end_old - x_end) < 1.0e-14:
            converged = True
            break
    if not converged:
        raise RuntimeError("Unable to find x_end in DE2 setup")

    # Generate uniform grid in x space
    h = (x_end - x_start) / (npoint - 1)
    x = np.linspace(x_start, x_end, npoint)

    # Transform to r space and calculate weights
    radial_nodes = np.exp(alpha * x - np.exp(-x))
    radial_weights = h * np.exp(3.0 * alpha * x - 3.0 * np.exp(-x)) * (alpha + np.exp(-x))

    return radial_nodes, radial_weights


def _get_sg_pruning_pattern(grid_level):
    match grid_level:
        case 0:
            return 194, 50, {
                'H'  : ("MULTIEXP", 23, 1.30,   None, None, [(6, 6), (18,3), (26,1), (38,1), (74,1), (110,1), (146,6), (86,1), (50,1), (38,1), (18,1)]),
                'LI' : ("MULTIEXP", 23, 1.95,   None, None, [(6, 6), (18,3), (26,1), (38,1), (74,1), (110,1), (146,6), (86,1), (50,1), (38,1), (18,1)]),
                'BE' : ("MULTIEXP", 23, 2.20,   None, None, [(6, 4), (18,2), (26,1), (38,2), (74,1), (86,1), (110,2), (146,5), (50,1), (38,1), (18,1), (6,2)]),
                'B'  : ("MULTIEXP", 23, 1.45,   None, None, [(6, 4), (26,4), (38,3), (86,3), (146,6), (38,1), (6,2)]),
                'C'  : ("MULTIEXP", 23, 1.20,   None, None, [(6, 6), (18,2), (26,1), (38,2), (50,2), (86,1), (110,1), (146,1), (170,2), (146,2), (86,1), (38,1), (18,1)]),
                'N'  : ("MULTIEXP", 23, 1.10,   None, None, [(6, 6), (18,3), (26,1), (38,2), (74,2), (110,1), (170,2), (146,3), (86,1), (50,2)]),
                'O'  : ("MULTIEXP", 23, 1.10,   None, None, [(6, 5), (18,1), (26,2), (38,1), (50,4), (86,1), (110,5), (86,1), (50,1), (38,1), (6,1)]),
                'F'  : ("MULTIEXP", 23, 1.20,   None, None, [(6, 4), (38,2), (50,4), (74,2), (110,2), (146,2), (110,2), (86,3), (50,1), (6,1)]),
                'NA' : ("MULTIEXP", 26, 2.30,   None, None, [(6, 6), (18,2), (26,3), (38,1), (50,2), (110,8), (74,2), (6,2)]),
                'MG' : ("MULTIEXP", 26, 2.20,   None, None, [(6, 5), (18,2), (26,2), (38,2), (50,2), (74,1), (110,2), (146,4), (110,1), (86,1), (38,2), (18,1), (6,1)]),
                'AL' : ("MULTIEXP", 26, 2.10,   None, None, [(6, 6), (18,2), (26,1), (38,2), (50,2), (74,1), (86,1), (146,2), (170,2), (110,2), (86,1), (74,1), (26,1), (18,1), (6,1)]),
                'SI' : ("MULTIEXP", 26, 1.30,   None, None, [(6, 5), (18,4), (38,4), (50,3), (74,1), (110,2), (146,1), (170,3), (86,1), (50,1), (6,1)]),
                'P'  : ("MULTIEXP", 26, 1.30,   None, None, [(6, 5), (18,4), (38,4), (50,3), (74,1), (110,2), (146,1), (170,3), (86,1), (50,1), (6,1)]),
                'S'  : ("MULTIEXP", 26, 1.10,   None, None, [(6, 4), (18,1), (26,8), (38,2), (50,1), (74,2), (110,1), (170,3), (146,1), (110,1), (50,1), (6,1)]),
                'CL' : ("MULTIEXP", 26, 1.45,   None, None, [(6, 4), (18,7), (26,2), (38,2), (50,1), (74,1), (110,2), (170,3), (146,1), (110,1), (86,1), (6,1)]),
                'HE' : ("HANDY",    50, 0.5882, None, None, [(6,16), (38,5), (86,4), (194,9), (86,16), ]),
                'NE' : ("HANDY",    50, 0.6838, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17), ]),
                'AR' : ("HANDY",    50, 1.3333, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19), ]),
                }
        case 1:
            return 194, 50, {
                'H'  : ("HANDY", 50, 1.0000, None, None, [(6,16), (38,5), (86,4), (194,9), (86,16)]),
                'HE' : ("HANDY", 50, 0.5882, None, None, [(6,16), (38,5), (86,4), (194,9), (86,16)]),
                'LI' : ("HANDY", 50, 3.0769, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17)]),
                'BE' : ("HANDY", 50, 2.0513, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17)]),
                'B'  : ("HANDY", 50, 1.5385, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17)]),
                'C'  : ("HANDY", 50, 1.2308, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17)]),
                'N'  : ("HANDY", 50, 1.0256, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17)]),
                'O'  : ("HANDY", 50, 0.8791, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17)]),
                'F'  : ("HANDY", 50, 0.7692, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17)]),
                'NE' : ("HANDY", 50, 0.6838, None, None, [(6,14), (38,7), (86,3), (194,9), (86,17)]),
                'NA' : ("HANDY", 50, 4.0909, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19)]),
                'MG' : ("HANDY", 50, 3.1579, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19)]),
                'AL' : ("HANDY", 50, 2.5714, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19)]),
                'SI' : ("HANDY", 50, 2.1687, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19)]),
                'P'  : ("HANDY", 50, 1.8750, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19)]),
                'S'  : ("HANDY", 50, 1.6514, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19)]),
                'CL' : ("HANDY", 50, 1.4754, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19)]),
                'AR' : ("HANDY", 50, 1.3333, None, None, [(6,12), (38,7), (86,5), (194,7), (86,19)]),
                }
        case 2:
            return 302, 75, {
                'H'  : ("DE2", 75, 1.50, 15.0, 2.6, [(6,35), (110,12), (302,16), (86,7), (26,5)]),
                'LI' : ("DE2", 75, 3.87, 38.7, 3.2, [(6,35), (110,12), (302,17), (86,7), (50,4)]),
                'BE' : ("DE2", 75, 2.65, 26.5, 2.4, [(6,35), (110,12), (302,17), (86,7), (50,4)]),
                'B'  : ("DE2", 75, 2.20, 22.0, 2.4, [(6,35), (110,12), (302,17), (146,7), (26,4)]),
                'C'  : ("DE2", 75, 1.71, 17.1, 2.2, [(6,35), (110,12), (302,17), (146,7), (26,4)]),
                'N'  : ("DE2", 75, 1.41, 14.1, 2.2, [(6,35), (110,12), (302,17), (86,7), (26,4)]),
                'O'  : ("DE2", 75, 1.23, 12.3, 2.2, [(6,30), (110,14), (302,18), (146,8), (50,5)]),
                'F'  : ("DE2", 75, 1.08, 10.8, 2.2, [(6,26), (110,16), (302,19), (110,8), (50,6)]),
                'NA' : ("DE2", 75, 4.21, 42.1, 3.2, [(6,35), (110,12), (302,17), (86,7), (50,4)]),
                'MG' : ("DE2", 75, 3.25, 32.5, 2.4, [(6,35), (110,12), (302,17), (86,7), (50,4)]),
                'AL' : ("DE2", 75, 3.43, 34.3, 2.5, [(6,32), (110,15), (302,17), (146,7), (86,4)]),
                'SI' : ("DE2", 75, 2.75, 27.5, 2.3, [(6,32), (110,15), (302,17), (146,7), (50,4)]),
                'P'  : ("DE2", 75, 2.32, 23.2, 2.5, [(6,30), (110,14), (302,17), (146,7), (38,7)]),
                'S'  : ("DE2", 75, 2.06, 20.6, 2.5, [(6,30), (110,14), (302,17), (146,7), (38,7)]),
                'CL' : ("DE2", 75, 1.84, 18.4, 2.5, [(6,26), (110,16), (302,19), (110,8), (50,6)]),
                }
        case 3:
            return 590, 99, {
                'H'  : ("DE2", 99, 1.50, 15.0, 2.7, [(6,45), (110,16), (590,21), (194,10), (50,7)]),
                'LI' : ("DE2", 99, 3.87, 38.7, 3.0, [(6,46), (110,16), (590,22), (146, 9), (50,6)]),
                'BE' : ("DE2", 99, 2.65, 26.5, 2.4, [(6,42), (86,6), (110,14), (590,22), (194, 3), (146, 6), (50,6)]),
                'B'  : ("DE2", 99, 2.20, 22.0, 2.4, [(6,42), (86,6), (110,14), (590,22), (194, 9), (50,6)]),
                'C'  : ("DE2", 99, 1.71, 17.1, 2.4, [(6,46), (146,16), (590,22), (302,1), (194, 2), (146, 6), (86,6)]),
                'N'  : ("DE2", 99, 1.41, 14.1, 2.4, [(6,40), (110,18), (590,24), (146,11), (50,6)]),
                'O'  : ("DE2", 99, 1.23, 12.3, 2.6, [(6,40), (110,14), (194,2), (302,2), (590,24), (302,1), (194, 1), (146, 8), (50,7)]),
                'F'  : ("DE2", 99, 1.08, 10.8, 2.1, [(6,35), (110,17), (194,4), (590,25), (194, 2), (110,8), (50,8)]),
                'NA' : ("DE2", 99, 4.21, 42.1, 3.2, [(6,46), (110,16), (590,22), (146, 9), (50,6)]),
                'MG' : ("DE2", 99, 3.25, 32.5, 2.6, [(6,48), (110,15), (590,20), (146, 7), (50,9)]),
                'AL' : ("DE2", 99, 3.43, 34.3, 2.6, [(6,42), (86,6), (110,14), (590,22), (194, 3), (146, 6), (50,6)]),
                'SI' : ("DE2", 99, 2.75, 27.5, 2.8, [(6,42), (86,6), (110,14), (590,22), (194, 9), (50,6)]),
                'P'  : ("DE2", 99, 2.32, 23.2, 2.4, [(6,35), (86,1), (110,18), (194,4), (590,25), (194, 2), (146, 8), (50,6)]),
                'S'  : ("DE2", 99, 2.06, 20.6, 2.4, [(6,35), (86,1), (110,18), (194,4), (590,25), (194, 2), (146, 8), (50,6)]),
                'CL' : ("DE2", 99, 1.84, 18.4, 2.6, [(6,35), (110,17), (194,4), (590,25), (194, 2), (110,8), (50,8)]),
                }
        case _:
            raise RuntimeError("Unexpected SG grid level")

# => Helpers to generate the GRIDn family of grids <= #

def _symbol_to_ahlrichs_radius(
    *,
    symbol,
    ):

    symbol = symbol.upper()

    radii = {
         "X" : 1.00,  
         "H" : 0.80,
        "HE" : 0.90,
        "LI" : 1.80,
        "BE" : 1.40,
         "B" : 1.30,
         "C" : 1.10,
         "N" : 0.90,
         "O" : 0.90,
         "F" : 0.90,
        "NE" : 0.90,
        "NA" : 1.40,
        "MG" : 1.30,
        "AL" : 1.30,
        "SI" : 1.20,
         "P" : 1.10,
         "S" : 1.00,
        "CL" : 1.00,
        "AR" : 1.00,
         "K" : 1.50,
        "CA" : 1.40,
        "SC" : 1.30,
        "TI" : 1.20,
         "V" : 1.20,
        "CR" : 1.20,
        "MN" : 1.20,
        "FE" : 1.20,
        "CO" : 1.20,
        "NI" : 1.10,
        "CU" : 1.10,
        "ZN" : 1.10,
        "GA" : 1.10,
        "GE" : 1.00,
        "AS" : 0.90,
        "SE" : 0.90,
        "BR" : 0.90,
        "KR" : 0.90,
    }

    return radii.get(symbol, 1.0)


def _build_ahlrichs_radial_quadrature(
    *,
    npoint,
    R,
    ):

    alpha = 0.6

    n = np.arange(1.0, npoint+1.0)
    z = n * np.pi / (npoint + 1.0)
    x = np.cos(z)
    y = np.sin(z)
    u = np.log((1.0 - x) / 2.0)
    v = ((1.0 + x)**alpha) / np.log(2.0)
    radial_nodes = - R * v * u
    radial_weights = np.pi / (npoint + 1.0) * y * R  * v * (-alpha * u / (1.0 + x) + 1.0 / (1.0 - x)) * radial_nodes**2

    return np.flip(radial_nodes).tolist(), np.flip(radial_weights).tolist()


def _get_grid_pruning_pattern(grid_level):
    match grid_level:
        case 1:
            return 194, 50, {
                'H'  : [(14,6),  (50,4), (50,10) ],
                'HE' : [(14,6),  (50,4), (50,10) ],
                'LI' : [(14,8),  (50,4), (110,13)],
                'BE' : [(14,8),  (50,4), (110,13)],
                'B'  : [(14,8),  (50,4), (110,13)],
                'C'  : [(14,8),  (50,4), (110,13)],
                'N'  : [(14,8),  (50,4), (110,13)],
                'O'  : [(14,8),  (50,4), (110,13)],
                'F'  : [(14,8),  (50,4), (110,13)],
                'NE' : [(14,8),  (50,4), (110,13)],
                'NA' : [(14,10), (50,5), (110,15)],
                'MG' : [(14,10), (50,5), (110,15)],
                'AL' : [(14,10), (50,5), (110,15)],
                'SI' : [(14,10), (50,5), (110,15)],
                'P'  : [(14,10), (50,5), (110,15)],
                'S'  : [(14,10), (50,5), (110,15)],
                'CL' : [(14,10), (50,5), (110,15)],
                'AR' : [(14,10), (50,5), (110,15)],
                'K'  : [(14,11), (50,6), (110,18)],
                'CA' : [(14,11), (50,6), (110,18)],
                'SC' : [(14,11), (50,6), (110,18)],
                'TI' : [(14,11), (50,6), (110,18)],
                'V'  : [(14,11), (50,6), (110,18)],
                'CR' : [(14,11), (50,6), (110,18)],
                'MN' : [(14,11), (50,6), (110,18)],
                'FE' : [(14,11), (50,6), (110,18)],
                'CO' : [(14,11), (50,6), (110,18)],
                'NI' : [(14,11), (50,6), (110,18)],
                'CU' : [(14,11), (50,6), (110,18)],
                'ZN' : [(14,11), (50,6), (110,18)],
                'GA' : [(14,11), (50,6), (110,18)],
                'GE' : [(14,11), (50,6), (110,18)],
                'AS' : [(14,11), (50,6), (110,18)],
                'SE' : [(14,11), (50,6), (110,18)],
                'BR' : [(14,11), (50,6), (110,18)],
                'KR' : [(14,11), (50,6), (110,18)],
                }
        case 2:
            return 302, 55, {
                'H'  : [(14,8),  (50,4), (110,13)],
                'HE' : [(14,8),  (50,4), (110,13)],
                'LI' : [(14,10), (50,5), (194,15)],
                'BE' : [(14,10), (50,5), (194,15)],
                'B'  : [(14,10), (50,5), (194,15)],
                'C'  : [(14,10), (50,5), (194,15)],
                'N'  : [(14,10), (50,5), (194,15)],
                'O'  : [(14,10), (50,5), (194,15)],
                'F'  : [(14,10), (50,5), (194,15)],
                'NE' : [(14,10), (50,5), (194,15)],
                'NA' : [(14,11), (50,6), (194,18)],
                'MG' : [(14,11), (50,6), (194,18)],
                'AL' : [(14,11), (50,6), (194,18)],
                'SI' : [(14,11), (50,6), (194,18)],
                'P'  : [(14,11), (50,6), (194,18)],
                'S'  : [(14,11), (50,6), (194,18)],
                'CL' : [(14,11), (50,6), (194,18)],
                'AR' : [(14,11), (50,6), (194,18)],
                'K'  : [(14,13), (50,7), (194,20)],
                'CA' : [(14,13), (50,7), (194,20)],
                'SC' : [(14,13), (50,7), (194,20)],
                'TI' : [(14,13), (50,7), (194,20)],
                'V'  : [(14,13), (50,7), (194,20)],
                'CR' : [(14,13), (50,7), (194,20)],
                'MN' : [(14,13), (50,7), (194,20)],
                'FE' : [(14,13), (50,7), (194,20)],
                'CO' : [(14,13), (50,7), (194,20)],
                'NI' : [(14,13), (50,7), (194,20)],
                'CU' : [(14,13), (50,7), (194,20)],
                'ZN' : [(14,13), (50,7), (194,20)],
                'GA' : [(14,13), (50,7), (194,20)],
                'GE' : [(14,13), (50,7), (194,20)],
                'AS' : [(14,13), (50,7), (194,20)],
                'SE' : [(14,13), (50,7), (194,20)],
                'BR' : [(14,13), (50,7), (194,20)],
                'KR' : [(14,13), (50,7), (194,20)],
                }
        case 3:
            return 434, 60, {
                'H'  : [(14,10), (50,5), (194,15)],
                'HE' : [(14,10), (50,5), (194,15)],
                'LI' : [(14,11), (50,6), (302,18)],
                'BE' : [(14,11), (50,6), (302,18)],
                'B'  : [(14,11), (50,6), (302,18)],
                'C'  : [(14,11), (50,6), (302,18)],
                'N'  : [(14,11), (50,6), (302,18)],
                'O'  : [(14,11), (50,6), (302,18)],
                'F'  : [(14,11), (50,6), (302,18)],
                'NE' : [(14,11), (50,6), (302,18)],
                'NA' : [(14,13), (50,7), (302,20)],
                'MG' : [(14,13), (50,7), (302,20)],
                'AL' : [(14,13), (50,7), (302,20)],
                'SI' : [(14,13), (50,7), (302,20)],
                'P'  : [(14,13), (50,7), (302,20)],
                'S'  : [(14,13), (50,7), (302,20)],
                'CL' : [(14,13), (50,7), (302,20)],
                'AR' : [(14,13), (50,7), (302,20)],
                'K'  : [(14,15), (50,7), (302,23)],
                'CA' : [(14,15), (50,7), (302,23)],
                'SC' : [(14,15), (50,7), (302,23)],
                'TI' : [(14,15), (50,7), (302,23)],
                'V'  : [(14,15), (50,7), (302,23)],
                'CR' : [(14,15), (50,7), (302,23)],
                'MN' : [(14,15), (50,7), (302,23)],
                'FE' : [(14,15), (50,7), (302,23)],
                'CO' : [(14,15), (50,7), (302,23)],
                'NI' : [(14,15), (50,7), (302,23)],
                'CU' : [(14,15), (50,7), (302,23)],
                'ZN' : [(14,15), (50,7), (302,23)],
                'GA' : [(14,15), (50,7), (302,23)],
                'GE' : [(14,15), (50,7), (302,23)],
                'AS' : [(14,15), (50,7), (302,23)],
                'SE' : [(14,15), (50,7), (302,23)],
                'BR' : [(14,15), (50,7), (302,23)],
                'KR' : [(14,15), (50,7), (302,23)],
                }
        case 4:
            return 770, 65, {
                'H'  : [(14,11), (50,6), (302,18)],
                'HE' : [(14,11), (50,6), (302,18)],
                'LI' : [(14,13), (50,7), (434,20)],
                'BE' : [(14,13), (50,7), (434,20)],
                'B'  : [(14,13), (50,7), (434,20)],
                'C'  : [(14,13), (50,7), (434,20)],
                'N'  : [(14,13), (50,7), (434,20)],
                'O'  : [(14,13), (50,7), (434,20)],
                'F'  : [(14,13), (50,7), (434,20)],
                'NE' : [(14,13), (50,7), (434,20)],
                'NA' : [(14,15), (50,7), (434,23)],
                'MG' : [(14,15), (50,7), (434,23)],
                'AL' : [(14,15), (50,7), (434,23)],
                'SI' : [(14,15), (50,7), (434,23)],
                'P'  : [(14,15), (50,7), (434,23)],
                'S'  : [(14,15), (50,7), (434,23)],
                'CL' : [(14,15), (50,7), (434,23)],
                'AR' : [(14,15), (50,7), (434,23)],
                'K'  : [(14,16), (50,9), (434,25)],
                'CA' : [(14,16), (50,9), (434,25)],
                'SC' : [(14,16), (50,9), (434,25)],
                'TI' : [(14,16), (50,9), (434,25)],
                'V'  : [(14,16), (50,9), (434,25)],
                'CR' : [(14,16), (50,9), (434,25)],
                'MN' : [(14,16), (50,9), (434,25)],
                'FE' : [(14,16), (50,9), (434,25)],
                'CO' : [(14,16), (50,9), (434,25)],
                'NI' : [(14,16), (50,9), (434,25)],
                'CU' : [(14,16), (50,9), (434,25)],
                'ZN' : [(14,16), (50,9), (434,25)],
                'GA' : [(14,16), (50,9), (434,25)],
                'GE' : [(14,16), (50,9), (434,25)],
                'AS' : [(14,16), (50,9), (434,25)],
                'SE' : [(14,16), (50,9), (434,25)],
                'BR' : [(14,16), (50,9), (434,25)],
                'KR' : [(14,16), (50,9), (434,25)],
                }
        case 5:
            return 1202, 75, {
                'H'  : [(14,15), (50,7),  (434,23)],
                'HE' : [(14,15), (50,7),  (434,23)],
                'LI' : [(14,16), (50,9),  (770,25)],
                'BE' : [(14,16), (50,9),  (770,25)],
                'B'  : [(14,16), (50,9),  (770,25)],
                'C'  : [(14,16), (50,9),  (770,25)],
                'N'  : [(14,16), (50,9),  (770,25)],
                'O'  : [(14,16), (50,9),  (770,25)],
                'F'  : [(14,16), (50,9),  (770,25)],
                'NE' : [(14,16), (50,9),  (770,25)],
                'NA' : [(14,18), (50,9),  (770,28)],
                'MG' : [(14,18), (50,9),  (770,28)],
                'AL' : [(14,18), (50,9),  (770,28)],
                'SI' : [(14,18), (50,9),  (770,28)],
                'P'  : [(14,18), (50,9),  (770,28)],
                'S'  : [(14,18), (50,9),  (770,28)],
                'CL' : [(14,18), (50,9),  (770,28)],
                'AR' : [(14,18), (50,9),  (770,28)],
                'K'  : [(14,20), (50,10), (770,30)],
                'CA' : [(14,20), (50,10), (770,30)],
                'SC' : [(14,20), (50,10), (770,30)],
                'TI' : [(14,20), (50,10), (770,30)],
                'V'  : [(14,20), (50,10), (770,30)],
                'CR' : [(14,20), (50,10), (770,30)],
                'MN' : [(14,20), (50,10), (770,30)],
                'FE' : [(14,20), (50,10), (770,30)],
                'CO' : [(14,20), (50,10), (770,30)],
                'NI' : [(14,20), (50,10), (770,30)],
                'CU' : [(14,20), (50,10), (770,30)],
                'ZN' : [(14,20), (50,10), (770,30)],
                'GA' : [(14,20), (50,10), (770,30)],
                'GE' : [(14,20), (50,10), (770,30)],
                'AS' : [(14,20), (50,10), (770,30)],
                'SE' : [(14,20), (50,10), (770,30)],
                'BR' : [(14,20), (50,10), (770,30)],
                'KR' : [(14,20), (50,10), (770,30)],
                }
        case _:
            raise RuntimeError("Unexpected grid level")


class CuestAtomGridList(object):

    def __init__(self):
        self.handles = []

    def append(self, handle):
        self.handles.append(handle)

    def __del__(self):
        for handle in self.handles:
            status = ce.cuestAtomGridDestroy(atomGrid=handle)
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestAtomGridDestroy failed')


class CuestMolecularGrid(object):
    """
    Builds integration grids according to one of two pruning schemes:

    family = GRID: The GRIDn pruning scheme
    * O. Treutler and R. Ahlrichs, J. Chem. Phys., 102, 346 (1995)  
    where valid values are 1 <= n <= 5, where larger numbers yield larger grids.

    family = SG: The SGn pruning scheme
    * S-H. Chien, P.M.W. Gill, J. Comput. Chem., 27, 730 (2006)
    * P.M.W. Gill, B.G. Johnson, J.A. Pople, Chem. Phys. Lett., 209(5-6), 506 (1993)
    * S. Dasgupta and J. M. HerbertJ. Comput. Chem., 38, 869 (2017)
    where valid values are 0 <= n <= 3, where larger numbers yield larger grids.
    """
    def __init__(
        self,
        *,
        handle : CuestHandle,
        grid_level : int,
        xyz : np.ndarray,
        Ns : list[int],
        family="GRID",
        ):

        self.initialized = False
        family = family.upper()

        supported_families = ["GRID", "SG"]
        if family not in supported_families:
            raise RuntimeError(f"Unrecognized grid family '{family}'.  Supported values are {', '.join(supported_families)}")

        if family == "GRID":
            if grid_level < 1 or grid_level > 5:
                raise RuntimeError("Only grid_level 1 to 5 (inclusive) is supported for the GRID family")
            self.grid_name = f'GRID{grid_level}'
            default_n_angular, default_n_radial, pruning_map = _get_grid_pruning_pattern(grid_level)
        else:
            if grid_level < 0 or grid_level > 3:
                raise RuntimeError("Only grid_level 0 to 3 (inclusive) is supported for the SG family")
            default_n_angular, default_n_radial, pruning_map = _get_sg_pruning_pattern(grid_level)
            self.grid_name = f'SG{grid_level}'

        self.handle = handle

        atomgrid_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_ATOMGRID_PARAMETERS)

        atomgrids = CuestAtomGridList()
        for atom,N in enumerate(Ns):
            symbol_upper = PeriodicTable.N_to_symbol_upper_table[N]
            if family == 'GRID':
                # GRIDn family
                ahlrichs_radius = _symbol_to_ahlrichs_radius(
                    symbol=symbol_upper,
                    )
                if symbol_upper in pruning_map:
                    num_angular_points = []
                    for n_angular, reps in pruning_map[symbol_upper]:
                        num_angular_points.extend([n_angular]*reps)
                else:
                    num_angular_points = [default_n_angular] * default_n_radial

                radial_nodes, radial_weights = _build_ahlrichs_radial_quadrature(
                    npoint=len(num_angular_points),
                    R=ahlrichs_radius,
                    )
            else:
                # SGn family
                if symbol_upper in pruning_map:
                    num_angular_points = []
                    radial_quadrature_type, npoints, R, r_end, alpha, n_angular_and_reps = pruning_map[symbol_upper]
                    for n_angular, reps in n_angular_and_reps:
                        num_angular_points.extend([n_angular]*reps)
                    if radial_quadrature_type == "HANDY":
                        radial_nodes, radial_weights = _build_handy_radial_quadrature(
                            npoint=npoints,
                            R=R,
                            )
                    elif radial_quadrature_type == "MULTIEXP":
                        radial_nodes, radial_weights = _build_multiexp_radial_quadrature(
                            npoint=npoints,
                            R=R,
                            )
                    elif radial_quadrature_type == "DE2":
                        radial_nodes, radial_weights = _build_de2_radial_quadrature(
                            npoint=npoints,
                            R=R,
                            r_start=1e-7,
                            r_end=r_end,
                            alpha=alpha,
                            )
                    else:
                        raise RuntimeError("Unexpected radial quadrature type")
                else:
                    # For atoms without an explicit spec we build an unpruned Euler-Maclaurin grid
                    num_angular_points = [default_n_angular] * default_n_radial

                    R = _symbol_to_handy_radius(symbol_upper)
                    radial_nodes, radial_weights = _build_handy_radial_quadrature(
                        npoint=default_n_radial,
                        R=R,
                        )

            atomgrid_handle = ce.cuestAtomGridHandle()
            status = ce.cuestAtomGridCreate(
                handle=handle.handle,
                numRadialPoints=len(num_angular_points),
                radialNodes=radial_nodes,
                radialWeights=radial_weights,
                numAngularPoints=num_angular_points,
                parameters=atomgrid_parameters.parameters,
                outAtomGrid=atomgrid_handle,
                )
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError('cuestAtomGridCreate failed')
            atomgrids.append(atomgrid_handle)

        del atomgrid_parameters

        moleculargrid_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_MOLECULARGRID_PARAMETERS)

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestMolecularGridCreateWorkspaceQuery(
            handle=handle.handle,
            numAtoms=len(Ns),
            atomGrid=atomgrids.handles,
            xyz=list(xyz.ravel()),
            parameters=moleculargrid_parameters.parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outGrid=ce.cuestMolecularGridHandle(),
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestMolecularGridCreateWorkspaceQuery failed')

        grid_persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
        grid_temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        self.moleculargrid_handle = ce.cuestMolecularGridHandle()

        status = ce.cuestMolecularGridCreate(
            handle=handle.handle,
            numAtoms=len(Ns),
            atomGrid=atomgrids.handles,
            xyz=list(xyz.ravel()),
            parameters=moleculargrid_parameters.parameters,
            persistentWorkspace=grid_persistent_workspace.pointer,
            temporaryWorkspace=grid_temporary_workspace.pointer,
            outGrid=self.moleculargrid_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestMolecularGridCreate failed')

        del moleculargrid_parameters
        del grid_temporary_workspace
        del atomgrids

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = grid_persistent_workspace

        self.initialized = True


    def __del__(self):

        if not self.initialized: return

        status = ce.cuestMolecularGridDestroy(
            grid=self.moleculargrid_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestMolecularDestroy failed')

    @memoized_property
    def natom(self):
        natom = ce.data_uint64_t()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_MOLECULARGRID,
            object=self.moleculargrid_handle,
            attribute=ce.CuestMolecularGridAttributes.CUEST_MOLECULARGRID_NUM_ATOM,
            attributeValue=natom,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return natom.value

    @memoized_property
    def npoint(self):
        npoint = ce.data_uint64_t()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_MOLECULARGRID,
            object=self.moleculargrid_handle,
            attribute=ce.CuestMolecularGridAttributes.CUEST_MOLECULARGRID_NUM_POINT,
            attributeValue=npoint,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return npoint.value

    @memoized_property
    def max_point(self):
        max_point = ce.data_uint64_t()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_MOLECULARGRID,
            object=self.moleculargrid_handle,
            attribute=ce.CuestMolecularGridAttributes.CUEST_MOLECULARGRID_MAX_POINT,
            attributeValue=max_point,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return max_point.value

    @memoized_property
    def max_radial_point(self):
        max_radial_point = ce.data_uint64_t()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_MOLECULARGRID,
            object=self.moleculargrid_handle,
            attribute=ce.CuestMolecularGridAttributes.CUEST_MOLECULARGRID_MAX_RADIAL_POINT,
            attributeValue=max_radial_point,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return max_radial_point.value

    @memoized_property
    def max_angular_point(self):
        max_angular_point = ce.data_uint64_t()
        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_MOLECULARGRID,
            object=self.moleculargrid_handle,
            attribute=ce.CuestMolecularGridAttributes.CUEST_MOLECULARGRID_MAX_ANGULAR_POINT,
            attributeValue=max_angular_point,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestQuery failed')
        return max_angular_point.value

    # => String Details <= #

    def __str__(self):
        return self.string

    @property
    def string(self):
        s = ''
        s += 'MolecularGrid:\n'
        s += '%-18s = %10s\n' % ('scheme name',        self.grid_name)
        s += '%-18s = %10d\n' % ('natom',              self.natom)
        s += '%-18s = %10d\n' % ('atomic max npoint',  self.max_point)
        s += '%-18s = %10d\n' % ('radial max npoint',  self.max_radial_point)
        s += '%-18s = %10d\n' % ('angular max npoint', self.max_angular_point)
        s += '%-18s = %10d\n' % ('total npoint',       self.npoint)
        return s

