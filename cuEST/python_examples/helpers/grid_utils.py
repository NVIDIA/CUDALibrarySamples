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

def symbol_to_ahlrichs_radius(
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


def build_ahlrichs_radial_quadrature(
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

    return np.flip(radial_nodes), np.flip(radial_weights)
