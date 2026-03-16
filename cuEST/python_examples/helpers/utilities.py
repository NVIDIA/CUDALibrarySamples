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

try:
    import numpy as np
except ImportError:
    raise RuntimeError("numpy could not be imported. It is available via\n\tpip install numpy")

def normalize_shell_coefficients(
    coefficients,
    exponents,
    L,
    normalization,
    ):
    """
    Normalizes a Gaussian shell of primitive functions, so that their self overlap is the
    value provided by the "normalization" argument.
    """

    exponents = np.array(exponents)
    coefficients = np.array(coefficients)
    assert len(exponents) == len(coefficients)

    dfact = 1
    for l in range(1, L+1):
        dfact *= 2 * l - 1

    # Primitive normalization
    output = np.sqrt( 2**L * (2 * exponents)**(L+1.5) / (np.pi**1.5 * dfact) ) * coefficients

    # Contraction normalization
    Q = 0.0
    for c1, e1 in zip(coefficients, exponents):
        for c2, e2 in zip(coefficients, exponents):
            Q += (np.sqrt(4 * e1 * e2) / (e1 + e2)) ** (L + 1.5) * c1 * c2
    Q = Q**(-0.5)
    Q *= np.sqrt(normalization)

    return output * Q
