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
    
class AOShell(object):

    def __init__(
        self,
        *,
        is_pure : bool, 
        L : int,
        exponents : np.ndarray,
        coefficients : np.ndarray,
        ):

        self.is_pure = is_pure
        self.L = L
        self.exponents = exponents
        self.coefficients = coefficients

        if exponents.shape != (self.nprimitive,): raise RuntimeError('exponents.shape != (nprimitive,)')
        if coefficients.shape != (self.nprimitive,): raise RuntimeError('coefficients.shape != (nprimitive,)')

        if L < 0: raise RuntimeError('L < 0')

    @property
    def nprimitive(self):
        return len(self.exponents)
        
    @property
    def npure(self):
        return 2 * self.L + 1

    @property
    def ncart(self):
        return (self.L + 1) * (self.L + 2) // 2

    @property
    def nao(self):
        return self.npure if self.is_pure else self.ncart

    def clone(self):
        return AOShell(
            is_pure=self.is_pure,
            L=self.L,
            exponents=self.exponents.copy(),
            coefficients=self.coefficients.copy(),
            )

    def update_to_pure(self):
        return AOShell(
            is_pure=True,
            L=self.L,
            exponents=self.exponents.copy(),
            coefficients=self.coefficients.copy(),
            )

    def update_to_cart(self):
        return AOShell(
            is_pure=False,
            L=self.L,
            exponents=self.exponents.copy(),
            coefficients=self.coefficients.copy(),
            )

    def __eq__(self, other):
        if not isinstance(other, AOShell):
            return NotImplemented
        if self.L != other.L: return False
        if self.nprimitive != other.nprimitive: return False
        if self.is_pure != other.is_pure: return False
        if any(self.exponents != other.exponents): return False
        if any(self.coefficients != other.coefficients): return False
        return True

    def __ne__(self, other):
        if not isinstance(other, AOShell):
            return NotImplemented
        if self.L != other.L: return True
        if self.nprimitive != other.nprimitive: return True
        if self.is_pure != other.is_pure: return True
        if any(self.exponents != other.exponents): return True
        if any(self.coefficients != other.coefficients): return True
        return False

    @staticmethod
    def compute_normalized_coefficients(
        *,
        L,
        exponents,
        coefficients_raw,
        normalization=1.0,
        ):

        dfact = 1
        for l in range(1,L+1):
            dfact *= 2*l-1

        coefficients = coefficients_raw * np.sqrt(2**L * (2.0 * exponents)**(L + 1.5) / (np.pi**(1.5) * dfact))

        V = 0.0
        for k1 in range(len(coefficients)):  
            for k2 in range(len(coefficients)):
                V += (np.sqrt(4.0 * exponents[k1] * exponents[k2]) / (exponents[k1] + exponents[k2]))**(L + 1.5) * coefficients_raw[k1] * coefficients_raw[k2]

        coefficients *= np.sqrt(normalization / V)

        return coefficients

    @staticmethod
    def build_from_raw_data(
        *,
        is_pure,
        L,
        exponents,
        coefficients_raw,
        normalization=1.0,
        ):

        coefficients = AOShell.compute_normalized_coefficients(
            L=L,
            exponents=exponents,
            coefficients_raw=coefficients_raw,
            normalization=normalization,
            )

        return AOShell(
            is_pure=is_pure,
            L=L,
            exponents=exponents,
            coefficients=coefficients,
            )
