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

import numpy as np
from .gpu_matrix import GPUMatrix
from .gpu_matrix_utility import GPUMatrixUtility
    
class DIIS(object):

    def __init__(
        self,
        *,
        max_nvector : int, 
        ):

        self.max_nvector = max_nvector

        self.state_vectors = []
        self.error_vectors = []

        self.Efull = np.zeros((max_nvector,)*2)
        
    @property
    def nvector(self) -> int:
        return len(self.state_vectors)

    @property
    def E(self) -> np.ndarray:
        return self.Efull[:self.nvector, :self.nvector]

    def iterate(
        self,
        *,
        state_vectors : list,
        error_vectors : list,
        ) -> list:

        # Worst-error replacement DIIS history update
        pivot = None
        if self.nvector < self.max_nvector:
            pivot = self.nvector
            self.state_vectors.append([sv.clone() for sv in state_vectors])
            self.error_vectors.append([ev.clone() for ev in error_vectors])
        else:
            pivot = np.argmax(np.diag(self.E))
            self.state_vectors[pivot] = [sv.clone() for sv in state_vectors]
            self.error_vectors[pivot] = [ev.clone() for ev in error_vectors]

        # Error metric inner product update
        for ind in range(self.nvector):
            Eval = sum(
                GPUMatrixUtility.ddot(
                    x=self.error_vectors[ind][k],
                    y=self.error_vectors[pivot][k],
                    )
                for k in range(len(error_vectors))
                )
            self.E[ind, pivot] = Eval
            self.E[pivot, ind] = Eval

        # DIIS extrapolation
        c, L = self.diis_coefficients
        result = [
            GPUMatrix(
                nrows=sv.nrows,
                ncols=sv.ncols,
                dtype=sv.dtype,
                initialize=True,
                )
            for sv in state_vectors
            ]
        for ind in range(self.nvector):
            for k, out in enumerate(result):
                GPUMatrixUtility.daxpy(
                    alpha=c[ind],
                    x=self.state_vectors[ind][k],
                    y=out,
                    )

        return result

    @property
    def diis_coefficients(self) -> (np.ndarray, float):
        
        E = self.E 

        B = np.zeros((self.nvector + 1,)*2)
        B[:-1, :-1] = E
        B[:-1, -1] = 1.0
        B[-1, :-1] = 1.0
         
        # Negative and zero trapping
        
        for ind in range(self.nvector):
            # Negatives are illegal
            if B[ind, ind] < 0.0:
                raise RuntimeError('Negative diagonal in B matrix.')
            # Zeros are serendipitous convergence
            if B[ind, ind] == 0.0:
                d = np.zeros(self.nvector)
                d[ind] = 1.0
                return d, 0.0

        # Balancing

        s = np.zeros((self.nvector + 1,))
        s[:-1] = np.diag(E)**(-0.5)
        s[-1] = 1.0

        B[...] *= np.outer(s, s)

        # Inversion

        # NOTE: We are calling numpy.linalg.inv here for simplicity
        # Production codes may desire a conditioned inverse, least-norm
        # solution, or fallback plan for relative condition numbers >~ 1.0E12
        Binv = np.linalg.inv(B) 
        
        # Result (last column of Binv plus balance)
        
        d = s[:-1] * Binv[:-1, -1]
        L = Binv[-1, -1]

        return d, L
