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

import cuest.bindings as ce    

from .ao_shell import AOShell

from .cuest_handle import CuestHandle

from .cuest_parameters import CuestParameters

class CuestAOShell(object):

    def __init__(
        self,
        *,
        handle : CuestHandle,
        ao_shell : AOShell,
        ):

        self.initialized = False

        self.handle = handle

        ao_shell_parameters = CuestParameters(
            parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
            )

        self.ao_shell_handle = ce.cuestAOShellHandle()

        status = ce.cuestAOShellCreate(
            handle=handle.handle,
            isPure=ao_shell.is_pure,
            L=ao_shell.L,
            numPrimitive=ao_shell.nprimitive,
            exponents=ao_shell.exponents,
            coefficients=ao_shell.coefficients,
            parameters=ao_shell_parameters.parameters,
            outShell=self.ao_shell_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestAOShellCreate failed')

        self.initialized = True
        
    def __del__(self):

        if not self.initialized: return

        status = ce.cuestAOShellDestroy(
            handle=self.ao_shell_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestAOShellDestroy failed')

    # NOTE: Production codes may want to provide parameter queries here to
    # inspect the contents of the AOShell object
