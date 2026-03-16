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

from .ao_basis import AOBasis

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace

from .cuest_handle import CuestHandle
from .cuest_ao_shell import CuestAOShell

from .cuest_parameters import CuestParameters
    
class CuestAOBasis(object):

    def __init__(
        self,
        *,
        handle : CuestHandle,
        basis : AOBasis,
        ):

        self.initialized = False

        self.handle = handle

        shells = [CuestAOShell(
            handle=handle,
            ao_shell=shell) for shell in basis.shells_unrolled]

        # Careful - "shells" owns these handles. 
        # OK as we will use readonly during this constructor
        shells_raw = [_.ao_shell_handle for _ in shells]

        nshell_per_atom = [len(_) for _ in basis.shells]        

        ao_basis_parameters = CuestParameters(
            parametersType=ce.CuestParametersType.CUEST_AOBASIS_PARAMETERS,
            )

        self.ao_basis_handle = ce.cuestAOBasisHandle()

        # => Workspace Query <= #

        persistent_workspace_descriptor = CuestWorkspaceDescriptor()
        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestAOBasisCreateWorkspaceQuery(
            handle=handle.handle,
            numAtoms=basis.natom,
            numShellsPerAtom=nshell_per_atom,
            shells=shells_raw,
            parameters=ao_basis_parameters.parameters,
            persistentWorkspaceDescriptor=persistent_workspace_descriptor.pointer,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            outBasis=self.ao_basis_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestAOBasisCreateWorkspaceQuery failed')
            
        persistent_workspace = CuestWorkspace(workspaceDescriptor=persistent_workspace_descriptor)
        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        # => Creation <= #

        status = ce.cuestAOBasisCreate(
            handle=handle.handle,
            numAtoms=basis.natom,
            numShellsPerAtom=nshell_per_atom,
            shells=shells_raw,
            parameters=ao_basis_parameters.parameters,
            persistentWorkspace=persistent_workspace.pointer,
            temporaryWorkspace=temporary_workspace.pointer,
            outBasis=self.ao_basis_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestAOBasisCreate failed')

        # Bind the lifetime of persistent_workspace to this object
        self.persistent_workspace = persistent_workspace

        self.initialized = True

    def __del__(self):

        if not self.initialized: return

        status = ce.cuestAOBasisDestroy(
            handle=self.ao_basis_handle,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestAOBasisDestroy failed')

    @property
    def is_pure(self):
        query = ce.data_int32_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_IS_PURE,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return bool(query.value)

    @property
    def is_cart(self):
        query = ce.data_int32_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_IS_CART,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return bool(query.value)

    @property
    def is_mixed(self):
        query = ce.data_int32_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_IS_MIXED,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return bool(query.value)

    @property
    def natom(self):
        query = ce.data_uint64_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_ATOM,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return int(query.value)

    @property
    def nshell(self):
        query = ce.data_uint64_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_SHELL,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return int(query.value)

    @property
    def nao(self):
        query = ce.data_uint64_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_AO,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return int(query.value)

    @property
    def ncart(self):
        query = ce.data_uint64_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_CART,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return int(query.value)

    @property
    def nprimitive(self):
        query = ce.data_uint64_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_NUM_PRIMITIVE,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return int(query.value)

    @property
    def max_L(self):
        query = ce.data_uint64_t()

        status = ce.cuestQuery(
            handle=self.handle.handle,
            type=ce.CuestType.CUEST_AOBASIS,
            object=self.ao_basis_handle,
            attribute=ce.CuestAOBasisAttributes.CUEST_AOBASIS_MAX_L,
            attributeValue=query,
            )

        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuest query failed')

        return int(query.value)

    def __str__(self):
        s = ''
        s += 'CuestAOBasis:\n'
        s += '%-10s = %6d\n' % ('natom', self.natom)
        s += '%-10s = %6d\n' % ('nshell', self.nshell)
        s += '%-10s = %6d\n' % ('nao', self.nao)
        s += '%-10s = %6d\n' % ('ncart', self.ncart)
        s += '%-10s = %6d\n' % ('nprimitive', self.nprimitive)
        s += '%-10s = %6d\n' % ('max_L', self.max_L)
        s += '%-10s = %6s\n' % ('is_pure', self.is_pure)
        s += '%-10s = %6s\n' % ('is_cart', self.is_cart)
        s += '%-10s = %6s\n' % ('is_mixed', self.is_mixed)
        return s
