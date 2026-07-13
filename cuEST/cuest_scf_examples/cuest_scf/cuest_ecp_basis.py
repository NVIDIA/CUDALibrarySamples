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

import sys

import cuest.bindings as ce

from .ecp_basis import ECPBasis

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace

from .cuest_handle import CuestHandle

from .cuest_parameters import CuestParameters
    
class CuestECPBasis(object):

    def __init__(
        self,
        *,
        handle: CuestHandle,
        ecp_basis: ECPBasis,
    ):
        self.initialized = False

        self.handle = handle
        self.ecp_basis = ecp_basis

        # Initialize all owned-handle state up front so that _destroy() can run
        # safely if construction fails partway through.
        self.ecp_atoms = []
        self.ecp_indices = []
        self.num_active_ecp = 0
        self.ecp_shell_parameters = None
        self.ecp_atom_parameters = None

        try:
            ecp_shell_parameters = ce.cuestECPShellParameters()
            status = ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_ECPSHELL_PARAMETERS,
                outParameters=ecp_shell_parameters,
            )
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError("cuestParametersCreate failed for ECPShell parameters")
            self.ecp_shell_parameters = ecp_shell_parameters

            ecp_atom_parameters = ce.cuestECPAtomParameters()
            status = ce.cuestParametersCreate(
                parametersType=ce.CuestParametersType.CUEST_ECPATOM_PARAMETERS,
                outParameters=ecp_atom_parameters,
            )
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                raise RuntimeError("cuestParametersCreate failed for ECPAtom parameters")
            self.ecp_atom_parameters = ecp_atom_parameters

            for atom_idx, ecp_atom in enumerate(self.ecp_basis.atoms):
                if not ecp_atom.is_active:
                    continue

                # Per-atom shell handles are transient: the C API copies them
                # into the ECP atom, so they are destroyed once the atom is
                # created (or once this atom fails) via the finally block below.
                top_shell_handle = None
                this_atom_shells = []
                try:
                    # Top shell
                    top_shell_handle = ce.cuestECPShellHandle()
                    top_shell = ecp_atom.top_shell

                    status = ce.cuestECPShellCreate(
                        handle=handle.handle,
                        L=top_shell.L,
                        numPrimitive=len(top_shell.es),
                        radialPowers=top_shell.ns,
                        coefficients=top_shell.ws,
                        exponents=top_shell.es,
                        parameters=ecp_shell_parameters,
                        outECPShell=top_shell_handle,
                    )
                    if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                        raise RuntimeError(f"cuestECPShellCreate failed for top shell on atom {atom_idx}")

                    # Other shells
                    for shell_idx, shell in enumerate(ecp_atom.shells):
                        ecp_shell_handle = ce.cuestECPShellHandle()

                        status = ce.cuestECPShellCreate(
                            handle=handle.handle,
                            L=shell.L,
                            numPrimitive=len(shell.es),
                            radialPowers=shell.ns,
                            coefficients=shell.ws,
                            exponents=shell.es,
                            parameters=ecp_shell_parameters,
                            outECPShell=ecp_shell_handle,
                        )
                        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                            raise RuntimeError(
                                f"cuestECPShellCreate failed for shell {shell_idx} on atom {atom_idx}"
                            )

                        this_atom_shells.append(ecp_shell_handle)

                    # ECP atom
                    ecp_atom_handle = ce.cuestECPAtomHandle()
                    status = ce.cuestECPAtomCreate(
                        handle=handle.handle,
                        numElectrons=ecp_atom.nelectron,
                        numShells=len(ecp_atom.shells),
                        shells=this_atom_shells,
                        topShell=top_shell_handle,
                        parameters=ecp_atom_parameters,
                        outECPAtom=ecp_atom_handle,
                    )
                    if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                        raise RuntimeError(f"cuestECPAtomCreate failed for atom {atom_idx}")

                    # Record the atom only once it is fully created, so
                    # ecp_atoms/ecp_indices stay aligned and cleanup-safe.
                    self.ecp_atoms.append(ecp_atom_handle)
                    self.ecp_indices.append(atom_idx)
                finally:
                    shell_handles = list(this_atom_shells)
                    if top_shell_handle is not None:
                        shell_handles.append(top_shell_handle)
                    for shell_handle in shell_handles:
                        destroy_status = ce.cuestECPShellDestroy(handle=shell_handle)
                        # Only surface a destroy failure when we are not already
                        # unwinding another exception, to avoid masking it.
                        if destroy_status != ce.CuestStatus.CUEST_STATUS_SUCCESS \
                                and sys.exc_info()[0] is None:
                            raise RuntimeError(
                                f"cuestECPShellDestroy failed on atom {atom_idx}"
                            )

            self.num_active_ecp = len(self.ecp_atoms)
            self.initialized = True
        except Exception:
            # Release whatever was created before re-raising; suppress any
            # cleanup error so the original exception is what propagates.
            try:
                self._destroy()
            except Exception:
                pass
            raise

    def _destroy(self):

        errors = []

        for atom_idx, ecp_atom_handle in zip(self.ecp_indices, self.ecp_atoms):
            status = ce.cuestECPAtomDestroy(handle=ecp_atom_handle)
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                errors.append(f"cuestECPAtomDestroy failed for atom {atom_idx}")
        self.ecp_atoms = []
        self.ecp_indices = []

        if self.ecp_atom_parameters is not None:
            status = ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_ECPATOM_PARAMETERS,
                parameters=self.ecp_atom_parameters,
                )
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                errors.append("cuestParametersDestroy failed for ECPAtom parameters")
            self.ecp_atom_parameters = None

        if self.ecp_shell_parameters is not None:
            status = ce.cuestParametersDestroy(
                parametersType=ce.CuestParametersType.CUEST_ECPSHELL_PARAMETERS,
                parameters=self.ecp_shell_parameters,
                )
            if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
                errors.append("cuestParametersDestroy failed for ECPShell parameters")
            self.ecp_shell_parameters = None

        if errors:
            raise RuntimeError("; ".join(errors))

    def __del__(self):

        if not self.initialized: return

        self._destroy()
