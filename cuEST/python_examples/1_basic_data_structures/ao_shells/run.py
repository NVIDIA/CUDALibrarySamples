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

import sys
from pathlib import Path

# Inject the directory where the example helper utilities live
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helpers.cuda_utils import WorkspaceDescriptor, Workspace
from helpers.utilities import normalize_shell_coefficients

# This is the def2-SVP basis set definition for O and H

H_basis = {
    'n_shells' : 3,
    'shell_types' : [0, 0, 1],
    'primitive_counts' : [3, 1, 1],
    'primitive_offsets' : [0, 3, 4],
    'exponents' : [
        13.0107010, 1.9622572, 0.44453796,
        0.12194962,
        0.8000000
        ],
    'coefficients' : [
        0.019682158, 0.13796524, 0.47831935,
        1.0,
        1.0
        ],
    }

O_basis = {
    'n_shells' : 6,
    'shell_types' : [0, 0, 0, 1, 1, 2],
    'primitive_counts' : [5, 1, 1, 3, 1, 1],
    'primitive_offsets' : [0, 5, 6, 7, 10, 11],
    'exponents' : [
        2266.1767785, 340.87010191, 77.363135167, 21.479644940, 6.6589433124,
        0.80975975668,
        0.25530772234,
        17.721504317, 3.8635505440, 1.0480920883,
        0.27641544411,
        1.2
        ],
    'coefficients' : [
        -0.0053431809926, -0.039890039230, -0.17853911985, -0.46427684959, -0.44309745172,
        1.0,
        1.0,
        0.043394573193, 0.23094120765, 0.51375311064,
        1.0,
        1.0
        ],
    }


def cuest_check(
    title,
    return_code
    ):

    if return_code != ce.CuestStatus.CUEST_STATUS_SUCCESS:
        raise RuntimeError(f"{title} failed with code {return_code}")



cuest_handle_parameters = ce.cuestHandleParameters()
cuest_check("Parameters Create",
    ce.cuestParametersCreate(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        outParameters=cuest_handle_parameters,
        )
    )
cuest_handle = ce.cuestHandle()
cuest_check("Cuest Handle Create",
    ce.cuestCreate(
        parameters=cuest_handle_parameters,
        handle=cuest_handle,
        )
    )
cuest_check("Parameters Destroy",
    ce.cuestParametersDestroy(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        parameters=cuest_handle_parameters,
        )
    )


aoshell_parameters = ce.cuestAOShellParameters()
cuest_check("AO Shell Parameters Create",
    ce.cuestParametersCreate(
        parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
        outParameters=aoshell_parameters,
        )
    )

# Make a water molecule as an example
basis_is_pure = 1
shells = []
atom_symbols = ["O", "H", "H"]
atom_bases = [O_basis, H_basis, H_basis]
for shellinfo in atom_bases:
    this_atom_shells = []

    for offset, count, L in zip(
        shellinfo['primitive_offsets'],
        shellinfo['primitive_counts'],
        shellinfo['shell_types']
        ):

        exponents = shellinfo['exponents'][offset:offset+count]
        coefficients = shellinfo['coefficients'][offset:offset+count]
        normalized_coefficients = normalize_shell_coefficients(
            coefficients=coefficients,
            exponents=exponents,
            L=L,
            normalization=1.0,
            )
        aoshell_handle = ce.cuestAOShellHandle()
        cuest_check("AO Shell Create",
            ce.cuestAOShellCreate(
                handle=cuest_handle,
                isPure=basis_is_pure,
                L=L,
                numPrimitive=len(exponents),
                exponents=exponents,
                coefficients=normalized_coefficients,
                parameters=aoshell_parameters,
                outShell=aoshell_handle
                )
            )
        this_atom_shells.append(aoshell_handle)
    shells.append(this_atom_shells)

cuest_check("AO Shell Parameters Destroy",
    ce.cuestParametersDestroy(
        parametersType=ce.CuestParametersType.CUEST_AOSHELL_PARAMETERS,
        parameters=aoshell_parameters,
        )
    )

# Query each shell for its metadata, and print
for atom_number, atom_shells in enumerate(shells):

    print(f"Atom {atom_number+1} ({atom_symbols[atom_number]})")

    for shell_count, atom_shell in enumerate(atom_shells):
        print(f"\tShell {shell_count+1}")
        is_pure = ce.data_int32_t()
        cuest_check("Query IS_PURE",
            ce.cuestQuery(
                handle=cuest_handle,
                type=ce.CuestType.CUEST_AOSHELL,
                object=atom_shell,
                attribute=ce.CuestAOShellAttributes.CUEST_AOSHELL_IS_PURE,
                attributeValue=is_pure,
                )
            )
        print(f"\t\tIs Pure                  : {is_pure.value}")

        L = ce.data_uint64_t()
        cuest_check("Query L",
            ce.cuestQuery(
                handle=cuest_handle,
                type=ce.CuestType.CUEST_AOSHELL,
                object=atom_shell,
                attribute=ce.CuestAOShellAttributes.CUEST_AOSHELL_L,
                attributeValue=L,
                )
            )
        print(f"\t\tAngular Momentum         : {L.value}")

        nprim = ce.data_uint64_t()
        cuest_check("Query Num Primitives",
            ce.cuestQuery(
                handle=cuest_handle,
                type=ce.CuestType.CUEST_AOSHELL,
                object=atom_shell,
                attribute=ce.CuestAOShellAttributes.CUEST_AOSHELL_NUM_PRIMITIVE,
                attributeValue=nprim,
                )
            )
        print(f"\t\tNum. Primitives          : {nprim.value}")

        nao = ce.data_uint64_t()
        cuest_check("Query Num Basis Functions",
            ce.cuestQuery(
                handle=cuest_handle,
                type=ce.CuestType.CUEST_AOSHELL,
                object=atom_shell,
                attribute=ce.CuestAOShellAttributes.CUEST_AOSHELL_NUM_AO,
                attributeValue=nao,
                )
            )
        print(f"\t\tNum. Basis Functions     : {nao.value}")

        npure = ce.data_uint64_t()
        cuest_check("Query Num Pure Functions",
            ce.cuestQuery(
                handle=cuest_handle,
                type=ce.CuestType.CUEST_AOSHELL,
                object=atom_shell,
                attribute=ce.CuestAOShellAttributes.CUEST_AOSHELL_NUM_PURE,
                attributeValue=npure,
                )
            )
        print(f"\t\tNum. Pure Functions      : {npure.value}")

        ncart = ce.data_uint64_t()
        cuest_check("Query Num Cart Functions",
            ce.cuestQuery(
                handle=cuest_handle,
                type=ce.CuestType.CUEST_AOSHELL,
                object=atom_shell,
                attribute=ce.CuestAOShellAttributes.CUEST_AOSHELL_NUM_CART,
                attributeValue=npure,
                )
            )
        print(f"\t\tNum. Cartesian Functions : {npure.value}")

# Destroy the shells
for atom_shells in shells:
    for atom_shell in atom_shells:
        cuest_check("Destroy AO Shell",
            ce.cuestAOShellDestroy(
                handle=atom_shell,
                )
            )

# Delete the cuEST handle
cuest_check('Destroy Cuest Handle',
    ce.cuestDestroy(
        handle=cuest_handle,
        )
    )
