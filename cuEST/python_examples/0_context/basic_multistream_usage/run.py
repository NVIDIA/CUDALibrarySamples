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

def cuest_check(
    title,
    return_code
    ):

    if return_code != ce.CuestStatus.CUEST_STATUS_SUCCESS:
        raise RuntimeError(f"{title} failed with code {return_code}")


cuest_handle_parameters = ce.cuestHandleParameters()
cuest_check('Create Handle Params',
    ce.cuestParametersCreate(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        outParameters=cuest_handle_parameters,
        )
    )

# Create a list of 2 cuEST handles.  Each one will hold its own unique
# resources Stream, cuBLAS, cuSolver.
cuest_handles = []
for _ in range(2):
    cuest_handle = ce.cuestHandle()
    cuest_check('Create Cuest Handle',
        ce.cuestCreate(
            parameters=cuest_handle_parameters,
            handle=cuest_handle,
            )
        )
    cuest_handles.append(cuest_handle)


cuest_check('Destroy Handle Params',
    ce.cuestParametersDestroy(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        parameters=cuest_handle_parameters,
        )
    )


# The cuEST handles are ready to be used for calculations


# Free both of the handles when finished, to free up resources
for cuest_handle in cuest_handles:
    cuest_check('Destroy Cuest Handle',
        ce.cuestDestroy(
            handle=cuest_handle,
            )
        )
