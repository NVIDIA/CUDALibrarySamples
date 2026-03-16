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


# Create default HandleParameters object to parameterize the cuEST handle.
cuest_handle_parameters = ce.cuestHandleParameters()

cuest_check('Create Handle Params',
    ce.cuestParametersCreate(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        outParameters=cuest_handle_parameters,
        )
    )

# Make a handle to hold a C uint64_t type, and populate it with the maximum number
# of Gauss-Hermite quadrature points supported by by the cuEST handle by default.
maxgh_handle = ce.data_uint64_t()
cuest_check('Query Handle Max Num GH',
    ce.cuestParametersQuery(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        parameters=cuest_handle_parameters,
        attribute=ce.CuestHandleParametersAttributes.CUEST_HANDLE_PARAMETERS_MAX_GAUSS_HERMITE,
        attributeValue=maxgh_handle,
        )
    )
print('Handle Max Num GH', maxgh_handle.value)

# Similarly, query for the maximum angular momentum supported for
# Cartesian->Spherical transforms in this cuEST handle instance.
maxl_handle = ce.data_uint64_t()
cuest_check('Query Handle Max L',
    ce.cuestParametersQuery(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        parameters=cuest_handle_parameters,
        attribute=ce.CuestHandleParametersAttributes.CUEST_HANDLE_PARAMETERS_MAX_L_SOLID_HARMONIC,
        attributeValue=maxl_handle,
        )
    )
print('Handle Max L', maxl_handle.value)

# ce.cuestParametersConfigure could be called here to override the default
# values if required

cuest_handle = ce.cuestHandle()
cuest_check('Create Cuest Handle',
    ce.cuestCreate(
        parameters=cuest_handle_parameters,
        handle=cuest_handle,
        )
    )

# Once the handle itself has been created, its parameters can be destroyed
cuest_check('Destroy Handle Params',
    ce.cuestParametersDestroy(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        parameters=cuest_handle_parameters,
        )
    )


# The handle is ready to be used for calculations


# Destroying the handle will also destoy CUDA stream, cuBLAS, and cuSolver
# handles held by the cuEST handle.
cuest_check('Destroy Cuest Handle',
    ce.cuestDestroy(
        handle=cuest_handle,
        )
    )
