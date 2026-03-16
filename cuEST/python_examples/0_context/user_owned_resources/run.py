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
from helpers.cuda_utils import make_stream,\
                               make_cublas_handle, free_cublas_handle,\
                               make_cusolver_handle, free_cusolver_handle,\
                               set_cublas_stream, set_cusolver_stream

def cuest_check(
    title,
    return_code
    ):

    if return_code != ce.CuestStatus.CUEST_STATUS_SUCCESS:
        raise RuntimeError(f"{title} failed with code {return_code}")


stream = make_stream()
cublas_c_handle = make_cublas_handle()
cusolver_c_handle = make_cusolver_handle()

set_cublas_stream(
    handle=cublas_c_handle,
    stream=stream.__int__(),
    )

set_cusolver_stream(
    handle=cusolver_c_handle,
    stream=stream.__int__(),
    )

# Create default HandleParameters object to parameterize the cuEST handle.
cuest_handle_parameters = ce.cuestHandleParameters()

cuest_check('Create Handle Params',
    ce.cuestParametersCreate(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        outParameters=cuest_handle_parameters,
        )
    )

# => Configure the cuEST handle paramters <= #

stream_handle = ce.data_cudaStream_t(stream)
cuest_check('Configure Handle Stream',
    ce.cuestParametersConfigure(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        parameters=cuest_handle_parameters,
        attribute=ce.CuestHandleParametersAttributes.CUEST_HANDLE_PARAMETERS_CUDASTREAM,
        attributeValue=stream_handle,
        )
    )

cublas_handle = ce.data_cublasHandle_t(cublas_c_handle.value)
cuest_check('Configure Handle cuBLAS',
    ce.cuestParametersConfigure(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        parameters=cuest_handle_parameters,
        attribute=ce.CuestHandleParametersAttributes.CUEST_HANDLE_PARAMETERS_CUBLAS,
        attributeValue=cublas_handle,
        )
    )

cusolver_handle = ce.data_cusolverDnHandle_t(cusolver_c_handle.value)
cuest_check('Configure Handle cuSolver',
    ce.cuestParametersConfigure(
        parametersType=ce.CuestParametersType.CUEST_HANDLE_PARAMETERS,
        parameters=cuest_handle_parameters,
        attribute=ce.CuestHandleParametersAttributes.CUEST_HANDLE_PARAMETERS_CUSOLVER,
        attributeValue=cusolver_handle,
        )
    )

# => Make cuEST Handle <= #

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

free_cublas_handle(
    handle=cublas_c_handle,
    )
free_cusolver_handle(
    handle=cusolver_c_handle,
    )
# A key difference between using cuda.bindings to wrap CUDA (as the stream is
# in this example) vs. using ctypes (used for cuBLAS and cuSolver) is that the
# former takes care of memory management automatically, so no call to free the
# stream is needed.  We can optionally free it as follows:
del stream
