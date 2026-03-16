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

from .cuest_handle import CuestHandle
from .cuest_pcm_int_plan import CuestPCMIntPlan

from .cuest_workspace_descriptor import CuestWorkspaceDescriptor
from .cuest_workspace import CuestWorkspace
from .cuest_parameters import CuestParameters
from .cuest_results import CuestResults

import numpy as np

class CuestPCMIntCompute(object):

    @staticmethod
    def compute_pcm_energy_and_potential(
        *,
        handle : CuestHandle,
        pcm_int_plan : CuestPCMIntPlan,
        Dptr,
        inQptr,
        outQptr,
        Vptr,
        ):

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(Dptr)

        inQptr2 = ce.Pointer()
        inQptr2.value = np.intp(inQptr)

        outQptr2 = ce.Pointer()
        outQptr2.value = np.intp(outQptr)

        Vptr2 = ce.Pointer()
        Vptr2.value = np.intp(Vptr)

        # => Workspace query <= #

        pcm_results_handle = CuestResults(resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS, results_handle=ce.cuestPCMResultsHandle())
        # pcm parameters
        pcm_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS)

        convergence_threshold = ce.data_double(pcm_int_plan.convergence_tol)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS,
            parameters=pcm_compute_parameters.parameters,
            attribute=ce.CuestPCMPotentialComputeParametersAttributes.CUEST_PCMPOTENTIALCOMPUTE_PARAMETERS_CONVERGENCE_THRESHOLD,
            attributeValue=convergence_threshold,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed: %d' % status)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestPCMPotentialComputeWorkspaceQuery(
            handle=handle.handle,
            plan=pcm_int_plan.pcm_int_plan_handle,
            parameters=pcm_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=Dptr2,
            inQ=inQptr2,
            outQ=outQptr2,
            outPCMResults=pcm_results_handle.results,
            outPCMPotentialMatrix=Vptr2,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPCMPotentialComputeWorkspaceQuery failed: %d' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestPCMPotentialCompute(
            handle=handle.handle,
            plan=pcm_int_plan.pcm_int_plan_handle,
            parameters=pcm_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            densityMatrix=Dptr2,
            inQ=inQptr2,
            outQ=outQptr2,
            outPCMResults=pcm_results_handle.results,
            outPCMPotentialMatrix=Vptr2,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPCMPotentialCompute failed: %d' % status)
        del pcm_compute_parameters

        # get results
        Epcm = ce.data_double()
        status = ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle.results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_PCM_DIELECTRIC_ENERGY,
            attributeValue=Epcm,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestResultsQuery failed: %d' % status)

        converged = ce.data_int32_t()
        status = ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle.results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED,
            attributeValue=converged,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestResultsQuery failed: %d' % status)

        residual = ce.data_double()
        status = ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle.results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED_RESIDUAL,
            attributeValue=residual,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestResultsQuery failed: %d' % status)

        del pcm_results_handle
        return Epcm.value, converged.value, residual.value

    @staticmethod
    def compute_pcm_gradient(
        *,
        handle : CuestHandle,
        pcm_int_plan : CuestPCMIntPlan,
        Dptr,
        inQptr,
        outQptr,
        Gptr,
        ):

        Dptr2 = ce.Pointer()
        Dptr2.value = np.intp(Dptr)

        inQptr2 = ce.Pointer()
        inQptr2.value = np.intp(inQptr)

        outQptr2 = ce.Pointer()
        outQptr2.value = np.intp(outQptr)

        Gptr2 = ce.Pointer()
        Gptr2.value = np.intp(Gptr)

        pcm_results_handle = CuestResults(resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS, results_handle=ce.cuestPCMResultsHandle())
        # pcm parameters
        pcm_derivative_compute_parameters = CuestParameters(parametersType=ce.CuestParametersType.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS)

        convergence_threshold = ce.data_double(pcm_int_plan.convergence_tol)
        status = ce.cuestParametersConfigure(
            parametersType=ce.CuestParametersType.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS,
            parameters=pcm_derivative_compute_parameters.parameters,
            attribute=ce.CuestPCMDerivativeComputeParametersAttributes.CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS_CONVERGENCE_THRESHOLD,
            attributeValue=convergence_threshold,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestParametersConfigure failed: %d' % status)

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        status = ce.cuestPCMDerivativeComputeWorkspaceQuery(
            handle=handle.handle,
            plan=pcm_int_plan.pcm_int_plan_handle,
            parameters=pcm_derivative_compute_parameters.parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            densityMatrix=Dptr2,
            inQ=inQptr2,
            outQ=outQptr2,
            outPCMResults=pcm_results_handle.results,
            outPCMGradient=Gptr2,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPCMDerivativeComputeWorkspaceQuery failed: %d' % status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestPCMDerivativeCompute(
            handle=handle.handle,
            plan=pcm_int_plan.pcm_int_plan_handle,
            parameters=pcm_derivative_compute_parameters.parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            densityMatrix=Dptr2,
            inQ=inQptr2,
            outQ=outQptr2,
            outPCMResults=pcm_results_handle.results,
            outPCMGradient=Gptr2,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestPCMDerivativeCompute failed: %d' % status)
        del pcm_derivative_compute_parameters

        # get results
        converged = ce.data_int32_t()
        status = ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle.results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED,
            attributeValue=converged,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestResultsQuery failed: %d' % status)

        residual = ce.data_double()
        status = ce.cuestResultsQuery(
            resultsType=ce.CuestResultsType.CUEST_PCM_RESULTS,
            results=pcm_results_handle.results,
            attribute=ce.CuestPCMResultAttributes.CUEST_PCMRESULT_CONVERGED_RESIDUAL,
            attributeValue=residual,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestResultsQuery failed: %d' % status)

        del pcm_results_handle
        return converged.value, residual.value
