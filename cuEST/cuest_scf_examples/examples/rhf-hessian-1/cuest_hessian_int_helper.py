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


import cuest.bindings as ce

from cuest_scf.gpu_matrix import GPUMatrix

from cuest_scf.cuest_handle import CuestHandle
from cuest_scf.cuest_oe_int_plan import CuestOEIntPlan
from cuest_scf.cuest_workspace import CuestWorkspace
from cuest_scf.cuest_workspace_descriptor import CuestWorkspaceDescriptor

class CuestHessianIntegralHelper(object):
    def __init__(
        self,
        *,
        cuest_handle : CuestHandle,
        oe_int_plan : CuestOEIntPlan,
        ):

        self.cuest_handle = cuest_handle
        self.oe_int_plan = oe_int_plan

    def compute_overlap_hessian_skeleton(
        self,
        perturbations_a : list,
        perturbations_b : list,
        density : GPUMatrix,
        out_hessian : GPUMatrix,
        ):

        # Contracts the second derivatives of the overlap matrix with density for
        # all atom pairs from perturbations_a and perturbations_b. Both lists must
        # be sorted in ascending order with no duplicates.
        if out_hessian.shape != (3 * len(perturbations_a), 3 * len(perturbations_b)):
            raise RuntimeError('out_hessian has the wrong shape')

        skeleton_parameters = ce.cuestOverlapHessianSkeletonComputeParameters()
        status = ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPHESSIANSKELETONCOMPUTE_PARAMETERS,
            outParameters=skeleton_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapHessianSkeletonComputeParametersCreate failed')

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        density_handle = ce.Pointer(density.pointer)
        out_hessian_handle = ce.Pointer(out_hessian.pointer)

        status = ce.cuestOverlapHessianSkeletonComputeWorkspaceQuery(
            handle=self.cuest_handle.handle,
            plan=self.oe_int_plan.oe_int_plan_handle,
            parameters=skeleton_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            perturbationsA=perturbations_a,
            perturbationsB=perturbations_b,
            densityMatrix=density_handle,
            outHessianBlock=out_hessian_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapHessianSkeletonComputeWorkspaceQuery failed', status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestOverlapHessianSkeletonCompute(
            handle=self.cuest_handle.handle,
            plan=self.oe_int_plan.oe_int_plan_handle,
            parameters=skeleton_parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            perturbationsA=perturbations_a,
            perturbationsB=perturbations_b,
            densityMatrix=density_handle,
            outHessianBlock=out_hessian_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapHessianSkeletonCompute failed')

        status = ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPHESSIANSKELETONCOMPUTE_PARAMETERS,
            parameters=skeleton_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapHessianSkeletonComputeParametersDestroy failed')

    def compute_kinetic_hessian_skeleton(
        self,
        perturbations_a : list,
        perturbations_b : list,
        density : GPUMatrix,
        out_hessian : GPUMatrix,
        ):

        # The kinetic counterpart of compute_overlap_hessian_skeleton.
        if out_hessian.shape != (3 * len(perturbations_a), 3 * len(perturbations_b)):
            raise RuntimeError('out_hessian has the wrong shape')

        skeleton_parameters = ce.cuestKineticHessianSkeletonComputeParameters()
        status = ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_KINETICHESSIANSKELETONCOMPUTE_PARAMETERS,
            outParameters=skeleton_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticHessianSkeletonComputeParametersCreate failed')

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        density_handle = ce.Pointer(density.pointer)
        out_hessian_handle = ce.Pointer(out_hessian.pointer)

        status = ce.cuestKineticHessianSkeletonComputeWorkspaceQuery(
            handle=self.cuest_handle.handle,
            plan=self.oe_int_plan.oe_int_plan_handle,
            parameters=skeleton_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            perturbationsA=perturbations_a,
            perturbationsB=perturbations_b,
            densityMatrix=density_handle,
            outHessianBlock=out_hessian_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticHessianSkeletonComputeWorkspaceQuery failed', status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestKineticHessianSkeletonCompute(
            handle=self.cuest_handle.handle,
            plan=self.oe_int_plan.oe_int_plan_handle,
            parameters=skeleton_parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            perturbationsA=perturbations_a,
            perturbationsB=perturbations_b,
            densityMatrix=density_handle,
            outHessianBlock=out_hessian_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticHessianSkeletonCompute failed')

        status = ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_KINETICHESSIANSKELETONCOMPUTE_PARAMETERS,
            parameters=skeleton_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticHessianSkeletonComputeParametersDestroy failed')

    def compute_overlap_hessian_rhs(
        self,
        perturbations : list,
        C_left : GPUMatrix,
        nmo_c1 : list,
        C_right : GPUMatrix,
        nmo_c2 : list,
        out_tensors : GPUMatrix,
        ):

        # Transforms the first derivatives of the overlap matrix to the MO basis
        # for several coefficient-matrix pairs in one call. C_left and C_right hold
        # the left and right matrices of the pairs back to back, and out_tensors
        # receives a [3 * len(perturbations), nmo_c1[i], nmo_c2[i]] tensor for each
        # pair i, back to back.
        if len(nmo_c1) != len(nmo_c2):
            raise RuntimeError('nmo_c1 and nmo_c2 describe different numbers of pairs')
        nout = 3 * len(perturbations) * sum(n1 * n2 for n1, n2 in zip(nmo_c1, nmo_c2))
        if out_tensors.size < nout:
            raise RuntimeError('out_tensors is too small')

        rhs_parameters = ce.cuestOverlapHessianRHSComputeParameters()
        status = ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPHESSIANRHSCOMPUTE_PARAMETERS,
            outParameters=rhs_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapHessianRHSComputeParametersCreate failed')

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        C_left_handle = ce.Pointer(C_left.pointer)
        C_right_handle = ce.Pointer(C_right.pointer)
        out_tensors_handle = ce.Pointer(out_tensors.pointer)

        status = ce.cuestOverlapHessianRHSComputeWorkspaceQuery(
            handle=self.cuest_handle.handle,
            plan=self.oe_int_plan.oe_int_plan_handle,
            parameters=rhs_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            perturbations=perturbations,
            numLeftOrbitals=nmo_c1,
            numRightOrbitals=nmo_c2,
            leftCoefficientMatrices=C_left_handle,
            rightCoefficientMatrices=C_right_handle,
            outTensors=out_tensors_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapHessianRHSComputeWorkspaceQuery failed', status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestOverlapHessianRHSCompute(
            handle=self.cuest_handle.handle,
            plan=self.oe_int_plan.oe_int_plan_handle,
            parameters=rhs_parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            perturbations=perturbations,
            numLeftOrbitals=nmo_c1,
            numRightOrbitals=nmo_c2,
            leftCoefficientMatrices=C_left_handle,
            rightCoefficientMatrices=C_right_handle,
            outTensors=out_tensors_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapHessianRHSCompute failed')

        status = ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_OVERLAPHESSIANRHSCOMPUTE_PARAMETERS,
            parameters=rhs_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestOverlapHessianRHSComputeParametersDestroy failed')

    def compute_kinetic_hessian_rhs(
        self,
        perturbations : list,
        C_left : GPUMatrix,
        nmo_c1 : list,
        C_right : GPUMatrix,
        nmo_c2 : list,
        out_tensors : GPUMatrix,
        ):

        # The kinetic counterpart of compute_overlap_hessian_rhs.
        if len(nmo_c1) != len(nmo_c2):
            raise RuntimeError('nmo_c1 and nmo_c2 describe different numbers of pairs')
        nout = 3 * len(perturbations) * sum(n1 * n2 for n1, n2 in zip(nmo_c1, nmo_c2))
        if out_tensors.size < nout:
            raise RuntimeError('out_tensors is too small')

        rhs_parameters = ce.cuestKineticHessianRHSComputeParameters()
        status = ce.cuestParametersCreate(
            parametersType=ce.CuestParametersType.CUEST_KINETICHESSIANRHSCOMPUTE_PARAMETERS,
            outParameters=rhs_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticHessianRHSComputeParametersCreate failed')

        temporary_workspace_descriptor = CuestWorkspaceDescriptor()

        C_left_handle = ce.Pointer(C_left.pointer)
        C_right_handle = ce.Pointer(C_right.pointer)
        out_tensors_handle = ce.Pointer(out_tensors.pointer)

        status = ce.cuestKineticHessianRHSComputeWorkspaceQuery(
            handle=self.cuest_handle.handle,
            plan=self.oe_int_plan.oe_int_plan_handle,
            parameters=rhs_parameters,
            temporaryWorkspaceDescriptor=temporary_workspace_descriptor.pointer,
            perturbations=perturbations,
            numLeftOrbitals=nmo_c1,
            numRightOrbitals=nmo_c2,
            leftCoefficientMatrices=C_left_handle,
            rightCoefficientMatrices=C_right_handle,
            outTensors=out_tensors_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticHessianRHSComputeWorkspaceQuery failed', status)

        temporary_workspace = CuestWorkspace(workspaceDescriptor=temporary_workspace_descriptor)

        status = ce.cuestKineticHessianRHSCompute(
            handle=self.cuest_handle.handle,
            plan=self.oe_int_plan.oe_int_plan_handle,
            parameters=rhs_parameters,
            temporaryWorkspace=temporary_workspace.pointer,
            perturbations=perturbations,
            numLeftOrbitals=nmo_c1,
            numRightOrbitals=nmo_c2,
            leftCoefficientMatrices=C_left_handle,
            rightCoefficientMatrices=C_right_handle,
            outTensors=out_tensors_handle,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticHessianRHSCompute failed')

        status = ce.cuestParametersDestroy(
            parametersType=ce.CuestParametersType.CUEST_KINETICHESSIANRHSCOMPUTE_PARAMETERS,
            parameters=rhs_parameters,
            )
        if status != ce.CuestStatus.CUEST_STATUS_SUCCESS:
            raise RuntimeError('cuestKineticHessianRHSComputeParametersDestroy failed')
