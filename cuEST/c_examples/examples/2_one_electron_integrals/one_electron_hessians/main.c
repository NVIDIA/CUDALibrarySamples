/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <cuest.h>

#include <helper_status.h>
#include <helper_workspaces.h>
#include <helper_xyz_parser.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>

/*
 * This sample demonstrates how to compute the one-electron Hessian contributions
 * for the overlap (S) and kinetic energy (T) matrices. There are two kinds of
 * Hessian compute calls:
 *
 *   Skeleton:  Contracts d^2 O_{mu nu} / dX_A dX_B against a density-like matrix
 *              and returns a dense (3*nA x 3*nB) Hessian block over two atom
 *              perturbation lists A and B.  For the SCF Hessian, O = S and the
 *              caller passes the energy-weighted density W; for the kinetic
 *              contribution the caller passes the density matrix D.
 *
 *   RHS:       Computes d O_{mu nu} / dX_A for a perturbation list and transforms
 *              the result into the MO basis via left/right coefficient matrices.
 *              The output is the CPSCF right-hand side; multiple coefficient-matrix
 *              pairs can be transformed in a single call.
 *
 * A "real" density matrix and MO coefficients are not available in this sample,
 * so synthetic random matrices are substituted.
 *
 * Both routines follow the standard cuEST pattern:
 *   1. Create parameter handle.
 *   2. WorkspaceQuery  -> allocate temporary workspace.
 *   3. Compute.
 *   4. Free workspace and parameter handle.
 */

/* Helper: fill a symmetric (N x N) GPU matrix with random values. */
static void fill_symmetric_matrix(double *d_A, uint64_t N)
{
    double *h_A = (double*) malloc(N * N * sizeof(double));
    if (!h_A) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    srand(0);
    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j <= i; j++) {
            double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
            h_A[i*N+j] = h_A[j*N+i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        }
    }
    cudaError_t err = cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    free(h_A);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed\n"); exit(EXIT_FAILURE); }
}

/* Helper: fill an arbitrary (M x N) GPU matrix with random values. */
static void fill_matrix(double *d_A, uint64_t M, uint64_t N)
{
    double *h_A = (double*) malloc(M * N * sizeof(double));
    if (!h_A) { fprintf(stderr, "malloc failed\n"); exit(EXIT_FAILURE); }
    srand(1);
    for (uint64_t i = 0; i < M * N; i++) {
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        h_A[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }
    cudaError_t err = cudaMemcpy(d_A, h_A, M * N * sizeof(double), cudaMemcpyHostToDevice);
    free(h_A);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy failed\n"); exit(EXIT_FAILURE); }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <xyz_file_path> <gbs_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* xyzFilePath = argv[1];
    const char* gbsFilePath = argv[2];

    /* Parse the XYZ file (convert angstrom -> bohr). */
    parsedXYZFile_t* xyzData = parseXYZFile(xyzFilePath, 1.0 / 0.52917720859);
    if (!xyzData) {
        fprintf(stderr, "Error: failed to parse file '%s'\n", xyzFilePath);
        exit(EXIT_FAILURE);
    }

    /**********************/
    /* cuEST Handle Setup */
    /**********************/

    cuestHandle_t handle;
    cuestHandleParameters_t handle_parameters;
    checkCuestErrors(cuestParametersCreate(CUEST_HANDLE_PARAMETERS, &handle_parameters));
    checkCuestErrors(cuestCreate(handle_parameters, &handle));
    checkCuestErrors(cuestParametersDestroy(CUEST_HANDLE_PARAMETERS, handle_parameters));

    /************************/
    /* cuEST AO Shell Setup */
    /************************/

    AtomShellData_t* shell_data = formAOShells(handle, xyzData, gbsFilePath, 1);

    uint64_t numAtoms        = shell_data->numAtoms;
    uint64_t numShellsTotal  = shell_data->numShellsTotal;
    uint64_t* numShellsPerAtom = shell_data->numShellsPerAtom;
    cuestAOShell_t* shells   = shell_data->shells;

    /************************/
    /* cuEST AO Basis Setup */
    /************************/

    cuestAOBasis_t basis;
    cuestAOBasisParameters_t basis_parameters;
    checkCuestErrors(cuestParametersCreate(CUEST_AOBASIS_PARAMETERS, &basis_parameters));

    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor =
        (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor =
        (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle, numAtoms, numShellsPerAtom, (const cuestAOShell_t*) shells,
        basis_parameters, persistentWorkspaceDescriptor, temporaryWorkspaceDescriptor, &basis));

    cuestWorkspace_t* persistentBasisWorkspace  = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryBasisWorkspace   = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestAOBasisCreate(
        handle, numAtoms, numShellsPerAtom, (const cuestAOShell_t*) shells,
        basis_parameters, persistentBasisWorkspace, temporaryBasisWorkspace, &basis));

    freeWorkspace(temporaryBasisWorkspace);
    checkCuestErrors(cuestParametersDestroy(CUEST_AOBASIS_PARAMETERS, basis_parameters));
    freeAOShellData(shell_data);

    /****************************/
    /* cuEST AO Pair List Setup */
    /****************************/

    cuestAOPairList_t pair_list;
    cuestAOPairListParameters_t pair_list_parameters;
    checkCuestErrors(cuestParametersCreate(CUEST_AOPAIRLIST_PARAMETERS, &pair_list_parameters));

    checkCuestErrors(cuestAOPairListCreateWorkspaceQuery(
        handle, basis, numAtoms, xyzData->xyzCPU, 1.0e-14,
        pair_list_parameters, persistentWorkspaceDescriptor, temporaryWorkspaceDescriptor,
        &pair_list));

    cuestWorkspace_t* persistentAOPairListWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryAOPairListWorkspace  = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestAOPairListCreate(
        handle, basis, numAtoms, xyzData->xyzCPU, 1.0e-14,
        pair_list_parameters, persistentAOPairListWorkspace, temporaryAOPairListWorkspace,
        &pair_list));

    checkCuestErrors(cuestParametersDestroy(CUEST_AOPAIRLIST_PARAMETERS, pair_list_parameters));
    freeWorkspace(temporaryAOPairListWorkspace);

    /******************************************/
    /* cuEST One-Electron Integral Plan Setup */
    /******************************************/

    cuestOEIntPlan_t oeint_plan;
    cuestOEIntPlanParameters_t oeint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(CUEST_OEINTPLAN_PARAMETERS, &oeint_plan_parameters));

    checkCuestErrors(cuestOEIntPlanCreateWorkspaceQuery(
        handle, basis, pair_list, oeint_plan_parameters,
        persistentWorkspaceDescriptor, temporaryWorkspaceDescriptor, &oeint_plan));

    cuestWorkspace_t* persistentOEIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryOEIntPlanWorkspace  = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestOEIntPlanCreate(
        handle, basis, pair_list, oeint_plan_parameters,
        persistentOEIntPlanWorkspace, temporaryOEIntPlanWorkspace, &oeint_plan));

    checkCuestErrors(cuestParametersDestroy(CUEST_OEINTPLAN_PARAMETERS, oeint_plan_parameters));
    freeWorkspace(temporaryOEIntPlanWorkspace);

    /************************************/
    /* Query system dimensions from AO basis */
    /************************************/

    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(handle, CUEST_AOBASIS, basis, CUEST_AOBASIS_NUM_AO,
                                &nao, sizeof(uint64_t)));

    printf("System: %llu atoms, %llu AOs\n",
           (unsigned long long) numAtoms, (unsigned long long) nao);

    /************************************/
    /* Build perturbation lists         */
    /************************************/

    /*
     * Use all atoms in both lists (full Hessian block).
     * In practice a caller may pass subsets to compute a rectangular sub-block.
     */
    uint64_t* perturbationsA = (uint64_t*) malloc(numAtoms * sizeof(uint64_t));
    uint64_t* perturbationsB = (uint64_t*) malloc(numAtoms * sizeof(uint64_t));
    if (!perturbationsA || !perturbationsB) {
        fprintf(stderr, "Failed to allocate perturbation lists\n");
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < numAtoms; i++) {
        perturbationsA[i] = i;
        perturbationsB[i] = i;
    }

    /************************************/
    /* Allocate density / MO matrices   */
    /************************************/

    /* Energy-weighted density W for skeleton (used in place of a real W here). */
    double *d_W;
    if (cudaMalloc((void**) &d_W, nao * nao * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for W\n"); exit(EXIT_FAILURE);
    }
    fill_symmetric_matrix(d_W, nao);

    /* Hessian output blocks: (3*natom x 3*natom) row-major. */
    double *d_d2SdR2, *d_d2TdR2;
    uint64_t hess_size = 3 * numAtoms * 3 * numAtoms;
    if (cudaMalloc((void**) &d_d2SdR2, hess_size * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d2SdR2\n"); exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_d2TdR2, hess_size * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d2TdR2\n"); exit(EXIT_FAILURE);
    }

    /**************************************/
    /* Overlap Hessian Skeleton           */
    /**************************************/

    cuestOverlapHessianSkeletonComputeParameters_t overlap_skeleton_params;
    checkCuestErrors(cuestParametersCreate(
        CUEST_OVERLAPHESSIANSKELETONCOMPUTE_PARAMETERS, &overlap_skeleton_params));

    checkCuestErrors(cuestOverlapHessianSkeletonComputeWorkspaceQuery(
        handle, oeint_plan, overlap_skeleton_params,
        temporaryWorkspaceDescriptor,
        numAtoms, perturbationsA,
        numAtoms, perturbationsB,
        d_W, d_d2SdR2));

    cuestWorkspace_t* temporaryOverlapSkeletonWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestOverlapHessianSkeletonCompute(
        handle, oeint_plan, overlap_skeleton_params,
        temporaryOverlapSkeletonWorkspace,
        numAtoms, perturbationsA,
        numAtoms, perturbationsB,
        d_W, d_d2SdR2));

    freeWorkspace(temporaryOverlapSkeletonWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_OVERLAPHESSIANSKELETONCOMPUTE_PARAMETERS, overlap_skeleton_params));

    printf("Overlap Hessian skeleton: computed (%llu x %llu) block\n",
           (unsigned long long)(3 * numAtoms), (unsigned long long)(3 * numAtoms));

    /**************************************/
    /* Kinetic Hessian Skeleton           */
    /**************************************/

    cuestKineticHessianSkeletonComputeParameters_t kinetic_skeleton_params;
    checkCuestErrors(cuestParametersCreate(
        CUEST_KINETICHESSIANSKELETONCOMPUTE_PARAMETERS, &kinetic_skeleton_params));

    checkCuestErrors(cuestKineticHessianSkeletonComputeWorkspaceQuery(
        handle, oeint_plan, kinetic_skeleton_params,
        temporaryWorkspaceDescriptor,
        numAtoms, perturbationsA,
        numAtoms, perturbationsB,
        d_W, d_d2TdR2));

    cuestWorkspace_t* temporaryKineticSkeletonWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestKineticHessianSkeletonCompute(
        handle, oeint_plan, kinetic_skeleton_params,
        temporaryKineticSkeletonWorkspace,
        numAtoms, perturbationsA,
        numAtoms, perturbationsB,
        d_W, d_d2TdR2));

    freeWorkspace(temporaryKineticSkeletonWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_KINETICHESSIANSKELETONCOMPUTE_PARAMETERS, kinetic_skeleton_params));

    printf("Kinetic Hessian skeleton: computed (%llu x %llu) block\n",
           (unsigned long long)(3 * numAtoms), (unsigned long long)(3 * numAtoms));

    /**************************************/
    /* RHS coefficient matrices           */
    /**************************************/

    /*
     * The RHS transform requires left and right MO coefficient matrices.
     * Here we use a single pair:
     *   C_left  (nmo x nao): the full coefficient matrix (nmo = nao for simplicity)
     *   C_right (nocc x nao): occupied orbitals (nocc = nao/2, rounded down)
     *
     * In practice these come from a preceding SCF calculation.
     */
    uint64_t nmo  = nao;
    uint64_t nocc = nao / 2 > 0 ? nao / 2 : 1;

    double *d_C_left, *d_C_right;
    if (cudaMalloc((void**) &d_C_left,  nmo  * nao * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for C_left\n"); exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_C_right, nocc * nao * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for C_right\n"); exit(EXIT_FAILURE);
    }
    fill_matrix(d_C_left,  nmo,  nao);
    fill_matrix(d_C_right, nocc, nao);

    /* One coefficient-matrix pair. */
    uint64_t numCoefficientMatrices = 1;
    uint64_t numLeftOrbitals[1]  = { nmo  };
    uint64_t numRightOrbitals[1] = { nocc };

    /* RHS output tensor for this pair: (3*natom x nmo x nocc). */
    double *d_rhs_S, *d_rhs_T;
    uint64_t rhs_size = 3 * numAtoms * nmo * nocc;
    if (cudaMalloc((void**) &d_rhs_S, rhs_size * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for rhs_S\n"); exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_rhs_T, rhs_size * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for rhs_T\n"); exit(EXIT_FAILURE);
    }

    /**************************************/
    /* Overlap Hessian RHS                */
    /**************************************/

    cuestOverlapHessianRHSComputeParameters_t overlap_rhs_params;
    checkCuestErrors(cuestParametersCreate(
        CUEST_OVERLAPHESSIANRHSCOMPUTE_PARAMETERS, &overlap_rhs_params));

    checkCuestErrors(cuestOverlapHessianRHSComputeWorkspaceQuery(
        handle, oeint_plan, overlap_rhs_params,
        temporaryWorkspaceDescriptor,
        numAtoms, perturbationsA,
        numCoefficientMatrices, numLeftOrbitals, numRightOrbitals,
        d_C_left, d_C_right, d_rhs_S));

    cuestWorkspace_t* temporaryOverlapRHSWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestOverlapHessianRHSCompute(
        handle, oeint_plan, overlap_rhs_params,
        temporaryOverlapRHSWorkspace,
        numAtoms, perturbationsA,
        numCoefficientMatrices, numLeftOrbitals, numRightOrbitals,
        d_C_left, d_C_right, d_rhs_S));

    freeWorkspace(temporaryOverlapRHSWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_OVERLAPHESSIANRHSCOMPUTE_PARAMETERS, overlap_rhs_params));

    printf("Overlap Hessian RHS: computed (%llu x %llu x %llu) tensor\n",
           (unsigned long long)(3 * numAtoms),
           (unsigned long long) nmo,
           (unsigned long long) nocc);

    /**************************************/
    /* Kinetic Hessian RHS                */
    /**************************************/

    cuestKineticHessianRHSComputeParameters_t kinetic_rhs_params;
    checkCuestErrors(cuestParametersCreate(
        CUEST_KINETICHESSIANRHSCOMPUTE_PARAMETERS, &kinetic_rhs_params));

    checkCuestErrors(cuestKineticHessianRHSComputeWorkspaceQuery(
        handle, oeint_plan, kinetic_rhs_params,
        temporaryWorkspaceDescriptor,
        numAtoms, perturbationsA,
        numCoefficientMatrices, numLeftOrbitals, numRightOrbitals,
        d_C_left, d_C_right, d_rhs_T));

    cuestWorkspace_t* temporaryKineticRHSWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestKineticHessianRHSCompute(
        handle, oeint_plan, kinetic_rhs_params,
        temporaryKineticRHSWorkspace,
        numAtoms, perturbationsA,
        numCoefficientMatrices, numLeftOrbitals, numRightOrbitals,
        d_C_left, d_C_right, d_rhs_T));

    freeWorkspace(temporaryKineticRHSWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_KINETICHESSIANRHSCOMPUTE_PARAMETERS, kinetic_rhs_params));

    printf("Kinetic Hessian RHS: computed (%llu x %llu x %llu) tensor\n",
           (unsigned long long)(3 * numAtoms),
           (unsigned long long) nmo,
           (unsigned long long) nocc);

    /**************************************/
    /* Cleanup                            */
    /**************************************/

    cudaFree((void*) d_rhs_T);
    cudaFree((void*) d_rhs_S);
    cudaFree((void*) d_C_right);
    cudaFree((void*) d_C_left);
    cudaFree((void*) d_d2TdR2);
    cudaFree((void*) d_d2SdR2);
    cudaFree((void*) d_W);

    free(perturbationsB);
    free(perturbationsA);

    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestOEIntPlanDestroy(oeint_plan));
    freeWorkspace(persistentOEIntPlanWorkspace);

    checkCuestErrors(cuestAOPairListDestroy(pair_list));
    freeWorkspace(persistentAOPairListWorkspace);

    checkCuestErrors(cuestAOBasisDestroy(basis));
    freeWorkspace(persistentBasisWorkspace);

    checkCuestErrors(cuestDestroy(handle));

    freeParsedXYZFile(xyzData);

    return 0;
}
