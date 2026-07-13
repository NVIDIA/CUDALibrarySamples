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
#include <inttypes.h>
#include <string.h>
#include <math.h>

#include <cuest.h>

#include <helper_status.h>
#include <helper_workspaces.h>
#include <helper_xyz_parser.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>
#include <helper_pcm.h>

/*
 * This sample shows how to use cuEST to compute PCM nuclear gradients
 * and PCM radii gradients. The polarizable continuum model (PCM)
 * accounts for solvation effects by modeling the solvent as a
 * polarizable dielectric continuum.
 *
 * Two gradient variants are demonstrated:
 *
 *   cuestPCMDerivativeCompute       - Nuclear (geometric) gradient:
 *                                     d(E_PCM)/d(R_A), size natoms×3.
 *
 *   cuestPCMRadiiDerivativeCompute  - Radii gradient:
 *                                     d(E_PCM)/d(r_A), size natoms.
 *                                     Required by radius-rescaling
 *                                     solvation models such as DRACO.
 *
 * The converged surface charges produced by the nuclear gradient call
 * are passed as the initial guess to the radii gradient call, avoiding
 * the cost of re-converging the PCG solver from scratch.
 */

/*
 * Fills an N×N device array with a random symmetric matrix using the
 * Box-Muller transform with a fixed seed so results are reproducible.
 */
void fill_symmetric_matrix(double *A, uint64_t N)
{
    double *Atmp = (double*) malloc(N * N * sizeof(double));
    if (!Atmp) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    srand(0);
    for (uint64_t i = 0; i < N; i++) {
        for (uint64_t j = 0; j <= i; j++) {
            double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
            Atmp[i*N+j] = Atmp[j*N+i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        }
    }
    cudaError_t err = cudaMemcpy(A, Atmp, N * N * sizeof(double), cudaMemcpyHostToDevice);
    free(Atmp);
    if (err != cudaSuccess) {
        fprintf(stderr, "Host to device copy failed\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <xyz_file_path> <gbs_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* xyzFilePath = argv[1];
    const char* gbsFilePath = argv[2];

    int is_pure = 1;

    parsedXYZFile_t* xyzData = parseXYZFile(xyzFilePath, 1.0 / 0.52917720859);
    if (!xyzData) {
        fprintf(stderr, "Error: failed to parse file '%s'\n", xyzFilePath);
        exit(EXIT_FAILURE);
    }

    double epsilon = 80.0;

    /* Derive per-atom PCM cavity parameters from the parsed geometry.
     * Bondi radii (scaled 1.2×) come from helper_pcm.h; zetas from the
     * York-Karplus table therein; angular grid sizes follow the convention
     * of 110 points for hydrogen and 194 for all heavier atoms. */
    uint64_t numAtomsPCM = xyzData->numAtoms;
    uint64_t *numAngularPointsPerAtom = (uint64_t*) malloc(numAtomsPCM * sizeof(uint64_t));
    double *zetas = (double*) malloc(numAtomsPCM * sizeof(double));
    double *atomicRadii = (double*) malloc(numAtomsPCM * sizeof(double));
    double *effectiveNuclearCharges = (double*) malloc(numAtomsPCM * sizeof(double));
    if (!numAngularPointsPerAtom || !zetas || !atomicRadii || !effectiveNuclearCharges) {
        fprintf(stderr, "Failed to allocate PCM parameter arrays\n");
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < numAtomsPCM; i++) {
        uint64_t nang = strcmp(xyzData->symbols[i], "H") == 0 ? 110 : 194;
        numAngularPointsPerAtom[i] = nang;
        zetas[i]                   = pcm_angular_points_to_zeta(nang);
        atomicRadii[i]             = symbol_to_scaled_bondi_radius_bohr(xyzData->symbols[i]);
        effectiveNuclearCharges[i] = -xyzData->chargesCPU[i];
    }

    /**********************/
    /* cuEST Handle Setup */
    /**********************/

    cuestHandle_t handle;
    cuestHandleParameters_t handle_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_HANDLE_PARAMETERS,
        &handle_parameters));
    checkCuestErrors(cuestCreate(
        handle_parameters,
        &handle));
    checkCuestErrors(cuestParametersDestroy(
        CUEST_HANDLE_PARAMETERS,
        handle_parameters));

    /************************/
    /* cuEST AO Shell Setup */
    /************************/

    AtomShellData_t* shell_data = formAOShells(handle, xyzData, gbsFilePath, is_pure);

    uint64_t numAtoms = shell_data->numAtoms;
    uint64_t* numShellsPerAtom = shell_data->numShellsPerAtom;
    cuestAOShell_t* shells = shell_data->shells;

    /************************/
    /* cuEST AO Basis Setup */
    /************************/

    cuestAOBasis_t basis;

    cuestAOBasisParameters_t basis_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOBASIS_PARAMETERS,
        &basis_parameters));

    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor  = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle,
        numAtoms,
        numShellsPerAtom,
        (const cuestAOShell_t*) shells,
        basis_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &basis));

    cuestWorkspace_t* persistentBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryBasisWorkspace  = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestAOBasisCreate(
        handle,
        numAtoms,
        numShellsPerAtom,
        (const cuestAOShell_t*) shells,
        basis_parameters,
        persistentBasisWorkspace,
        temporaryBasisWorkspace,
        &basis));

    freeWorkspace(temporaryBasisWorkspace);

    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS,
        basis_parameters));

    freeAOShellData(shell_data);

    /****************************/
    /* cuEST AO Pair List Setup */
    /****************************/

    cuestAOPairList_t pair_list;

    cuestAOPairListParameters_t pair_list_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOPAIRLIST_PARAMETERS,
        &pair_list_parameters));

    checkCuestErrors(cuestAOPairListCreateWorkspaceQuery(
        handle,
        basis,
        numAtoms,
        xyzData->xyzCPU,
        1.0e-14,
        pair_list_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &pair_list));

    cuestWorkspace_t* persistentPairListWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryPairListWorkspace  = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestAOPairListCreate(
        handle,
        basis,
        numAtoms,
        xyzData->xyzCPU,
        1.0e-14,
        pair_list_parameters,
        persistentPairListWorkspace,
        temporaryPairListWorkspace,
        &pair_list));

    freeWorkspace(temporaryPairListWorkspace);

    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOPAIRLIST_PARAMETERS,
        pair_list_parameters));

    /********************************/
    /* cuEST OE Integral Plan Setup */
    /********************************/

    cuestOEIntPlan_t oeint_plan;

    cuestOEIntPlanParameters_t oeint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_OEINTPLAN_PARAMETERS,
        &oeint_plan_parameters));

    checkCuestErrors(cuestOEIntPlanCreateWorkspaceQuery(
        handle,
        basis,
        pair_list,
        oeint_plan_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &oeint_plan));

    cuestWorkspace_t* persistentOEIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryOEIntPlanWorkspace  = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestOEIntPlanCreate(
        handle,
        basis,
        pair_list,
        oeint_plan_parameters,
        persistentOEIntPlanWorkspace,
        temporaryOEIntPlanWorkspace,
        &oeint_plan));

    freeWorkspace(temporaryOEIntPlanWorkspace);

    checkCuestErrors(cuestParametersDestroy(
        CUEST_OEINTPLAN_PARAMETERS,
        oeint_plan_parameters));

    /*********************************/
    /* cuEST PCM Integral Plan Setup */
    /*********************************/

    cuestPCMIntPlan_t pcmint_plan;

    cuestPCMIntPlanParameters_t pcmint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_PCMINTPLAN_PARAMETERS,
        &pcmint_plan_parameters));

    checkCuestErrors(cuestPCMIntPlanCreateWorkspaceQuery(
        handle,
        oeint_plan,
        pcmint_plan_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &numAngularPointsPerAtom[0],
        epsilon,
        &zetas[0],
        &atomicRadii[0],
        &effectiveNuclearCharges[0],
        &pcmint_plan));

    cuestWorkspace_t* persistentPCMIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);
    cuestWorkspace_t* temporaryPCMIntPlanWorkspace  = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestPCMIntPlanCreate(
        handle,
        oeint_plan,
        pcmint_plan_parameters,
        persistentPCMIntPlanWorkspace,
        temporaryPCMIntPlanWorkspace,
        &numAngularPointsPerAtom[0],
        epsilon,
        &zetas[0],
        &atomicRadii[0],
        &effectiveNuclearCharges[0],
        &pcmint_plan));

    freeWorkspace(temporaryPCMIntPlanWorkspace);

    checkCuestErrors(cuestParametersDestroy(
        CUEST_PCMINTPLAN_PARAMETERS,
        pcmint_plan_parameters));

    /*************************/
    /* Compute PCM Gradients */
    /*************************/

    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(
        handle,
        CUEST_AOBASIS,
        basis,
        CUEST_AOBASIS_NUM_AO,
        &nao,
        sizeof(uint64_t)));

    /* npoint is the sum of numAngularPointsPerAtom. */
    uint64_t npoint = 0;
    checkCuestErrors(cuestQuery(
        handle,
        CUEST_PCMINTPLAN,
        pcmint_plan,
        CUEST_PCMINTPLAN_NUM_POINT,
        &npoint,
        sizeof(uint64_t)));

    /*
     * d_inQ and d_outQ_nuc are used for the nuclear gradient PCG solve.
     * d_outQ_nuc (converged charges) is then passed as the warm-start
     * initial guess d_inQ for the radii gradient PCG solve, saving the
     * cost of re-converging from scratch.
     */
    double *d_D, *d_inQ, *d_outQ_nuc, *d_outQ_radii;
    double *d_gradient, *d_radii_gradient;
    if (cudaMalloc((void**) &d_D,             nao    * nao    * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void**) &d_inQ,           npoint          * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void**) &d_outQ_nuc,      npoint          * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void**) &d_outQ_radii,    npoint          * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void**) &d_gradient,      numAtoms * 3    * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void**) &d_radii_gradient,numAtoms        * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffers\n");
        exit(EXIT_FAILURE);
    }

    /* Populate density matrix with a random symmetric matrix. A real calculation would supply 
     * the total (alpha + beta) SCF density matrix here. */
    fill_symmetric_matrix(d_D, nao);

    /* Zero the initial charge guess; the PCG solver will find the converged charges. */
    if (cudaMemset(d_inQ, 0, npoint * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed\n");
        exit(EXIT_FAILURE);
    }

    cuestPCMResults_t pcm_results;
    checkCuestErrors(cuestResultsCreate(
        CUEST_PCM_RESULTS,
        &pcm_results));

    /************************************/
    /* PCM Nuclear (geometric) gradient */
    /************************************/

    cuestPCMDerivativeComputeParameters_t pcm_deriv_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS,
        &pcm_deriv_parameters));

    checkCuestErrors(cuestPCMDerivativeComputeWorkspaceQuery(
        handle,
        pcmint_plan,
        pcm_deriv_parameters,
        temporaryWorkspaceDescriptor,
        d_D,
        d_inQ,
        d_outQ_nuc,
        pcm_results,
        d_gradient));

    cuestWorkspace_t* temporaryPCMDerivWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestPCMDerivativeCompute(
        handle,
        pcmint_plan,
        pcm_deriv_parameters,
        temporaryPCMDerivWorkspace,
        d_D,
        d_inQ,
        d_outQ_nuc,
        pcm_results,
        d_gradient));

    freeWorkspace(temporaryPCMDerivWorkspace);

    checkCuestErrors(cuestParametersDestroy(
        CUEST_PCMDERIVATIVECOMPUTE_PARAMETERS,
        pcm_deriv_parameters));

    /**********************/
    /* PCM Radii gradient */
    /**********************/

    cuestPCMRadiiDerivativeComputeParameters_t pcm_radii_deriv_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_PCMRADIIDERIVATIVECOMPUTE_PARAMETERS,
        &pcm_radii_deriv_parameters));

    /*
     * Pass d_outQ_nuc as the initial charge guess for the radii gradient
     * PCG solve. Starting from converged nuclear-gradient charges is an
     * effective warm start because the two problems share the same cavity.
     */
    checkCuestErrors(cuestPCMRadiiDerivativeComputeWorkspaceQuery(
        handle,
        pcmint_plan,
        pcm_radii_deriv_parameters,
        temporaryWorkspaceDescriptor,
        d_D,
        d_outQ_nuc,
        d_outQ_radii,
        pcm_results,
        d_radii_gradient));

    cuestWorkspace_t* temporaryPCMRadiiDerivWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestPCMRadiiDerivativeCompute(
        handle,
        pcmint_plan,
        pcm_radii_deriv_parameters,
        temporaryPCMRadiiDerivWorkspace,
        d_D,
        d_outQ_nuc,
        d_outQ_radii,
        pcm_results,
        d_radii_gradient));

    freeWorkspace(temporaryPCMRadiiDerivWorkspace);

    checkCuestErrors(cuestParametersDestroy(
        CUEST_PCMRADIIDERIVATIVECOMPUTE_PARAMETERS,
        pcm_radii_deriv_parameters));

    cudaFree(d_D);
    cudaFree(d_inQ);
    cudaFree(d_outQ_nuc);
    cudaFree(d_outQ_radii);
    cudaFree(d_gradient);
    cudaFree(d_radii_gradient);

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    checkCuestErrors(cuestResultsDestroy(CUEST_PCM_RESULTS, pcm_results));

    checkCuestErrors(cuestPCMIntPlanDestroy(pcmint_plan));
    freeWorkspace(persistentPCMIntPlanWorkspace);

    checkCuestErrors(cuestOEIntPlanDestroy(oeint_plan));
    freeWorkspace(persistentOEIntPlanWorkspace);

    checkCuestErrors(cuestAOPairListDestroy(pair_list));
    freeWorkspace(persistentPairListWorkspace);

    checkCuestErrors(cuestAOBasisDestroy(basis));
    freeWorkspace(persistentBasisWorkspace);

    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestDestroy(handle));

    free(numAngularPointsPerAtom);
    free(zetas);
    free(atomicRadii);
    free(effectiveNuclearCharges);

    freeParsedXYZFile(xyzData);

    return 0;
}
