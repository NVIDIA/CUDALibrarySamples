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
#include <cublas_v2.h>

/*
 * This sample shows how to use the density-fitting (DF) MO integral
 * derivative routine to compute the nuclear gradient of an energy written
 * as a contraction of a four-index two-particle density matrix (TPDM)
 * with MO basis integrals,
 *
 *     E = sum_{pqrs} (pq|rs) Gamma_pq^rs.
 *
 * The derivative routine does not accept the four-index TPDM directly.
 * Instead, it takes the three-index TPDM
 *
 *     Gamma_pq^A = sum_{rs} B_rs^A Gamma_pq^rs,
 *
 * where B are the DF integrals without the operator weighting, obtained
 * from cuestDFMOIntegralsCompute with the operator weighting set to NONE.
 * Two cases are shown below:
 *   - Case 1: the simple (pq|rs) Gamma_pq^rs term, which is bra-ket
 *     symmetric, Gamma_pq^rs = Gamma_rs^pq.
 *   - Case 2: the (ij|ab) Gamma_ij^ab term, with i,j occupied and a,b
 *     virtual, which is not bra-ket symmetric.  The three-index TPDMs
 *     Gamma_ij^A and Gamma_ab^A, each scaled by 1/2, are passed as a batch
 *     of two so that bra-ket symmetry is preserved.
 *
 * Synthetic data is generated to populate these arrays.  The number of
 * occupied orbitals is calculated from the sum of atomic charges.
 */

/*
 * This is a helper function to generate synthetic data to populate
 * a matrix.
 */
void fill_matrix(double *A, uint64_t M, uint64_t N)
{
    double *Atmp = (double*) malloc(M * N * sizeof(double));
    if (!Atmp) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    /* Force this function to return the same values on mulitple calls */
    srand(0);
    for (uint64_t i=0, ij=0; i<M; i++) {
        for (uint64_t j=0; j<N; j++, ij++) {
            /* Box-Muller transform with mu=0 and stddev=1.0 */
            double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
            Atmp[ij] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        }
    }
    cudaError_t err = cudaMemcpy(A, Atmp,  M * N * sizeof(double), cudaMemcpyHostToDevice);
    free(Atmp);
    if (err != cudaSuccess) {
        fprintf(stderr, "Host to device copy failed\n");
        exit(EXIT_FAILURE);
    }
}

/*
 * This is a helper function to generate synthetic data to populate
 * a symmetric matrix.
 */
void fill_symmetric_matrix(double *A, uint64_t N)
{
    double *Atmp = (double*) malloc(N * N * sizeof(double));
    if (!Atmp) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    /* Force this function to return the same values on mulitple calls */
    srand(0);
    for (uint64_t i=0; i<N; i++) {
        for (uint64_t j=0; j<=i; j++) {
            /* Box-Muller transform with mu=0 and stddev=1.0 */
            double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
            double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
            Atmp[i*N+j] = Atmp[j*N+i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        }
    }
    cudaError_t err = cudaMemcpy(A, Atmp,  N * N * sizeof(double), cudaMemcpyHostToDevice);
    free(Atmp);
    if (err != cudaSuccess) {
        fprintf(stderr, "Host to device copy failed\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    /* Check that an xyz file has been provided */
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <xyz_file_path> <primary_gbs_file_path> <auxiliary_gbs_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* xyzFilePath = argv[1];
    const char* primaryGBSFilePath = argv[2];
    const char* auxiliaryGBSFilePath = argv[3];

    /* Parse the XYZ file */
    parsedXYZFile_t* xyzData = parseXYZFile(xyzFilePath, 1.0 / 0.52917720859);
    if (!xyzData) {
        fprintf(stderr, "Error: failed to parse file '%s'\n", xyzFilePath);
        exit(EXIT_FAILURE);
    }

    /**********************/
    /* cuEST Handle Setup */
    /**********************/

    /* Create the cuEST handle. */
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

    /****************************************/
    /* cuEST AO Shell Setup (primary basis) */
    /****************************************/

    /* Use the AO shell helper to build the array of AO shells */
    AtomShellData_t* primaryShellData = formAOShells(handle, xyzData, primaryGBSFilePath, 1);

    /* Unpack the AtomShellData_t struct */
    uint64_t numAtoms = primaryShellData->numAtoms;
    uint64_t numPrimaryShellsTotal = primaryShellData->numShellsTotal;
    uint64_t* numPrimaryShellsPerAtom = primaryShellData->numShellsPerAtom;
    cuestAOShell_t* primaryShells = primaryShellData->shells;

    /****************************************/
    /* cuEST AO Basis Setup (primary basis) */
    /****************************************/

    /* Declare the AO basis handle. */
    cuestAOBasis_t primaryBasis;

    /* Declare the AO basis parameter handle. */
    cuestAOBasisParameters_t primaryBasisParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOBASIS_PARAMETERS,
        &primaryBasisParameters));

    /* Allocate space for workspace descriptors. Will be used to determine how large the workspace needs to be. */
    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    /* Make the workspace query to populate the workspace descriptors. */
    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle,
        numAtoms,
        numPrimaryShellsPerAtom,
        (const cuestAOShell_t*) primaryShells,
        primaryBasisParameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &primaryBasis));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentPrimaryBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryPrimaryBasisWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the AO basis handle. */
    checkCuestErrors(cuestAOBasisCreate(
        handle,
        numAtoms,
        numPrimaryShellsPerAtom,
        (const cuestAOShell_t*) primaryShells,
        primaryBasisParameters,
        persistentPrimaryBasisWorkspace,
        temporaryPrimaryBasisWorkspace,
        &primaryBasis));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryPrimaryBasisWorkspace);

    /* The AO basis parameters are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS,
        primaryBasisParameters));

    /* Free all the shell data */
    freeAOShellData(primaryShellData);

    /******************************************/
    /* cuEST AO Shell Setup (auxiliary basis) */
    /******************************************/

    /* Use the AO shell helper to build the array of AO shells */
    AtomShellData_t* auxiliaryShellData = formAOShells(handle, xyzData, auxiliaryGBSFilePath, 1);

    /* Unpack the AtomShellData_t struct */
    uint64_t numAuxiliaryShellsTotal = auxiliaryShellData->numShellsTotal;
    uint64_t* numAuxiliaryShellsPerAtom = auxiliaryShellData->numShellsPerAtom;
    cuestAOShell_t* auxiliaryShells = auxiliaryShellData->shells;

    /******************************************/
    /* cuEST AO Basis Setup (auxiliary basis) */
    /******************************************/

    /* Declare the AO basis handle. */
    cuestAOBasis_t auxiliaryBasis;

    /* Declare the AO basis parameter handle. */
    cuestAOBasisParameters_t auxiliaryBasisParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOBASIS_PARAMETERS,
        &auxiliaryBasisParameters));

    /* Make the workspace query to populate the workspace descriptors. */
    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle,
        numAtoms,
        numAuxiliaryShellsPerAtom,
        (const cuestAOShell_t*) auxiliaryShells,
        auxiliaryBasisParameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &auxiliaryBasis));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentAuxiliaryBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryAuxiliaryBasisWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the AO basis handle. */
    checkCuestErrors(cuestAOBasisCreate(
        handle,
        numAtoms,
        numAuxiliaryShellsPerAtom,
        (const cuestAOShell_t*) auxiliaryShells,
        auxiliaryBasisParameters,
        persistentAuxiliaryBasisWorkspace,
        temporaryAuxiliaryBasisWorkspace,
        &auxiliaryBasis));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryAuxiliaryBasisWorkspace);

    /* The AO basis parameters are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS,
        auxiliaryBasisParameters));

    /* Free all the shell data */
    freeAOShellData(auxiliaryShellData);

    /****************************/
    /* cuEST AO Pair List Setup */
    /****************************/

    /* Declare the AO pair list handle. */
    cuestAOPairList_t pair_list;

    /* Declare the AO pair list parameter handle. */
    cuestAOPairListParameters_t pair_list_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOPAIRLIST_PARAMETERS,
        &pair_list_parameters));

    /* Determine how large the workspace must be to form and store the pair list. */
    checkCuestErrors(cuestAOPairListCreateWorkspaceQuery(
        handle,
        primaryBasis,
        numAtoms,
        xyzData->xyzCPU,
        1.0e-14,
        pair_list_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &pair_list));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentAOPairListWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryAOPairListWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the pair list. */
    checkCuestErrors(cuestAOPairListCreate(
        handle,
        primaryBasis,
        numAtoms,
        xyzData->xyzCPU,
        1.0e-14,
        pair_list_parameters,
        persistentAOPairListWorkspace,
        temporaryAOPairListWorkspace,
        &pair_list));

    /* The AO pair list parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOPAIRLIST_PARAMETERS,
        pair_list_parameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryAOPairListWorkspace);

    /********************************/
    /* cuEST DF Integral Plan Setup */
    /********************************/

    /* Declare the DF integral plan handle. */
    cuestDFIntPlan_t dfint_plan;

    /* Declare the DF integral plan parameter handle. */
    cuestDFIntPlanParameters_t dfint_plan_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DFINTPLAN_PARAMETERS,
        &dfint_plan_parameters));

    /*
     * The default behavior of the DFIntPlan is to cache the 3-index integrals so
     * they can be read from disk every transform. A more compute-intensive but
     * more memory-efficient approach is to recompute the 3-index integrals on the fly.
     * This can be achieved by configuring the dfint_plan_parameters with the
     * CUEST_DFINTPLAN_PARAMETERS_THREE_INDEX_INTEGRAL_DIRECT attribute set to 1.
     */

    /*
     * Integral plans that use cached integrals, have a range-separated component,
     * and are intended for use with the MO integral derivative code must be created
     * with the CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_NONE attribute.
     * While it's not strictly necessary to do this for integral direct plans or those
     * that involve the plain Coulomb operator, there's no performance penalty to doing
     * so in these cases; we therefore strongly recommend that
     * CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_NONE be set unconditionally
     * when making a plan that is to be used for MO derivatives.
     */
    cuestDFIntPlanParametersCacheOperatorWeighting_t cacheOperatorWeighting = CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_NONE;
    checkCuestErrors(cuestParametersConfigure(
        CUEST_DFINTPLAN_PARAMETERS,
        dfint_plan_parameters,
        CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING,
        &cacheOperatorWeighting,
        sizeof(cacheOperatorWeighting)));

    /* Determine how large the workspace must be to form and store the DF integral plan. */
    checkCuestErrors(cuestDFIntPlanCreateWorkspaceQuery(
        handle,
        primaryBasis,
        auxiliaryBasis,
        pair_list,
        dfint_plan_parameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &dfint_plan));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentDFIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryDFIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the DF integral plan. */
    checkCuestErrors(cuestDFIntPlanCreate(
        handle,
        primaryBasis,
        auxiliaryBasis,
        pair_list,
        dfint_plan_parameters,
        persistentDFIntPlanWorkspace,
        temporaryDFIntPlanWorkspace,
        &dfint_plan));

    /* The DF integral plan parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DFINTPLAN_PARAMETERS,
        dfint_plan_parameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryDFIntPlanWorkspace);


    /*****************************/
    /* Set Up Orbitals and TPDMs */
    /*****************************/

    /* Query the AO basis for the number of basis functions */
    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(
        handle,
        CUEST_AOBASIS,
        primaryBasis,
        CUEST_AOBASIS_NUM_AO,
        &nao,
        sizeof(uint64_t)));

    /* Query the auxiliary basis for the number of basis functions */
    uint64_t naux = 0;
    checkCuestErrors(cuestQuery(
        handle,
        CUEST_AOBASIS,
        auxiliaryBasis,
        CUEST_AOBASIS_NUM_AO,
        &naux,
        sizeof(uint64_t)));

    /* Get nocc for a neutral molecule */
    uint64_t nocc = 0;
    for (size_t i=0; i<xyzData->numAtoms; i++) {
        nocc += (uint64_t) (-1.0 * xyzData->chargesCPU[i]);
    }
    nocc = nocc / 2;

    /*
     * Define nvir as the remaining orbitals for this example.
     * In practice, this comes from a separate SCF/MO calculation.
     */
    uint64_t nvir = (nao > nocc) ? (nao - nocc) : 0;
    if (nvir == 0) {
        fprintf(stderr, "No virtual orbitals available in this simple example.\n");
        exit(EXIT_FAILURE);
    }

    /* Number of MOs, and the sizes of the composite pq, ij and ab indices */
    uint64_t nmo = nocc + nvir;
    uint64_t npq = nmo * nmo;
    uint64_t noo = nocc * nocc;
    uint64_t nvv = nvir * nvir;

    /*
     * Orbital counts for the coefficient matrices.  Case 1 uses all MOs on
     * both sides of a single pair of coefficient matrices, and Case 2 uses a
     * batch of two pairs, [occ, occ] and [vir, vir].
     */
    uint64_t numMO[1] = {nmo};
    uint64_t numOccVir[2] = {nocc, nvir};

    /*
     * Allocate space for the MO coefficients, with shape [nmo, nao].  The
     * occupied orbitals are the first nocc rows and the virtual orbitals are
     * the last nvir rows, so the same array is [C_occ, C_vir] when
     * concatenated for the batch of two.
     */
    double* d_C;
    if (cudaMalloc((void**) &d_C, nmo * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the MO coefficients\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Fill with random numbers. In real use cases, these are molecular
     * orbital coefficients.
     */
    fill_matrix(d_C, nmo, nao);

    /* Create the cuBLAS handle used to form the three-index TPDMs. */
    cublasHandle_t cublas_handle;
    if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create cuBLAS handle.\n");
        exit(EXIT_FAILURE);
    }
    double zero = 0.0;
    double one = 1.0;
    double half = 0.5;

    /* Declare the DF MO integral derivative parameter handle. */
    cuestDFMOIntegralsDerivativeComputeParameters_t dfmo_grad_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DFMOINTEGRALSDERIVATIVECOMPUTE_PARAMETERS,
        &dfmo_grad_parameters));

    /*
     * The DF MO integral transformations can benefit from a large temporary
     * workspace. Here, a workspace descriptor is used to set an upper limit
     * for the size of an internal scratch buffer. As a general rule, 2 GB
     * is a reasonable default value for this space, but larger allocations
     * may result in more efficient transformations.
     */
    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 2000000000;

    /******************************************/
    /* Case 1: sum_{pqrs} (pq|rs) Gamma_pq^rs */
    /******************************************/

    /* Allocate space for B_pq^A, with shape [naux, nmo * nmo] */
    double* d_Tensors;
    if (cudaMalloc((void**) &d_Tensors, naux * npq * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the DF MO tensors\n");
        exit(EXIT_FAILURE);
    }

    /*
     * The operator weighting of the integral plan is applied by the
     * derivative routine, so the DF MO integrals used to form the
     * three-index TPDMs must not include it.  By default the weighting
     * is included, so it is turned off here.  The DF integral plan was
     * created with CUEST_DFINTPLAN_PARAMETERS_CACHE_OPERATOR_WEIGHTING_NONE
     * so that it can serve this request.
     */
    cuestDFMOIntegralsComputeParameters_t dfmo_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        &dfmo_parameters));

    cuestDFMOIntegralsComputeParametersOperatorWeighting_t operatorWeighting = CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS_OPERATOR_WEIGHTING_NONE;
    checkCuestErrors(cuestParametersConfigure(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        dfmo_parameters,
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS_OPERATOR_WEIGHTING,
        &operatorWeighting,
        sizeof(operatorWeighting)));

    /* Compute DF MO integrals without the operator weighting */
    checkCuestErrors(cuestDFMOIntegralsComputeWorkspaceQuery(
        handle,
        dfint_plan,
        dfmo_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        1,
        numMO,
        numMO,
        d_C,
        d_C,
        d_Tensors));

    cuestWorkspace_t* temporaryDFMOWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestDFMOIntegralsCompute(
        handle,
        dfint_plan,
        dfmo_parameters,
        variableBufferSize,
        temporaryDFMOWorkspace,
        1,
        numMO,
        numMO,
        d_C,
        d_C,
        d_Tensors));

    freeWorkspace(temporaryDFMOWorkspace);

    /*
     * Allocate space for the four-index TPDM, with shape [pq, rs].
     */
    double* d_Gamma;
    if (cudaMalloc((void**) &d_Gamma, npq * npq * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the four-index TPDM\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Fill with random numbers. In real use cases, this is a bra-ket symmetric
     * two-particle density matrix.
     */
    fill_symmetric_matrix(d_Gamma, npq);

    /*
     * Form the three-index TPDM, Gamma_pq^A, by contracting the DF integral factors
     * with the four-index TPDM.  The arrays are row-major, so cuBLAS
     * sees them transposed.  The cuEST handle executes on its own CUDA
     * stream, so the device is synchronized before and after the cuBLAS call.
     * See the user_owned_resources example for details about how to initialize
     * cuBLAS and cuEST to use the same stream and avoid this step.
     */
    double* d_Gamma3;
    if (cudaMalloc((void**) &d_Gamma3, naux * npq * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the three-index TPDM\n");
        exit(EXIT_FAILURE);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        exit(EXIT_FAILURE);
    }
    if (cublasDgemm(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        npq,
        naux,
        npq,
        &one,
        d_Gamma,
        npq,
        d_Tensors,
        npq,
        &zero,
        d_Gamma3,
        npq) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasDgemm failed\n");
        exit(EXIT_FAILURE);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate space for the gradient, with shape [natom, 3] */
    double* d_grad1;
    if (cudaMalloc((void**) &d_grad1, xyzData->numAtoms * 3 * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the gradient\n");
        exit(EXIT_FAILURE);
    }

    /* Compute the gradient of the energy with respect to the nuclear coordinates */
    checkCuestErrors(cuestDFMOIntegralsDerivativeComputeWorkspaceQuery(
        handle,
        dfint_plan,
        dfmo_grad_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        1,
        numMO,
        numMO,
        d_C,
        d_C,
        d_Gamma3,
        d_grad1));

    cuestWorkspace_t* temporaryDFMOgradWorkspace1 = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestDFMOIntegralsDerivativeCompute(
        handle,
        dfint_plan,
        dfmo_grad_parameters,
        variableBufferSize,
        temporaryDFMOgradWorkspace1,
        1,
        numMO,
        numMO,
        d_C,
        d_C,
        d_Gamma3,
        d_grad1));

    freeWorkspace(temporaryDFMOgradWorkspace1);

    /*******************************************/
    /* Case 2: sum_{ijab} (ij|ab) Gamma_ij^ab  */
    /*******************************************/

    /*
     * This term is not bra-ket symmetric, so the three-index TPDMs are
     * built from two terms, each with a scale factor of 1/2,
     *
     *     Gamma_ij^A = 1/2 sum_ab B_ab^A Gamma_ij^ab
     *     Gamma_ab^A = 1/2 sum_ij B_ij^A Gamma_ij^ab
     *
     * and passed as a batch of two.
     */

    /*
     * Allocate space for B_ij^A, with shape [naux, noo], followed by
     * B_ab^A, with shape [naux, nvv]
     */
    double* d_TensorsOV;
    if (cudaMalloc((void**) &d_TensorsOV, naux * (noo + nvv) * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the DF MO tensors\n");
        exit(EXIT_FAILURE);
    }

    /* Compute DF MO integrals (B_ij^A, B_ab^A) without the operator weighting */
    checkCuestErrors(cuestDFMOIntegralsComputeWorkspaceQuery(
        handle,
        dfint_plan,
        dfmo_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        2,
        numOccVir,
        numOccVir,
        d_C,
        d_C,
        d_TensorsOV));

    cuestWorkspace_t* temporaryDFMOWorkspaceOV = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestDFMOIntegralsCompute(
        handle,
        dfint_plan,
        dfmo_parameters,
        variableBufferSize,
        temporaryDFMOWorkspaceOV,
        2,
        numOccVir,
        numOccVir,
        d_C,
        d_C,
        d_TensorsOV));

    freeWorkspace(temporaryDFMOWorkspaceOV);

    /*
     * Allocate space for the four-index TPDM, with shape [ij, ab], and fill
     * with random numbers.  This is the only block of the TPDM that is needed.
     */
    double* d_GammaOV;
    if (cudaMalloc((void**) &d_GammaOV, noo * nvv * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the four-index TPDM\n");
        exit(EXIT_FAILURE);
    }

    fill_matrix(d_GammaOV, noo, nvv);

    /*
     * Form the three-index TPDMs, Gamma_ij^A with shape [naux, noo],
     * followed by Gamma_ab^A with shape [naux, nvv], including the scale
     * factor of 1/2.
     */
    double* d_Gamma3OV;
    if (cudaMalloc((void**) &d_Gamma3OV, naux * (noo + nvv) * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the three-index TPDMs\n");
        exit(EXIT_FAILURE);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        exit(EXIT_FAILURE);
    }
    if (cublasDgemm(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        noo,
        naux,
        nvv,
        &half,
        d_GammaOV,
        nvv,
        d_TensorsOV + naux * noo,
        nvv,
        &zero,
        d_Gamma3OV,
        noo) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasDgemm failed\n");
        exit(EXIT_FAILURE);
    }

    if (cublasDgemm(
        cublas_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        nvv,
        naux,
        noo,
        &half,
        d_GammaOV,
        nvv,
        d_TensorsOV,
        noo,
        &zero,
        d_Gamma3OV + naux * noo,
        nvv) != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "cublasDgemm failed\n");
        exit(EXIT_FAILURE);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate space for the gradient, with shape [natom, 3] */
    double* d_grad2;
    if (cudaMalloc((void**) &d_grad2, xyzData->numAtoms * 3 * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer for the gradient\n");
        exit(EXIT_FAILURE);
    }

    /*
     * Compute the gradient.  The orbitals [C_occ, C_vir] are used on both
     * the left and the right for the batch of two.
     */
    checkCuestErrors(cuestDFMOIntegralsDerivativeComputeWorkspaceQuery(
        handle,
        dfint_plan,
        dfmo_grad_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        2,
        numOccVir,
        numOccVir,
        d_C,
        d_C,
        d_Gamma3OV,
        d_grad2));

    cuestWorkspace_t* temporaryDFMOgradWorkspace2 = allocateWorkspace(temporaryWorkspaceDescriptor);

    checkCuestErrors(cuestDFMOIntegralsDerivativeCompute(
        handle,
        dfint_plan,
        dfmo_grad_parameters,
        variableBufferSize,
        temporaryDFMOgradWorkspace2,
        2,
        numOccVir,
        numOccVir,
        d_C,
        d_C,
        d_Gamma3OV,
        d_grad2));

    freeWorkspace(temporaryDFMOgradWorkspace2);

    free(variableBufferSize);

    /* The DF MO integral parameter handles are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DFMOINTEGRALSCOMPUTE_PARAMETERS,
        dfmo_parameters));
    checkCuestErrors(cuestParametersDestroy(
        CUEST_DFMOINTEGRALSDERIVATIVECOMPUTE_PARAMETERS,
        dfmo_grad_parameters));

    /* Destroy the cuBLAS handle. */
    if (cublasDestroy(cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to destroy cuBLAS handle.\n");
        exit(EXIT_FAILURE);
    }

    /* Free MO coefficient, TPDM, DF MO tensor and gradient buffers */
    if (cudaFree((void*) d_C) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Tensors) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Gamma) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Gamma3) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_grad1) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_TensorsOV) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_GammaOV) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_Gamma3OV) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_grad2) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /* Destroy the DF integral plan handle. */
    checkCuestErrors(cuestDFIntPlanDestroy(dfint_plan));
    freeWorkspace(persistentDFIntPlanWorkspace);

    /* Destroy the AO pair list handle. */
    checkCuestErrors(cuestAOPairListDestroy(pair_list));
    freeWorkspace(persistentAOPairListWorkspace);

    /* Destroy the AO basis handle (auxiliary basis). */
    checkCuestErrors(cuestAOBasisDestroy(auxiliaryBasis));
    freeWorkspace(persistentAuxiliaryBasisWorkspace);

    /* Destroy the AO basis handle (primary basis). */
    checkCuestErrors(cuestAOBasisDestroy(primaryBasis));
    freeWorkspace(persistentPrimaryBasisWorkspace);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    /* Free the XYZ data */
    freeParsedXYZFile(xyzData);

    return 0;
}

