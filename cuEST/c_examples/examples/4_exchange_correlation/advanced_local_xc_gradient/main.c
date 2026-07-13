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
#include <stdbool.h>

#include <cuest.h>

#include <helper_status.h>
#include <helper_workspaces.h>
#include <helper_xyz_parser.h>
#include <helper_grid.h>
#include <helper_gbs_parser.h>
#include <helper_ao_shells.h>


/*
 * This example shows how to evaluate the gradient of the exchange correlation
 * energy with respect to nuclear coordinates using the advanced interface.
 * As with the potential example, the (rate-limiting) density collocation and
 * its adjoint are performed using cuEST on the GPU, while the density and
 * its derivatives are shuttled to the CPU for the (relatively cheap)
 * evaluation of the functional and its partial derivatives.  The resulting
 * per-point XC potential array is then handed back to cuEST, which assembles
 * the gradient of the XC energy with respect to nuclear coordinates.  The
 * gradient has two parts: a "basis center" contribution from the motion of
 * the AO centers (computed by cuestXCDerivativeCompute) and a "grid center"
 * contribution from the motion of the grid points.  The grid center
 * contribution is mapped back to atomic forces by cuestXCGridDerivativeCompute.
 *
 * This depends on an AO basis set, a molecular grid (here, a 75, 302 grid),
 * and coefficient/density matrices. In this sample, synthetic data is used
 * for those matrices. In real applications of cuEST, these matrices should be
 * obtained during an SCF procedure. Examples of both RKS and UKS usage are
 * provided. Simple examples of LDA, GGA, and meta-GGA functionals are
 * included, and the result can optionally be compared against the corresponding
 * libxc implementation.  Unlike libxc, the simple reference implementations
 * provided here do very minimal sanity checking on the input quantities, so
 * some auxilliary sanitization is performed on the input grid to ensure that
 * the simple reference implementations does not encounter numerical issues
 * due to unphysical density derivatives that violate known constraints.
 */

 /*
  * Standlone functional implementations are provided below.  However, if
  * libxc is available it can be used to check the results.  The libxc calls
  * below guarded by this macro are to guide users already familiar with libxc.
  */
 #define CHECK_AGAINST_LIBXC 0
 #if CHECK_AGAINST_LIBXC
 #include <xc.h>
 #include <xc_funcs.h>
 #endif
 
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
 * Simple reference implementation of Libxc's LDA_X exchange functional.
 */
 void my_xc_lda_exc_vxc(
    size_t npoints,
    const double* p_rho,
    double* p_f,
    double* p_f_rho,
    int is_spin_polarized)
{
    const double C_x = 0.75 * cbrt(3.0 / M_PI);
    const double K_x = C_x * cbrt(2.0);

    const double rho_threshold = 1.0e-15;

    if (!is_spin_polarized) {
        for (size_t i = 0; i < npoints; i++) {
            const double rho = p_rho[i];

            if (rho < rho_threshold) {
                p_f[i] = 0.0;
                p_f_rho[i] = 0.0;
                continue;
            }

            const double rho_13 = cbrt(rho);
            const double eps_x  = -C_x * rho_13;

            p_f[i] = eps_x;
            p_f_rho[i] = (4.0 / 3.0) * eps_x;
        }
    } else {
        for (size_t i = 0; i < npoints; i++) {
            const double rho_a = p_rho[2 * i + 0];
            const double rho_b = p_rho[2 * i + 1];

            const double rho_total = rho_a + rho_b;
            if (rho_total < rho_threshold) {
                p_f[i] = 0.0;
                p_f_rho[2 * i + 0] = 0.0;
                p_f_rho[2 * i + 1] = 0.0;
                continue;
            }

            const double rho_a_13 = cbrt(rho_a);
            const double rho_a_43 = rho_a * rho_a_13;
            const double rho_b_13 = cbrt(rho_b);
            const double rho_b_43 = rho_b * rho_b_13;

            p_f[i] = -K_x * (rho_a_43 + rho_b_43) / rho_total;
            p_f_rho[2 * i + 0] = -(4.0 / 3.0) * K_x * rho_a_13;
            p_f_rho[2 * i + 1] = -(4.0 / 3.0) * K_x * rho_b_13;
        }
    }
}

/*
 * Simple reference implementation of libxc's GGA_X_B86 exchange functional.
 */
static void b86_eval_one_channel(
    double rho_s,
    double gamma_s,
    double* f_per_vol,
    double* df_drho_s,
    double* df_dgamma_s)
{
    const double C_x = 0.75 * cbrt(3.0 / M_PI);
    const double K_x = C_x * cbrt(2.0);
    const double beta = 0.0036;
    const double gamma = 0.004;

    const double rho_threshold = 1.0e-15;

    if (rho_s < rho_threshold) {
        *f_per_vol = 0.0;
        *df_drho_s = 0.0;
        *df_dgamma_s = 0.0;
        return;
    }

    const double rho_13 = cbrt(rho_s);
    const double rho_43 = rho_s * rho_13;
    const double rho_83 = rho_43 * rho_43;

    const double x_sq = gamma_s / rho_83;
    const double D = 1.0 + gamma * x_sq;
    const double D2 = D * D;
    const double h = x_sq / D;

    *f_per_vol = -rho_43 * (K_x + beta * h);
    *df_drho_s = -(4.0 / 3.0) * rho_13 * (K_x + beta * h)
               + (8.0 / 3.0) * beta * rho_13 * x_sq / D2;
    *df_dgamma_s = -beta / (rho_43 * D2);
}

void my_xc_gga_exc_vxc(
    size_t npoints,
    const double* p_rho,
    const double* p_gamma,
    double* p_f,
    double* p_f_rho,
    double* p_f_gamma,
    int is_spin_polarized)
{
    const double rho_threshold = 1.0e-15;

    if (!is_spin_polarized) {
        for (size_t i = 0; i < npoints; i++) {
            const double rho = p_rho[i];

            if (rho < rho_threshold) {
                p_f[i] = 0.0;
                p_f_rho[i] = 0.0;
                p_f_gamma[i] = 0.0;
                continue;
            }

            const double rho_s = 0.5  * rho;
            const double gamma_s = 0.25 * p_gamma[i];

            double f_s, df_drho_s, df_dgamma_s;
            b86_eval_one_channel(rho_s, gamma_s, &f_s, &df_drho_s, &df_dgamma_s);

            const double f_total_per_vol = 2.0 * f_s;

            p_f[i] = f_total_per_vol / rho;
            p_f_rho[i] = df_drho_s;
            p_f_gamma[i] = 0.5 * df_dgamma_s;
        }
    } else {
        for (size_t i = 0; i < npoints; i++) {
            const double rho_a = p_rho[2 * i + 0];
            const double rho_b = p_rho[2 * i + 1];
            const double gamma_aa = p_gamma[3 * i + 0];
            /* gamma_ab is unused -- exchange has no cross-spin term. */
            const double gamma_bb = p_gamma[3 * i + 2];

            const double rho_total = rho_a + rho_b;
            if (rho_total < rho_threshold) {
                p_f[i] = 0.0;
                p_f_rho[2 * i + 0] = 0.0;
                p_f_rho[2 * i + 1] = 0.0;
                p_f_gamma[3 * i + 0] = 0.0;
                p_f_gamma[3 * i + 1] = 0.0;
                p_f_gamma[3 * i + 2] = 0.0;
                continue;
            }

            double f_a, df_drho_a, df_dgamma_a;
            double f_b, df_drho_b, df_dgamma_b;
            b86_eval_one_channel(rho_a, gamma_aa, &f_a, &df_drho_a, &df_dgamma_a);
            b86_eval_one_channel(rho_b, gamma_bb, &f_b, &df_drho_b, &df_dgamma_b);

            p_f[i] = (f_a + f_b) / rho_total;
            p_f_rho[2 * i + 0] = df_drho_a;
            p_f_rho[2 * i + 1] = df_drho_b;
            p_f_gamma[3 * i + 0] = df_dgamma_a;
            p_f_gamma[3 * i + 1] = 0.0;
            p_f_gamma[3 * i + 2] = df_dgamma_b;
        }
    }
}

/*
 * Simple reference implementation of libxc's MGGA_X_LTA exchange functional.
 */
static void lta_eval_one_channel(
    double rho_s,
    double tau_s,
    double* f_per_vol,
    double* df_drho_s,
    double* df_dtau_s)
{
    const double X_FACTOR_C = 0.9305257363491000250020102180716672510262; /* (3/8)·(3/π)^(1/3)·4^(2/3) */
    const double K_FACTOR_C = 4.557799872345597137288163759599305358515; /* (3/10)·(6π²)^(2/3) = τ_unif,σ/ρ_σ^(5/3) */
    const double LTA_C = 1.0 / K_FACTOR_C;

    const double rho_threshold = 1.0e-15;
    const double tau_threshold = 1.0e-20;
    if (rho_s < rho_threshold) rho_s = rho_threshold;
    if (tau_s < tau_threshold) tau_s = tau_threshold;

    const double rho_13 = cbrt(rho_s);
    const double rho_43 = rho_s * rho_13;
    const double rho_53 = rho_43 * rho_13;

    const double A = LTA_C * tau_s / rho_53;
    const double F_x = pow(A, 0.8);

    *f_per_vol   = -X_FACTOR_C * rho_43 * F_x;
    *df_drho_s   = 0.0;
    *df_dtau_s   = -0.8 * X_FACTOR_C * rho_43 * F_x / tau_s;
}

void my_xc_mgga_exc_vxc(
    size_t npoints,
    const double* p_rho,
    const double* p_gamma,
    const double* p_tau,
    double* p_f,
    double* p_f_rho,
    double* p_f_gamma,
    double* p_f_tau,
    int is_spin_polarized)
{
    (void) p_gamma;  /* LTA has no gradient dependence */

    const double rho_threshold = 1.0e-15;

    if (!is_spin_polarized) {
        for (size_t i = 0; i < npoints; i++) {
            const double rho = p_rho[i];
            const double tau = p_tau[i];

            if (rho < rho_threshold) {
                p_f[i] = 0.0;
                p_f_rho[i] = 0.0;
                p_f_gamma[i] = 0.0;
                p_f_tau[i] = 0.0;
                continue;
            }

            const double rho_s = 0.5 * rho;
            const double tau_s = 0.5 * tau;

            double f_s, df_drho_s, df_dtau_s;
            lta_eval_one_channel(rho_s, tau_s, &f_s, &df_drho_s, &df_dtau_s);

            /* Both spin channels contribute equally. */
            const double f_total_per_vol = 2.0 * f_s;

            p_f[i] = f_total_per_vol / rho;
            p_f_rho[i] = df_drho_s;
            p_f_gamma[i] = 0.0;
            p_f_tau[i] = df_dtau_s;
        }
        return;
    } else {

        for (size_t i = 0; i < npoints; i++) {
            const double rho_a = p_rho[2 * i + 0];
            const double rho_b = p_rho[2 * i + 1];
            const double rho_total = rho_a + rho_b;

            const double tau_a = p_tau[2 * i + 0];
            const double tau_b = p_tau[2 * i + 1];

            if (rho_total < rho_threshold) {
                p_f[i] = 0.0;
                p_f_rho[2 * i + 0] = 0.0;
                p_f_rho[2 * i + 1] = 0.0;
                p_f_gamma[3 * i + 0] = 0.0;
                p_f_gamma[3 * i + 1] = 0.0;
                p_f_gamma[3 * i + 2] = 0.0;
                p_f_tau[2 * i + 0] = 0.0;
                p_f_tau[2 * i + 1] = 0.0;
                continue;
            }

            double f_a, df_drho_a, df_dtau_a;
            double f_b, df_drho_b, df_dtau_b;
            lta_eval_one_channel(rho_a, tau_a, &f_a, &df_drho_a, &df_dtau_a);
            lta_eval_one_channel(rho_b, tau_b, &f_b, &df_drho_b, &df_dtau_b);

            p_f[i] = (f_a + f_b) / rho_total;
            p_f_rho[2 * i + 0] = df_drho_a;
            p_f_rho[2 * i + 1] = df_drho_b;
            p_f_gamma[3 * i + 0] = 0.0;
            p_f_gamma[3 * i + 1] = 0.0;
            p_f_gamma[3 * i + 2] = 0.0;
            p_f_tau[2 * i + 0] = df_dtau_a;
            p_f_tau[2 * i + 1] = df_dtau_b;
        }
    }
}

/*
 * Sanitize routines to zeros outdensity (and derivatives thereof) inputs
 * where the results are either nonphysical or numerically unstable.  The
 * !(x >= y) syntax used in the comparisons is an idiomatic way to check
 * for x < y while catching the case where either x or y is NaN.
 */
#define SANITIZE_RHO_SAFE  1.0e-10
#define SANITIZE_ZETA_SAFE 1.0e-8
#define SANITIZE_XS_SQ_MAX 1.0e1
#define SANITIZE_ZETA_MGGA 1.0e-4

static void sanitize_grid_rks_lda(size_t npoint, double *h_rho)
{
    size_t kept = 0, zeroed = 0;
    for (size_t i = 0; i < npoint; i++) {
        if (!(h_rho[i] >= SANITIZE_RHO_SAFE)) {
            h_rho[i] = 0.0;
            zeroed++;
        } else {
            kept++;
        }
    }
    printf(" >> sanitize_grid_rks_lda: kept %zu / %zu points (zeroed %zu)\n",
           kept, npoint, zeroed);
}

static void sanitize_grid_uks_lda(size_t npoint, double *h_rho)
{
    size_t kept = 0, zeroed = 0;
    for (size_t i = 0; i < npoint; i++) {
        double *rho_a = &h_rho[2 * i + 0];
        double *rho_b = &h_rho[2 * i + 1];

        int zero_out = 0;
        if (!(*rho_a >= SANITIZE_RHO_SAFE)) zero_out = 1;
        if (!(*rho_b >= SANITIZE_RHO_SAFE)) zero_out = 1;
        if (!zero_out) {
            const double rt = *rho_a + *rho_b;
            if (2.0 * (*rho_a) < SANITIZE_ZETA_SAFE * rt) zero_out = 1;
            if (2.0 * (*rho_b) < SANITIZE_ZETA_SAFE * rt) zero_out = 1;
        }

        if (zero_out) {
            *rho_a = 0.0;  *rho_b = 0.0;
            zeroed++;
        } else {
            kept++;
        }
    }
    printf(" >> sanitize_grid_uks_lda: kept %zu / %zu points (zeroed %zu)\n",
           kept, npoint, zeroed);
}

static void sanitize_grid_uks_mgga(size_t npoint, double *h_rho, double *h_tau)
{
    size_t kept = 0, zeroed = 0;
    for (size_t i = 0; i < npoint; i++) {
        double *rho_a = &h_rho[2 * i + 0];
        double *rho_b = &h_rho[2 * i + 1];
        double *tau_a = &h_tau[2 * i + 0];
        double *tau_b = &h_tau[2 * i + 1];

        int zero_out = 0;
        if (!(*rho_a >= SANITIZE_RHO_SAFE)) zero_out = 1;
        if (!(*rho_b >= SANITIZE_RHO_SAFE)) zero_out = 1;
        if (!zero_out) {
            const double rt = *rho_a + *rho_b;
            if (2.0 * (*rho_a) < SANITIZE_ZETA_MGGA * rt) zero_out = 1;
            if (2.0 * (*rho_b) < SANITIZE_ZETA_MGGA * rt) zero_out = 1;
        }

        if (zero_out) {
            *rho_a = 0.0;  *rho_b = 0.0;
            *tau_a = 0.0;  *tau_b = 0.0;
            zeroed++;
        } else {
            kept++;
        }
    }
    printf(" >> sanitize_grid_uks_mgga: kept %zu / %zu points (zeroed %zu)\n",
           kept, npoint, zeroed);
}

static void sanitize_grid_rks_gga(size_t npoint, double *h_rho, double *h_gamma)
{
    size_t kept = 0, zeroed = 0;
    for (size_t i = 0; i < npoint; i++) {
        double *rho   = &h_rho  [i];
        double *gamma = &h_gamma[i];

        int zero_out = 0;
        if (!(*rho   >= SANITIZE_RHO_SAFE)) zero_out = 1;
        if (!(*gamma >= 0.0))               zero_out = 1;
        if (!zero_out) {
            const double rho_13_   = cbrt(*rho);
            const double rho_43_   = (*rho) * rho_13_;
            const double rho_83_   = rho_43_ * rho_43_;
            const double gamma_eff = (*gamma) * cbrt(4.0);
            if (gamma_eff > SANITIZE_XS_SQ_MAX * rho_83_) zero_out = 1;
        }

        if (zero_out) {
            *rho = 0.0;  *gamma = 0.0;
            zeroed++;
        } else {
            kept++;
        }
    }
    printf(" >> sanitize_grid_rks_gga: kept %zu / %zu points (zeroed %zu)\n",
           kept, npoint, zeroed);
}

static void sanitize_grid_uks_gga(size_t npoint, double *h_rho, double *h_gamma)
{
    size_t kept = 0, zeroed = 0;
    for (size_t i = 0; i < npoint; i++) {
        double *rho_a = &h_rho  [2 * i + 0];
        double *rho_b = &h_rho  [2 * i + 1];
        double *saa   = &h_gamma[3 * i + 0];
        double *sab   = &h_gamma[3 * i + 1];
        double *sbb   = &h_gamma[3 * i + 2];

        int zero_out = 0;
        if (!(*rho_a >= SANITIZE_RHO_SAFE)) zero_out = 1;
        if (!(*rho_b >= SANITIZE_RHO_SAFE)) zero_out = 1;
        if (!zero_out) {
            const double rt = *rho_a + *rho_b;
            if (2.0 * (*rho_a) < SANITIZE_ZETA_SAFE * rt) zero_out = 1;
            if (2.0 * (*rho_b) < SANITIZE_ZETA_SAFE * rt) zero_out = 1;
        }
        if (!(*saa >= 0.0))    zero_out = 1;
        if (!(*sbb >= 0.0))    zero_out = 1;
        if (!isfinite(*sab))   zero_out = 1;
        /* Per-spin reduced-gradient ceiling: σ_σσ ≤ XS_SQ_MAX·ρ_σ^(8/3). */
        if (!zero_out) {
            const double rho_a_13 = cbrt(*rho_a);
            const double rho_a_43 = (*rho_a) * rho_a_13;
            const double rho_a_83 = rho_a_43 * rho_a_43;
            if (*saa > SANITIZE_XS_SQ_MAX * rho_a_83) zero_out = 1;
        }
        if (!zero_out) {
            const double rho_b_13 = cbrt(*rho_b);
            const double rho_b_43 = (*rho_b) * rho_b_13;
            const double rho_b_83 = rho_b_43 * rho_b_43;
            if (*sbb > SANITIZE_XS_SQ_MAX * rho_b_83) zero_out = 1;
        }

        if (zero_out) {
            *rho_a = 0.0;  *rho_b = 0.0;
            *saa   = 0.0;  *sab   = 0.0;  *sbb = 0.0;
            zeroed++;
            continue;
        }

        /* Cauchy-Schwarz clip on the cross-spin term. */
        const double s_max = sqrt((*saa) * (*sbb));
        if (*sab >  s_max) *sab =  s_max;
        if (*sab < -s_max) *sab = -s_max;
        kept++;
    }
    printf(" >> sanitize_grid_uks_gga: kept %zu / %zu points (zeroed %zu)\n",
           kept, npoint, zeroed);
}

void run_ansatz(cuestHandle_t handle, parsedXYZFile_t* xyzData, AtomShellData_t* shellData, cuestXCAdvancedComputeParametersApproximation_t ansatz)
{

    /* Unpack the AtomShellData_t struct */
    uint64_t numAtoms = shellData->numAtoms;
    uint64_t* numShellsPerAtom = shellData->numShellsPerAtom;
    cuestAOShell_t* shells = shellData->shells;

    /************************/
    /* cuEST AO Basis Setup */
    /************************/

    /* Declare the AO basis handle. */
    cuestAOBasis_t basis;

    /* Declare the AO basis parameter handle. */
    cuestAOBasisParameters_t basisParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_AOBASIS_PARAMETERS, 
        &basisParameters));

    /* Allocate space for workspace descriptors. Will be used to determine how large the workspace needs to be. */
    cuestWorkspaceDescriptor_t* persistentWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    cuestWorkspaceDescriptor_t* temporaryWorkspaceDescriptor = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));

    /* Make the workspace query to populate the workspace descriptors. */
    checkCuestErrors(cuestAOBasisCreateWorkspaceQuery(
        handle, 
        numAtoms, 
        numShellsPerAtom, 
        (const cuestAOShell_t*) shells, 
        basisParameters, 
        persistentWorkspaceDescriptor, 
        temporaryWorkspaceDescriptor, 
        &basis));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentBasisWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryBasisWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the AO basis handle. */
    checkCuestErrors(cuestAOBasisCreate(
        handle, 
        numAtoms, 
        numShellsPerAtom, 
        (const cuestAOShell_t*) shells, 
        basisParameters, 
        persistentBasisWorkspace, 
        temporaryBasisWorkspace, 
        &basis));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryBasisWorkspace);

    /* The AO basis parameters are no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_AOBASIS_PARAMETERS, 
        basisParameters));

    /***************************/
    /* cuEST Atomic Grid Setup */
    /***************************/

    /* This forms an unpruned (75, 302) grid with Ahlrichs radial quadrature. */
    cuestAtomGrid_t* atomGrid = formDirectProductAtomGrid(
        handle,
        xyzData,
        75,
        302);

    /******************************/
    /* cuEST Molecular Grid Setup */
    /******************************/

    /* Declare the molecular grid handle. */
    cuestMolecularGrid_t molecularGrid;

    /* Declare and create the molecular grid parameter handle. */
    cuestMolecularGridParameters_t molecularGridParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_MOLECULARGRID_PARAMETERS, 
        &molecularGridParameters));

    /* Determine the workspaces required to construct the molecular grid. */
    checkCuestErrors(cuestMolecularGridCreateWorkspaceQuery(
        handle,
        xyzData->numAtoms,
        atomGrid,
        xyzData->xyzCPU,   
        molecularGridParameters,
        persistentWorkspaceDescriptor,
        temporaryWorkspaceDescriptor,
        &molecularGrid));
    
    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentGridWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryGridWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /* Create the molecular grid. */
    checkCuestErrors(cuestMolecularGridCreate(
        handle,
        xyzData->numAtoms,
        atomGrid,
        xyzData->xyzCPU,   
        molecularGridParameters,
        persistentGridWorkspace,
        temporaryGridWorkspace,
        &molecularGrid));

    /* The molecular grid parameter handle is no longer needed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_MOLECULARGRID_PARAMETERS,
        molecularGridParameters));

    /* The temporary workspace can be freed. */
    freeWorkspace(temporaryGridWorkspace);

    /* The atom grid is not needed once the molecular grid is created */
    for (int n=0; n<xyzData->numAtoms; n++) {
        checkCuestErrors(cuestAtomGridDestroy(atomGrid[n]));
    }
    free(atomGrid);

    /********************************/
    /* cuEST XC Integral Plan Setup */
    /********************************/

    /* Declare the XC integral plan handle. */
    cuestXCIntPlan_t xcIntPlan;

    /* Declare and create the XC integral plan parameters handle. */
    cuestXCIntPlanParameters_t xcIntPlanParameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCINTPLAN_PARAMETERS, 
        &xcIntPlanParameters));

    /* Determine the workspaces required to construct the XC integral plan. */
    checkCuestErrors(cuestXCIntPlanCreateWorkspaceQuery(
        handle, 
        basis, 
        molecularGrid,
        CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_PBE,
        xcIntPlanParameters, 
        persistentWorkspaceDescriptor, 
        temporaryWorkspaceDescriptor, 
        &xcIntPlan));

    /* Allocate buffers for the temporary and persistent workspaces. */
    cuestWorkspace_t* persistentXCIntPlanWorkspace = allocateWorkspace(persistentWorkspaceDescriptor);;
    cuestWorkspace_t* temporaryXCIntPlanWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);;

    /*
     * Create the XC integral plan.  We're not using cuEST to evaluate functionals directly
     * in this example, so the HF functional spec is just an arbitrary placeholder here.
     */
    checkCuestErrors(cuestXCIntPlanCreate(
        handle, 
        basis, 
        molecularGrid,
        CUEST_XCINTPLAN_PARAMETERS_FUNCTIONAL_HF,
        xcIntPlanParameters, 
        persistentXCIntPlanWorkspace, 
        temporaryXCIntPlanWorkspace, 
        &xcIntPlan));

    /* The XC integral plan paramaters can be destroyed. */
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCINTPLAN_PARAMETERS, 
        xcIntPlanParameters));

    /* The temporary workspace is no longer needed. */
    freeWorkspace(temporaryXCIntPlanWorkspace);

    /* Query the AO basis for the number of basis functions */
    uint64_t nao = 0;
    checkCuestErrors(cuestQuery(
        handle, 
        CUEST_AOBASIS, 
        basis, 
        CUEST_AOBASIS_NUM_AO,        
        &nao,        
        sizeof(uint64_t)));

    uint64_t npoint = 0;
    checkCuestErrors(cuestQuery(
        handle, 
        CUEST_MOLECULARGRID, 
        molecularGrid, 
        CUEST_MOLECULARGRID_NUM_POINT,        
        &npoint,        
        sizeof(uint64_t)));

    /* 
     * The amount of memory used as a temporary buffer during collocation is controlled
     * by specifying a variable buffer size. 2 GB is a reasonable workspace size.
     * When possible, larger workspaces might provide additional performance benefits.
     */
    cuestWorkspaceDescriptor_t* variableBufferSize = (cuestWorkspaceDescriptor_t*) malloc(sizeof(cuestWorkspaceDescriptor_t));
    variableBufferSize->hostBufferSizeInBytes = 0;
    variableBufferSize->deviceBufferSizeInBytes = 2000000000;

    /*****************************************/
    /* cuEST Compute the RKS XC Gradient     */
    /*****************************************/

    /* Get nocc for a neutral molecule */
    uint64_t nocc = 0;
    for (size_t i=0; i<xyzData->numAtoms; i++) {
        nocc += (uint64_t) fabs(xyzData->chargesCPU[i]);
    }
    nocc = nocc / 2;

    /* Allocate space for Cocc matrix */
    double *d_Cocc;
    if (cudaMalloc((void**) &d_Cocc, nocc * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    
    /* 
     * Fill with random numbers. In real use cases, these are 
     * occupied molecular orbital coefficients.
     */
    fill_matrix(d_Cocc, nocc, nao);

    /* Compute integration weights and copy to the host */
    double *d_weights;
    if (cudaMalloc((void**)&d_weights, npoint * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    cuestXCIntegrationWeightComputeParameters_t weight_compute_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCINTEGRATIONWEIGHTCOMPUTE_PARAMETERS,
        &weight_compute_parameters));
    checkCuestErrors(cuestXCIntegrationWeightComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_TOTAL,
        weight_compute_parameters,
        temporaryWorkspaceDescriptor,
        d_weights));
    cuestWorkspace_t* temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCIntegrationWeightCompute(
        handle,
        xcIntPlan,
        CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_TOTAL,
        weight_compute_parameters,
        temporaryWorkspace,
        d_weights
        ));
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCINTEGRATIONWEIGHTCOMPUTE_PARAMETERS,
        weight_compute_parameters));
    freeWorkspace(temporaryWorkspace);

    double *h_weights = (double*) malloc(npoint * sizeof(double));
    if(cudaMemcpy(h_weights, d_weights, npoint * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }
    cudaFree(d_weights);

    // => Compute grid density (and derivatives) <= //
    cuestXCDensityComputeParameters_t density_compute_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCDENSITYCOMPUTE_PARAMETERS,
        &density_compute_parameters));

    uint64_t ncomponents = 0;
    switch(ansatz) {
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_LDA:
            ncomponents = 1;
            break;
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_GGA:
            ncomponents = 4;
            break;
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_METAGGA:
            ncomponents = 5;
            break;
        default:
            fprintf(stderr, "Invalid ansatz type\n");
            exit(EXIT_FAILURE);
    }

    double *d_rho;
    if (cudaMalloc((void**)&d_rho, npoint * ncomponents * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    checkCuestErrors(cuestXCDensityComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        ansatz,
        density_compute_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        nocc,
        d_Cocc,
        d_rho
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCDensityCompute(
        handle,
        xcIntPlan,
        ansatz,
        density_compute_parameters,
        variableBufferSize,
        temporaryWorkspace,
        nocc,
        d_Cocc,
        d_rho
        ));
    freeWorkspace(temporaryWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCDENSITYCOMPUTE_PARAMETERS,
        density_compute_parameters));

    /* Copy the density and its derivatives to host, then transpose it for easier access */
    double *h_rho_derivs = (double*) malloc(npoint * ncomponents * sizeof(double));
    if(cudaMemcpy(h_rho_derivs, d_rho, npoint * ncomponents * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }
    cudaFree(d_rho);
    double *h_rho_derivsT = (double*) malloc(npoint * ncomponents * sizeof(double));
    for (int p = 0; p < npoint; p++) {
        for (int c = 0; c < ncomponents; c++) {
            h_rho_derivsT[c * npoint + p] = h_rho_derivs[p * ncomponents + c];
        }
    }
    free(h_rho_derivs);

    double *h_Vxc_grid;
    h_Vxc_grid = (double*) malloc(npoint * ncomponents * sizeof(double));
    if (!h_Vxc_grid) {
        fprintf(stderr, "Failed to allocate memory for Vxc_grid\n");
        exit(EXIT_FAILURE);
    }

    double *h_rho, *h_gamma, *h_tau;
    double *rho_0, *rho_x, *rho_y, *rho_z, *tau_0;
#if CHECK_AGAINST_LIBXC
    double *f_libxc, *f_rho_libxc, *f_gamma_libxc, *f_tau_libxc;
    xc_func_type func;
    double abs_error, rel_error;
    double max_abs_f_error, max_abs_f_rho_error, max_abs_f_gamma_error, max_abs_f_tau_error;
    double max_rel_f_error, max_rel_f_rho_error, max_rel_f_gamma_error, max_rel_f_tau_error;
#endif
    double *f, *f_rho, *f_gamma, *f_tau;
    f = (double*) malloc(npoint * sizeof(double));
    if (!f) {
        fprintf(stderr, "Failed to allocate memory for f\n");
        exit(EXIT_FAILURE);
    }
    f_rho = (double*) malloc(npoint * sizeof(double));
    if (!f_rho) {
        fprintf(stderr, "Failed to allocate memory for f_rho\n");
        exit(EXIT_FAILURE);
    }
    switch (ansatz) {
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_LDA:
            rho_0 = &h_rho_derivsT[0 * npoint];
            h_rho = (double*) malloc(npoint * sizeof(double));
            if (!h_rho) {
                fprintf(stderr, "Failed to allocate memory for h_rho\n");
                exit(EXIT_FAILURE);
            }
            for (int point = 0; point < npoint; point++) {
                h_rho[point] = 2 * h_rho_derivsT[point];
            }

            /*
             * This would not be needed if we were using real densities; it's just to catch
             * the cases where randomly generated density derivatives are unphysical.
             */
            sanitize_grid_rks_lda(npoint, h_rho);

            my_xc_lda_exc_vxc(
                npoint,
                h_rho,
                f,
                f_rho,
                0);

            /* Any other exchange and/or correlation functionals will be handled here */

            /* Clamp NaNs */
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f[i])) nan_found = 1;
                if (!isfinite(f_rho[i])) nan_found = 1;
                if (nan_found) {
                    f[i] = 0.0;
                    f_rho[i] = 0.0;
                }
            }
#if CHECK_AGAINST_LIBXC
            f_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_rho_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_rho_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_rho_libxc\n");
                exit(EXIT_FAILURE);
            }
            xc_func_init(&func, XC_LDA_X, XC_UNPOLARIZED);
            xc_lda_exc_vxc(
                &func,
                npoint,
                h_rho,
                f_libxc,
                f_rho_libxc);
            xc_func_end(&func);
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f_libxc[i])) nan_found = 1;
                if (!isfinite(f_rho_libxc[i])) nan_found = 1;
                if (nan_found) {
                    f_libxc[i] = 0.0;
                    f_rho_libxc[i] = 0.0;
                }
            }
            max_abs_f_error = 0.0;
            max_abs_f_rho_error = 0.0;
            max_rel_f_error = 0.0;
            max_rel_f_rho_error = 0.0;
            for (int point = 0; point < npoint; point++) {
                if (f_libxc[point] != 0.0) {
                    abs_error = fabs(f[point] - f_libxc[point]);
                    if (abs_error > max_abs_f_error) max_abs_f_error = abs_error;
                    rel_error = abs_error / fabs(f_libxc[point]);
                    if (rel_error > max_rel_f_error) max_rel_f_error = rel_error;
                }
                if (f_rho_libxc[point] != 0.0) {
                    abs_error = fabs(f_rho[point] - f_rho_libxc[point]);
                    if (abs_error > max_abs_f_rho_error) max_abs_f_rho_error = abs_error;
                    rel_error = abs_error / fabs(f_rho_libxc[point]);
                    if (rel_error > max_rel_f_rho_error) max_rel_f_rho_error = rel_error;
                }
            }
            printf("RKS LDA  Maximum error of f:       %8.2e (absolute): %8.2e (relative)\n", max_abs_f_error, max_rel_f_error);
            printf("RKS LDA  Maximum error of f_rho:   %8.2e (absolute): %8.2e (relative)\n", max_abs_f_rho_error, max_rel_f_rho_error);
            free(f_libxc);
            free(f_rho_libxc);
#endif
            for (int point = 0; point < npoint; point++) {
                double w = h_weights[point];
                h_Vxc_grid[point] = w * f_rho[point];
            }

            free(h_rho);

            break;
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_GGA:
            h_rho = (double*) malloc(npoint * sizeof(double));
            if (!h_rho) {
                fprintf(stderr, "Failed to allocate memory for rho\n");
                exit(EXIT_FAILURE);
            }
            h_gamma = (double*) malloc(npoint * sizeof(double));
            if (!h_gamma) {
                fprintf(stderr, "Failed to allocate memory for gamma\n");
                exit(EXIT_FAILURE);
            }    
            rho_0 = &h_rho_derivsT[0 * npoint];
            rho_x = &h_rho_derivsT[1 * npoint];
            rho_y = &h_rho_derivsT[2 * npoint];
            rho_z = &h_rho_derivsT[3 * npoint];
            for (int point = 0; point < npoint; point++) {
                h_rho[point] = 2 * rho_0[point];
                h_gamma[point] = 4 * (rho_x[point] * rho_x[point] + rho_y[point] * rho_y[point] + rho_z[point] * rho_z[point]);
            }

            /*
             * This would not be needed if we were using real densities; it's just to catch
             * the cases where randomly generated density derivatives are unphysical.
             */
            sanitize_grid_rks_gga(npoint, h_rho, h_gamma);

            f_gamma = (double*) malloc(npoint * sizeof(double));
            if (!f_gamma) {
                fprintf(stderr, "Failed to allocate memory for f_gamma\n");
                exit(EXIT_FAILURE);
            }
            my_xc_gga_exc_vxc(
                npoint,
                h_rho,
                h_gamma,
                f,
                f_rho,
                f_gamma,
                0);
#if CHECK_AGAINST_LIBXC
            f_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_rho_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_rho_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_rho_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_gamma_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_gamma_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_gamma_libxc\n");
                exit(EXIT_FAILURE);
            }
            xc_func_init(&func, XC_GGA_X_B86, XC_UNPOLARIZED);
            xc_gga_exc_vxc(
                &func,
                npoint,
                h_rho,
                h_gamma,
                f_libxc,
                f_rho_libxc,
                f_gamma_libxc);
            xc_func_end(&func);
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f_libxc[i])) nan_found = 1;
                if (!isfinite(f_rho_libxc[i])) nan_found = 1;
                if (!isfinite(f_gamma_libxc[i])) nan_found = 1;
                if (nan_found) {
                    f_libxc[i] = 0.0;
                    f_rho_libxc[i] = 0.0;
                    f_gamma_libxc[i] = 0.0;
                }
            }
            max_abs_f_error = 0.0;
            max_abs_f_rho_error = 0.0;
            max_rel_f_error = 0.0;
            max_rel_f_rho_error = 0.0;
            for (int point = 0; point < npoint; point++) {
                if (f_libxc[point] != 0.0) {
                    abs_error = fabs(f[point] - f_libxc[point]);
                    if (abs_error > max_abs_f_error) max_abs_f_error = abs_error;
                    rel_error = abs_error / fabs(f_libxc[point]);
                    if (rel_error > max_rel_f_error) max_rel_f_error = rel_error;
                }
                if (f_rho_libxc[point] != 0.0) {
                    abs_error = fabs(f_rho[point] - f_rho_libxc[point]);
                    if (abs_error > max_abs_f_rho_error) max_abs_f_rho_error = abs_error;
                    rel_error = abs_error / fabs(f_rho_libxc[point]);
                    if (rel_error > max_rel_f_rho_error) max_rel_f_rho_error = rel_error;
                }
                if (f_gamma_libxc[point] != 0.0) {
                    abs_error = fabs(f_gamma[point] - f_gamma_libxc[point]);
                    if (abs_error > max_abs_f_gamma_error) max_abs_f_gamma_error = abs_error;
                    rel_error = abs_error / fabs(f_gamma_libxc[point]);
                    if (rel_error > max_rel_f_gamma_error) max_rel_f_gamma_error = rel_error;
                }
            }
            printf("RKS GGA  Maximum error of f:       %8.2e (absolute): %8.2e (relative)\n", max_abs_f_error, max_rel_f_error);
            printf("RKS GGA  Maximum error of f_rho:   %8.2e (absolute): %8.2e (relative)\n", max_abs_f_rho_error, max_rel_f_rho_error);
            printf("RKS GGA  Maximum error of f_gamma: %8.2e (absolute): %8.2e (relative)\n", max_abs_f_gamma_error, max_rel_f_gamma_error);
            free(f_libxc);
            free(f_rho_libxc);
            free(f_gamma_libxc);
#endif
            /* Any other exchange and/or correlation functionals will be handled here */

            // Clamp NaNs
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f[i])) nan_found = 1;
                if (!isfinite(f_rho[i])) nan_found = 1;
                if (!isfinite(f_gamma[i])) nan_found = 1;
                if (nan_found) {
                    f[i] = 0.0;
                    f_rho[i] = 0.0;
                    f_gamma[i] = 0.0;
                }
            }
            for (int point = 0; point < npoint; point++) {
                double w = h_weights[point];
                h_Vxc_grid[4 * point + 0] = w * f_rho[point];
                h_Vxc_grid[4 * point + 1] = 4 * w * f_gamma[point] * rho_x[point];
                h_Vxc_grid[4 * point + 2] = 4 * w * f_gamma[point] * rho_y[point];
                h_Vxc_grid[4 * point + 3] = 4 * w * f_gamma[point] * rho_z[point];
            }

            free(h_rho);
            free(h_gamma);
            free(f_gamma);

            break;
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_METAGGA:
            h_rho = (double*) malloc(npoint * sizeof(double));
            if (!h_rho) {
                fprintf(stderr, "Failed to allocate memory for h_rho\n");
                exit(EXIT_FAILURE);
            }
            h_gamma = (double*) malloc(npoint * sizeof(double));
            if (!h_gamma) {
                fprintf(stderr, "Failed to allocate memory for h_gamma\n");
                exit(EXIT_FAILURE);
            }
            h_tau = (double*) malloc(npoint * sizeof(double));
            if (!h_tau) {
                fprintf(stderr, "Failed to allocate memory for h_tau\n");
                exit(EXIT_FAILURE);
            }
            rho_0 = &h_rho_derivsT[0 * npoint];
            rho_x = &h_rho_derivsT[1 * npoint];
            rho_y = &h_rho_derivsT[2 * npoint];
            rho_z = &h_rho_derivsT[3 * npoint];
            tau_0 = &h_rho_derivsT[4 * npoint];
            for (int point = 0; point < npoint; point++) {
                h_rho[point] = 2 * rho_0[point];
                h_gamma[point] = 4 * (rho_x[point] * rho_x[point] + rho_y[point] * rho_y[point] + rho_z[point] * rho_z[point]);
                h_tau[point] = 2 * tau_0[point];
            }

            /*
             * MGGA_X_LTA has no σ dependence and only requires ρ_σ>0
             * and τ_σ>0, which the inner kernel already guards against —
             * no separate grid sanitization is needed for this functional.
             */

            f_gamma = (double*) malloc(npoint * sizeof(double));
            if (!f_gamma) {
                fprintf(stderr, "Failed to allocate memory for f_gamma\n");
                exit(EXIT_FAILURE);
            }
            f_tau = (double*) malloc(npoint * sizeof(double));
            if (!f_tau) {
                fprintf(stderr, "Failed to allocate memory for f_tau\n");
                exit(EXIT_FAILURE);
            }
            my_xc_mgga_exc_vxc(
                npoint,
                h_rho,
                h_gamma,
                h_tau,
                f,
                f_rho,
                f_gamma,
                f_tau,
                0);

            /* Any other exchange and/or correlation functionals will be handled here */

            /* Clamp NaNs */
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f[i])) nan_found = 1;
                if (!isfinite(f_rho[i])) nan_found = 1;
                if (!isfinite(f_gamma[i])) nan_found = 1;
                if (!isfinite(f_tau[i])) nan_found = 1;
                if (nan_found) {
                    f[i] = 0.0;
                    f_rho[i] = 0.0;
                    f_gamma[i] = 0.0;
                    f_tau[i] = 0.0;
                }
            }
#if CHECK_AGAINST_LIBXC
            f_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_rho_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_rho_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_rho_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_gamma_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_gamma_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_gamma_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_tau_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_tau_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_tau_libxc\n");
                exit(EXIT_FAILURE);
            }
            xc_func_init(&func, XC_MGGA_X_LTA, XC_UNPOLARIZED);
            xc_mgga_exc_vxc(
                &func,
                npoint,
                h_rho,
                h_gamma,
                NULL,
                h_tau,
                f_libxc,
                f_rho_libxc,
                f_gamma_libxc,
                NULL,
                f_tau_libxc);
            xc_func_end(&func);
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f_libxc[i])) nan_found = 1;
                if (!isfinite(f_rho_libxc[i])) nan_found = 1;
                if (!isfinite(f_gamma_libxc[i])) nan_found = 1;
                if (!isfinite(f_tau_libxc[i])) nan_found = 1;
                if (nan_found) {
                    f_libxc[i] = 0.0;
                    f_rho_libxc[i] = 0.0;
                    f_gamma_libxc[i] = 0.0;
                    f_tau_libxc[i] = 0.0;
                }
            }
            max_abs_f_error = 0.0;
            max_abs_f_rho_error = 0.0;
            max_abs_f_gamma_error = 0.0;
            max_abs_f_tau_error = 0.0;
            max_rel_f_error = 0.0;
            max_rel_f_rho_error = 0.0;
            max_rel_f_gamma_error = 0.0;
            for (int point = 0; point < npoint; point++) {
                if (f_libxc[point] != 0.0) {
                    abs_error = fabs(f[point] - f_libxc[point]);
                    if (abs_error > max_abs_f_error) max_abs_f_error = abs_error;
                    rel_error = abs_error / fabs(f_libxc[point]);
                    if (rel_error > max_rel_f_error) max_rel_f_error = rel_error;
                }
            }
            for (int point = 0; point < npoint; point++) {
                if (f_rho_libxc[point] != 0.0) {
                    abs_error = fabs(f_rho[point] - f_rho_libxc[point]);
                    if (abs_error > max_abs_f_rho_error) max_abs_f_rho_error = abs_error;
                    rel_error = abs_error / fabs(f_rho_libxc[point]);
                    if (rel_error > max_rel_f_rho_error) max_rel_f_rho_error = rel_error;
                }
            }
            for (int point = 0; point < npoint; point++) {
                if (f_gamma_libxc[point] != 0.0) {
                    abs_error = fabs(f_gamma[point] - f_gamma_libxc[point]);
                    if (abs_error > max_abs_f_gamma_error) max_abs_f_gamma_error = abs_error;
                    rel_error = abs_error / fabs(f_gamma_libxc[point]);
                    if (rel_error > max_rel_f_gamma_error) max_rel_f_gamma_error = rel_error;
                }
            }
            for (int point = 0; point < npoint; point++) {
                if (f_tau_libxc[point] != 0.0) {
                    abs_error = fabs(f_tau[point] - f_tau_libxc[point]);
                    if (abs_error > max_abs_f_tau_error) max_abs_f_tau_error = abs_error;
                    rel_error = abs_error / fabs(f_tau_libxc[point]);
                    if (rel_error > max_rel_f_tau_error) max_rel_f_tau_error = rel_error;
                }
            }
            printf("RKS mGGA Maximum error of f:       %8.2e (absolute): %8.2e (relative)\n", max_abs_f_error, max_rel_f_error);
            printf("RKS mGGA Maximum error of f_rho:   %8.2e (absolute): %8.2e (relative)\n", max_abs_f_rho_error, max_rel_f_rho_error);
            printf("RKS mGGA Maximum error of f_gamma: %8.2e (absolute): %8.2e (relative)\n", max_abs_f_gamma_error, max_rel_f_gamma_error);
            printf("RKS mGGA Maximum error of f_tau:   %8.2e (absolute): %8.2e (relative)\n", max_abs_f_tau_error, max_rel_f_tau_error);
            free(f_libxc);
            free(f_rho_libxc);
            free(f_gamma_libxc);
            free(f_tau_libxc);
        #endif

            for (int point = 0; point < npoint; point++) {
                double w = h_weights[point];
                h_Vxc_grid[5 * point + 0] = w * f_rho[point];
                h_Vxc_grid[5 * point + 1] = 4 * w * f_gamma[point] * rho_x[point];
                h_Vxc_grid[5 * point + 2] = 4 * w * f_gamma[point] * rho_y[point];
                h_Vxc_grid[5 * point + 3] = 4 * w * f_gamma[point] * rho_z[point];
                h_Vxc_grid[5 * point + 4] = w * f_tau[point];
            }

            free(h_rho);
            free(h_gamma);
            free(h_tau);
            free(f_gamma);
            free(f_tau);

            break;
        default:
            fprintf(stderr, "Invalid ansatz type\n");
            exit(EXIT_FAILURE);
    }

    /*
     * => Build the bare per-point XC energy and the XC energy/density <=
     *
     * h_exc[point] = f[point] * rho_0[point] is the alpha-only per-point
     * Exc density (without integration weights).  This is what
     * cuestXCGridDerivativeCompute expects; the factor of 2 for RKS
     * doubled occupation is applied to the final gradient further down.
     */
    double *h_exc = (double*) malloc(npoint * sizeof(double));
    if (!h_exc) {
        fprintf(stderr, "Failed to allocate memory for h_exc\n");
        exit(EXIT_FAILURE);
    }
    double Exc = 0.0;
    double integrated_density = 0.0;
    for (int point = 0; point < npoint; point++) {
        h_exc[point] = f[point] * rho_0[point];
        Exc += 2 * h_weights[point] * h_exc[point];
        integrated_density += 2 * h_weights[point] * rho_0[point];
    }

    free(f);
    free(f_rho);
    free(h_weights);
    free(h_rho_derivsT);

    /* Allocate device buffers for the gradient pipeline. */
    double *d_Vxc_grid;
    if (cudaMalloc((void**)&d_Vxc_grid, npoint * ncomponents * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if(cudaMemcpy(d_Vxc_grid, h_Vxc_grid, npoint * ncomponents * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to device\n");
        exit(EXIT_FAILURE);
    }
    free(h_Vxc_grid);

    double *d_exc;
    if (cudaMalloc((void**)&d_exc, npoint * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(d_exc, h_exc, npoint * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to device\n");
        exit(EXIT_FAILURE);
    }
    free(h_exc);

    /* Output gradient buffers: the basis-center atomic gradient and the
     * per-point grid-gradient contribution that feeds the grid derivative
     * call. */
    double *d_grad_atom;
    if (cudaMalloc((void**)&d_grad_atom, numAtoms * 3 * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    double *d_grad_grid;
    if (cudaMalloc((void**)&d_grad_grid, npoint * 3 * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /*
     * cuestXCDerivativeCompute: given the per-point XC potential
     * (d_Vxc_grid built above) and the occupied MO coefficients, this
     * produces (i) the AO basis-center contribution to the atomic
     * gradient (d_grad_atom) and (ii) the per-point grid gradient
     * contribution (d_grad_grid) which is consumed by
     * cuestXCGridDerivativeCompute below.
     */
    cuestXCDerivativeComputeParameters_t derivative_compute_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCDERIVATIVECOMPUTE_PARAMETERS,
        &derivative_compute_parameters));
    checkCuestErrors(cuestXCDerivativeComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        ansatz,
        derivative_compute_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        nocc,
        d_Cocc,
        d_Vxc_grid,
        d_grad_atom,
        d_grad_grid
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCDerivativeCompute(
        handle,
        xcIntPlan,
        ansatz,
        derivative_compute_parameters,
        variableBufferSize,
        temporaryWorkspace,
        nocc,
        d_Cocc,
        d_Vxc_grid,
        d_grad_atom,
        d_grad_grid
    ));
    freeWorkspace(temporaryWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCDERIVATIVECOMPUTE_PARAMETERS,
        derivative_compute_parameters));
    cudaFree(d_Vxc_grid);

    /* Copy the basis-center atomic gradient to the host accumulator. */
    double *h_grad = (double*) malloc(numAtoms * 3 * sizeof(double));
    if (!h_grad) {
        fprintf(stderr, "Failed to allocate memory for h_grad\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(h_grad, d_grad_atom, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }

    /*
     * cuestXCGridDerivativeCompute: given the bare per-point Exc
     * density (d_exc) and the grid-gradient contribution from
     * cuestXCDerivativeCompute (d_grad_grid), assemble the contribution to the
     * atomic gradient from motion of the grid points.  The result is written
     * into d_grad_atom, overwriting the basis-center contribution that we
     * already copied out above.
     */
    cuestXCGridDerivativeComputeParameters_t grid_derivative_compute_parameters;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCGRIDDERIVATIVECOMPUTE_PARAMETERS,
        &grid_derivative_compute_parameters));
    checkCuestErrors(cuestXCGridDerivativeComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        grid_derivative_compute_parameters,
        temporaryWorkspaceDescriptor,
        d_exc,
        d_grad_grid,
        d_grad_atom
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCGridDerivativeCompute(
        handle,
        xcIntPlan,
        grid_derivative_compute_parameters,
        temporaryWorkspace,
        d_exc,
        d_grad_grid,
        d_grad_atom
    ));
    freeWorkspace(temporaryWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCGRIDDERIVATIVECOMPUTE_PARAMETERS,
        grid_derivative_compute_parameters));

    /* Add the grid-center contribution to the basis-center contribution
     * already accumulated in h_grad, then multiply by 2 to account for
     * the double-occupancy of RKS orbitals. */
    double *h_grad_grid_contrib = (double*) malloc(numAtoms * 3 * sizeof(double));
    if (!h_grad_grid_contrib) {
        fprintf(stderr, "Failed to allocate memory for h_grad_grid_contrib\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(h_grad_grid_contrib, d_grad_atom, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < numAtoms * 3; i++) {
        h_grad[i] = 2.0 * (h_grad[i] + h_grad_grid_contrib[i]);
    }
    free(h_grad_grid_contrib);

    printf("RKS XC gradient (Hartree/Bohr):\n");
    for (uint64_t a = 0; a < numAtoms; a++) {
        printf("  atom %3lu: %16.10f %16.10f %16.10f\n",
               (unsigned long) a,
               h_grad[3*a+0], h_grad[3*a+1], h_grad[3*a+2]);
    }
    free(h_grad);

    cudaFree(d_grad_atom);
    cudaFree(d_grad_grid);
    cudaFree(d_exc);

    /* Free Cocc matrix */
    if (cudaFree((void*) d_Cocc) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /*****************************************/
    /* cuEST Compute the UKS XC Gradient     */
    /*****************************************/

    /* Get noccA/noccB for a doublet cation. */
    uint64_t noccA = nocc;
    uint64_t noccB = nocc-1;

    /* Allocate space for Cocc matrices */
    double *d_CoccA, *d_CoccB;
    if (cudaMalloc((void**) &d_CoccA, noccA * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void**) &d_CoccB, noccB * nao * sizeof(double))) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    
    /* 
     * Fill with random numbers. In real use cases, these are 
     * occupied molecular orbital coefficients.
     */
    fill_matrix(d_CoccA, noccA, nao);
    fill_matrix(d_CoccB, noccB, nao);

    /* Compute integration weights and copy to the host */
    if (cudaMalloc((void**)&d_weights, npoint * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCINTEGRATIONWEIGHTCOMPUTE_PARAMETERS,
        &weight_compute_parameters));
    checkCuestErrors(cuestXCIntegrationWeightComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_TOTAL,
        weight_compute_parameters,
        temporaryWorkspaceDescriptor,
        d_weights
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCIntegrationWeightCompute(
        handle,
        xcIntPlan,
        CUEST_XCINTEGRATIONWEIGHT_PARAMETERS_WEIGHTTYPE_TOTAL,
        weight_compute_parameters,
        temporaryWorkspace,
        d_weights
        ));
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCINTEGRATIONWEIGHTCOMPUTE_PARAMETERS,
        weight_compute_parameters));
    freeWorkspace(temporaryWorkspace);
    h_weights = (double*) malloc(npoint * sizeof(double));
    if(cudaMemcpy(h_weights, d_weights, npoint * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }
    cudaFree(d_weights);

    /* Compute grid density (and derivatives) */
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCDENSITYCOMPUTE_PARAMETERS,
        &density_compute_parameters));

    ncomponents = 0;
    switch(ansatz) {
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_LDA:
            ncomponents = 1;
            break;
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_GGA:
            ncomponents = 4;
            break;
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_METAGGA:
            ncomponents = 5;
            break;
        default:
            fprintf(stderr, "Invalid ansatz type\n");
            exit(EXIT_FAILURE);
    }

    double *d_rho_a;
    if (cudaMalloc((void**)&d_rho_a, npoint * ncomponents * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    double *d_rho_b;
    if (cudaMalloc((void**)&d_rho_b, npoint * ncomponents * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    checkCuestErrors(cuestXCDensityComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        ansatz,
        density_compute_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        noccA,
        d_CoccA,
        d_rho_a
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCDensityCompute(
        handle,
        xcIntPlan,
        ansatz,
        density_compute_parameters,
        variableBufferSize,
        temporaryWorkspace,
        noccA,
        d_CoccA,
        d_rho_a
        ));
    freeWorkspace(temporaryWorkspace);

    checkCuestErrors(cuestXCDensityComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        ansatz,
        density_compute_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        noccB,
        d_CoccB,
        d_rho_b
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCDensityCompute(
        handle,
        xcIntPlan,
        ansatz,
        density_compute_parameters,
        variableBufferSize,
        temporaryWorkspace,
        noccB,
        d_CoccB,
        d_rho_b
        ));
    freeWorkspace(temporaryWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCDENSITYCOMPUTE_PARAMETERS,
        density_compute_parameters));

    /* Copy the density and its derivatives to host, then transpose it for easier access */
    double *h_rho_a_derivs = (double*) malloc(npoint * ncomponents * sizeof(double));
    if (!h_rho_a_derivs) {
        fprintf(stderr, "Failed to allocate memory for h_rho_a_derivs\n");
        exit(EXIT_FAILURE);
    }
    if(cudaMemcpy(h_rho_a_derivs, d_rho_a, npoint * ncomponents * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }
    double *h_rho_b_derivs = (double*) malloc(npoint * ncomponents * sizeof(double));
    if(cudaMemcpy(h_rho_b_derivs, d_rho_b, npoint * ncomponents * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }
    cudaFree(d_rho_a);
    cudaFree(d_rho_b);
    double *h_rho_a_derivsT = (double*) malloc(npoint * ncomponents * sizeof(double));
    if (!h_rho_a_derivsT) {
        fprintf(stderr, "Failed to allocate memory for h_rho_a_derivsT\n");
        exit(EXIT_FAILURE);
    }
    double *h_rho_b_derivsT = (double*) malloc(npoint * ncomponents * sizeof(double));
    if (!h_rho_b_derivsT) {
        fprintf(stderr, "Failed to allocate memory for h_rho_b_derivsT\n");
        exit(EXIT_FAILURE);
    }
    for (int p = 0; p < npoint; p++) {
        for (int c = 0; c < ncomponents; c++) {
            h_rho_a_derivsT[c * npoint + p] = h_rho_a_derivs[p * ncomponents + c];
            h_rho_b_derivsT[c * npoint + p] = h_rho_b_derivs[p * ncomponents + c];
        }
    }
    free(h_rho_a_derivs);
    free(h_rho_b_derivs);

    double *h_Vxc_grid_a = (double*) malloc(npoint * ncomponents * sizeof(double));
    if (!h_Vxc_grid_a) {
        fprintf(stderr, "Failed to allocate memory for h_Vxc_grid_a\n");
        exit(EXIT_FAILURE);
    }
    double *h_Vxc_grid_b = (double*) malloc(npoint * ncomponents * sizeof(double));
    if (!h_Vxc_grid_b) {
        fprintf(stderr, "Failed to allocate memory for h_Vxc_grid_b\n");
        exit(EXIT_FAILURE);
    }

    f = (double*) malloc(npoint * sizeof(double));
    if (!f) {
        fprintf(stderr, "Failed to allocate memory for f\n");
        exit(EXIT_FAILURE);
    }
    f_rho = (double*) malloc(npoint * 2 * sizeof(double));
    if (!f_rho) {
        fprintf(stderr, "Failed to allocate memory for f_rho\n");
        exit(EXIT_FAILURE);
    }

    double *rho_0_a, *rho_0_b, *rho_x_a, *rho_x_b, *rho_y_a, *rho_y_b, *rho_z_a, *rho_z_b, *tau_0_a, *tau_0_b;
    double w, f_gamma_aa, f_gamma_ab, f_gamma_bb;
    switch (ansatz) {
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_LDA:
            rho_0_a = &h_rho_a_derivsT[0];
            rho_0_b = &h_rho_b_derivsT[0];
            h_rho = (double*) malloc(npoint * 2 * sizeof(double));
            if (!h_rho) {
                fprintf(stderr, "Failed to allocate memory for rho\n");
                exit(EXIT_FAILURE);
            }
            for (int point = 0; point < npoint; point++) {
                h_rho[2 * point + 0] = rho_0_a[point];
                h_rho[2 * point + 1] = rho_0_b[point];
            }

            /*
             * This would not be needed if we were using real densities; it's just to catch
             * the cases where randomly generated density derivatives are unphysical.
             */
            sanitize_grid_uks_lda(npoint, h_rho);

            my_xc_lda_exc_vxc(
                npoint,
                h_rho,
                f,
                f_rho,
                1);

            /* Any other exchange and/or correlation functionals will be handled here */

            /* Clamp NaNs */
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f[i])) nan_found = 1;
                if (!isfinite(f_rho[2 * i + 0])) nan_found = 1;
                if (!isfinite(f_rho[2 * i + 1])) nan_found = 1;
                if (nan_found) {
                    f[i] = 0.0;
                    f_rho[2 * i + 0] = 0.0;
                    f_rho[2 * i + 1] = 0.0;
                }
            }
#if CHECK_AGAINST_LIBXC
            f_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_rho_libxc = (double*) malloc(npoint * 2 * sizeof(double));
            if (!f_rho_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_rho_libxc\n");
                exit(EXIT_FAILURE);
            }
            xc_func_init(&func, XC_LDA_X, XC_POLARIZED);
            xc_lda_exc_vxc(
                &func,
                npoint,
                h_rho,
                f_libxc,
                f_rho_libxc);
            xc_func_end(&func);
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f_libxc[i])) nan_found = 1;
                if (!isfinite(f_rho_libxc[2 * i + 0])) nan_found = 1;
                if (!isfinite(f_rho_libxc[2 * i + 1])) nan_found = 1;
                if (nan_found) {
                    f_libxc[i] = 0.0;
                    f_rho_libxc[2 * i + 0] = 0.0;
                    f_rho_libxc[2 * i + 1] = 0.0;
                }
            }
            max_abs_f_error = 0.0;
            max_abs_f_rho_error = 0.0;
            max_rel_f_error = 0.0;
            max_rel_f_rho_error = 0.0;
            for (int point = 0; point < npoint; point++) {
                if (f_libxc[point] != 0.0) {
                    abs_error = fabs(f[point] - f_libxc[point]);
                    if (abs_error > max_abs_f_error) max_abs_f_error = abs_error;
                    rel_error = abs_error / fabs(f_libxc[point]);
                    if (rel_error > max_rel_f_error) max_rel_f_error = rel_error;
                }
                if (f_rho_libxc[2 * point + 0] != 0.0) {
                    abs_error = fabs(f_rho[2 * point + 0] - f_rho_libxc[2 * point + 0]);
                    if (abs_error > max_abs_f_rho_error) max_abs_f_rho_error = abs_error;
                    rel_error = abs_error / fabs(f_rho_libxc[2 * point + 0]);
                    if (rel_error > max_rel_f_rho_error) max_rel_f_rho_error = rel_error;
                }
                if (f_rho_libxc[2 * point + 1] != 0.0) {
                    abs_error = fabs(f_rho[2 * point + 1] - f_rho_libxc[2 * point + 1]);
                    if (abs_error > max_abs_f_rho_error) max_abs_f_rho_error = abs_error;
                    rel_error = abs_error / fabs(f_rho_libxc[2 * point + 1]);
                    if (rel_error > max_rel_f_rho_error) max_rel_f_rho_error = rel_error;
                }
            }
            printf("UKS LDA  Maximum error of f:       %8.2e (absolute): %8.2e (relative)\n", max_abs_f_error, max_rel_f_error);
            printf("UKS LDA  Maximum error of f_rho:   %8.2e (absolute): %8.2e (relative)\n", max_abs_f_rho_error, max_rel_f_rho_error);
            free(f_libxc);
            free(f_rho_libxc);
#endif
            for (int point = 0; point < npoint; point++) {
                double w = h_weights[point];
                h_Vxc_grid_a[point] = w * f_rho[2 * point + 0];
                h_Vxc_grid_b[point] = w * f_rho[2 * point + 1];
            }
            free(h_rho);

            break;
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_GGA:
            h_rho = (double*) malloc(npoint * 2 * sizeof(double));
            if (!h_rho) {
                fprintf(stderr, "Failed to allocate memory for h_rho\n");
                exit(EXIT_FAILURE);
            }
            h_gamma = (double*) malloc(npoint * 3 * sizeof(double));
            if (!h_gamma) {
                fprintf(stderr, "Failed to allocate memory for h_gamma\n");
                exit(EXIT_FAILURE);
            }
    
            rho_0_a = &h_rho_a_derivsT[0 * npoint];
            rho_x_a = &h_rho_a_derivsT[1 * npoint];
            rho_y_a = &h_rho_a_derivsT[2 * npoint];
            rho_z_a = &h_rho_a_derivsT[3 * npoint];

            rho_0_b = &h_rho_b_derivsT[0 * npoint];
            rho_x_b = &h_rho_b_derivsT[1 * npoint];
            rho_y_b = &h_rho_b_derivsT[2 * npoint];
            rho_z_b = &h_rho_b_derivsT[3 * npoint];
            for (int point = 0; point < npoint; point++) {
                h_rho[2 * point + 0] = rho_0_a[point];
                h_rho[2 * point + 1] = rho_0_b[point];
                h_gamma[3 * point + 0] = rho_x_a[point] * rho_x_a[point]
                                       + rho_y_a[point] * rho_y_a[point]
                                       + rho_z_a[point] * rho_z_a[point];
                h_gamma[3 * point + 1] = rho_x_a[point] * rho_x_b[point]
                                       + rho_y_a[point] * rho_y_b[point]
                                       + rho_z_a[point] * rho_z_b[point];
                h_gamma[3 * point + 2] = rho_x_b[point] * rho_x_b[point]
                                       + rho_y_b[point] * rho_y_b[point]
                                       + rho_z_b[point] * rho_z_b[point];
            }

            /*
             * This would not be needed if we were using real densities; it's just to catch
             * the cases where randomly generated density derivatives are unphysical.
             */
            sanitize_grid_uks_gga(npoint, h_rho, h_gamma);

            f_gamma = (double*) malloc(npoint * 3 * sizeof(double));
            if (!f_gamma) {
                fprintf(stderr, "Failed to allocate memory for f_gamma\n");
                exit(EXIT_FAILURE);
            }
    
            my_xc_gga_exc_vxc(
                npoint,
                h_rho,
                h_gamma,
                f,
                f_rho,
                f_gamma,
                1);

            /* Any other exchange and/or correlation functionals will be handled here */

            /* Clamp NaNs */
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f[i])) nan_found = 1;
                if (!isfinite(f_rho[2 * i + 0])) nan_found = 1;
                if (!isfinite(f_rho[2 * i + 1])) nan_found = 1;
                if (!isfinite(f_gamma[3 * i + 0])) nan_found = 1;
                if (!isfinite(f_gamma[3 * i + 1])) nan_found = 1;
                if (!isfinite(f_gamma[3 * i + 2])) nan_found = 1;
                if (nan_found) {
                    f[i] = 0.0;
                    f_rho[2 * i + 0] = 0.0;
                    f_rho[2 * i + 1] = 0.0;
                    f_gamma[3 * i + 0] = 0.0;
                    f_gamma[3 * i + 1] = 0.0;
                    f_gamma[3 * i + 2] = 0.0;
                }
            }
#if CHECK_AGAINST_LIBXC
            f_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_rho_libxc = (double*) malloc(npoint * 2 * sizeof(double));
            if (!f_rho_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_rho_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_gamma_libxc = (double*) malloc(npoint * 3 * sizeof(double));
            if (!f_gamma_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_gamma_libxc\n");
                exit(EXIT_FAILURE);
            }
            xc_func_init(&func, XC_GGA_X_B86, XC_POLARIZED);
            xc_gga_exc_vxc(
                &func,
                npoint,
                h_rho,
                h_gamma,
                f_libxc,
                f_rho_libxc,
                f_gamma_libxc);
            xc_func_end(&func);
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f_libxc[i])) nan_found = 1;
                if (!isfinite(f_rho_libxc[2 * i + 0])) nan_found = 1;
                if (!isfinite(f_rho_libxc[2 * i + 1])) nan_found = 1;
                if (!isfinite(f_gamma_libxc[3 * i + 0])) nan_found = 1;
                if (!isfinite(f_gamma_libxc[3 * i + 1])) nan_found = 1;
                if (!isfinite(f_gamma_libxc[3 * i + 2])) nan_found = 1;
                if (nan_found) {
                    f_libxc[i] = 0.0;
                    f_rho_libxc[2 * i + 0] = 0.0;
                    f_rho_libxc[2 * i + 1] = 0.0;
                    f_gamma_libxc[3 * i + 0] = 0.0;
                    f_gamma_libxc[3 * i + 1] = 0.0;
                    f_gamma_libxc[3 * i + 2] = 0.0;
                }
            }
            max_abs_f_error = 0.0;
            max_abs_f_rho_error = 0.0;
            max_rel_f_error = 0.0;
            max_rel_f_rho_error = 0.0;
            max_rel_f_gamma_error = 0.0;
            max_abs_f_gamma_error = 0.0;
            for (int point = 0; point < npoint; point++) {
                if (f_libxc[point] != 0.0) {
                    abs_error = fabs(f[point] - f_libxc[point]);
                    if (abs_error > max_abs_f_error) max_abs_f_error = abs_error;
                    rel_error = abs_error / fabs(f_libxc[point]);
                    if (rel_error > max_rel_f_error) max_rel_f_error = rel_error;
                }
                for (int sp = 0; sp < 2; sp++) {
                    if (f_rho_libxc[2 * point + sp] != 0.0) {
                        abs_error = fabs(f_rho[2 * point + sp] - f_rho_libxc[2 * point + sp]);
                        if (abs_error > max_abs_f_rho_error) max_abs_f_rho_error = abs_error;
                        rel_error = abs_error / fabs(f_rho_libxc[2 * point + sp]);
                        if (rel_error > max_rel_f_rho_error) max_rel_f_rho_error = rel_error;
                    }
                }
                for (int c = 0; c < 3; c++) {
                    if (f_gamma_libxc[3 * point + c] != 0.0) {
                        abs_error = fabs(f_gamma[3 * point + c] - f_gamma_libxc[3 * point + c]);
                        if (abs_error > max_abs_f_gamma_error) max_abs_f_gamma_error = abs_error;
                        rel_error = abs_error / fabs(f_gamma_libxc[3 * point + c]);
                        if (rel_error > max_rel_f_gamma_error) max_rel_f_gamma_error = rel_error;
                    }
                }
            }
            printf("UKS GGA  Maximum error of f:       %8.2e (absolute): %8.2e (relative)\n", max_abs_f_error, max_rel_f_error);
            printf("UKS GGA  Maximum error of f_rho:   %8.2e (absolute): %8.2e (relative)\n", max_abs_f_rho_error, max_rel_f_rho_error);
            printf("UKS GGA  Maximum error of f_gamma: %8.2e (absolute): %8.2e (relative)\n", max_abs_f_gamma_error, max_rel_f_gamma_error);
            free(f_libxc);
            free(f_rho_libxc);
            free(f_gamma_libxc);
#endif
            for (int point = 0; point < npoint; point++) {
                w = h_weights[point];
                f_gamma_aa = f_gamma[3 * point + 0];
                f_gamma_ab = f_gamma[3 * point + 1];
                f_gamma_bb = f_gamma[3 * point + 2];

                h_Vxc_grid_a[4 * point + 0] = w * f_rho[2 * point + 0];
                h_Vxc_grid_a[4 * point + 1] = w * (2 * f_gamma_aa * rho_x_a[point] + f_gamma_ab * rho_x_b[point]);
                h_Vxc_grid_a[4 * point + 2] = w * (2 * f_gamma_aa * rho_y_a[point] + f_gamma_ab * rho_y_b[point]);
                h_Vxc_grid_a[4 * point + 3] = w * (2 * f_gamma_aa * rho_z_a[point] + f_gamma_ab * rho_z_b[point]);

                h_Vxc_grid_b[4 * point + 0] = w * f_rho[2 * point + 1];
                h_Vxc_grid_b[4 * point + 1] = w * (2 * f_gamma_bb * rho_x_b[point] + f_gamma_ab * rho_x_a[point]);
                h_Vxc_grid_b[4 * point + 2] = w * (2 * f_gamma_bb * rho_y_b[point] + f_gamma_ab * rho_y_a[point]);
                h_Vxc_grid_b[4 * point + 3] = w * (2 * f_gamma_bb * rho_z_b[point] + f_gamma_ab * rho_z_a[point]);
            }
            free(h_rho);
            free(h_gamma);
            free(f_gamma);

            break;
        case CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_METAGGA:
            h_rho = (double*) malloc(npoint * 2 * sizeof(double));
            if (!h_rho) {
                fprintf(stderr, "Failed to allocate memory for h_rho\n");
                exit(EXIT_FAILURE);
            }
            h_gamma = (double*) malloc(npoint * 3 * sizeof(double));
            if (!h_gamma) {
                fprintf(stderr, "Failed to allocate memory for h_gamma\n");
                exit(EXIT_FAILURE);
            }
            h_tau = (double*) malloc(npoint * 2 * sizeof(double));
            if (!h_tau) {
                fprintf(stderr, "Failed to allocate memory for h_tau\n");
                exit(EXIT_FAILURE);
            }
            rho_0_a = &h_rho_a_derivsT[0 * npoint];
            rho_x_a = &h_rho_a_derivsT[1 * npoint];
            rho_y_a = &h_rho_a_derivsT[2 * npoint];
            rho_z_a = &h_rho_a_derivsT[3 * npoint];
            tau_0_a = &h_rho_a_derivsT[4 * npoint];

            rho_0_b = &h_rho_b_derivsT[0 * npoint];
            rho_x_b = &h_rho_b_derivsT[1 * npoint];
            rho_y_b = &h_rho_b_derivsT[2 * npoint];
            rho_z_b = &h_rho_b_derivsT[3 * npoint];
            tau_0_b = &h_rho_b_derivsT[4 * npoint];
            for (int point = 0; point < npoint; point++) {
                h_rho[2 * point + 0] = rho_0_a[point];
                h_rho[2 * point + 1] = rho_0_b[point];
                h_gamma[3 * point + 0] = rho_x_a[point] * rho_x_a[point]
                                       + rho_y_a[point] * rho_y_a[point]
                                       + rho_z_a[point] * rho_z_a[point];
                h_gamma[3 * point + 1] = rho_x_a[point] * rho_x_b[point]
                                       + rho_y_a[point] * rho_y_b[point]
                                       + rho_z_a[point] * rho_z_b[point];
                h_gamma[3 * point + 2] = rho_x_b[point] * rho_x_b[point]
                                       + rho_y_b[point] * rho_y_b[point]
                                       + rho_z_b[point] * rho_z_b[point];
                h_tau[2 * point + 0] = tau_0_a[point];
                h_tau[2 * point + 1] = tau_0_b[point];
            }

            /*
             * This would not be needed if we were using real densities; it's just to catch
             * the cases where randomly generated density derivatives are unphysical.
             */
            sanitize_grid_uks_mgga(npoint, h_rho, h_tau);

            f_gamma = (double*) malloc(npoint * 3 * sizeof(double));
            if (!f_gamma) {
                fprintf(stderr, "Failed to allocate memory for f_gamma\n");
                exit(EXIT_FAILURE);
            }
            f_tau = (double*) malloc(npoint * 2 * sizeof(double));
            if (!f_tau) {
                fprintf(stderr, "Failed to allocate memory for f_tau\n");
                exit(EXIT_FAILURE);
            }
            my_xc_mgga_exc_vxc(
                npoint,
                h_rho,
                h_gamma,
                h_tau,
                f,
                f_rho,
                f_gamma,
                f_tau,
                1);

            /* Any other exchange and/or correlation functionals will be handled here */

            /* Clamp NaNs */
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f[i])) nan_found = 1;
                if (!isfinite(f_rho[2 * i + 0])) nan_found = 1;
                if (!isfinite(f_rho[2 * i + 1])) nan_found = 1;
                if (!isfinite(f_gamma[3 * i + 0])) nan_found = 1;
                if (!isfinite(f_gamma[3 * i + 1])) nan_found = 1;
                if (!isfinite(f_gamma[3 * i + 2])) nan_found = 1;
                if (!isfinite(f_tau[2 * i + 0])) nan_found = 1;
                if (!isfinite(f_tau[2 * i + 1])) nan_found = 1;
                if (nan_found) {
                    f[i] = 0.0;
                    f_rho[2 * i + 0] = 0.0;
                    f_rho[2 * i + 1] = 0.0;
                    f_gamma[3 * i + 0] = 0.0;
                    f_gamma[3 * i + 1] = 0.0;
                    f_gamma[3 * i + 2] = 0.0;
                    f_tau[2 * i + 0] = 0.0;
                    f_tau[2 * i + 1] = 0.0;
                }
            }
#if CHECK_AGAINST_LIBXC
            f_libxc = (double*) malloc(npoint * sizeof(double));
            if (!f_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_rho_libxc = (double*) malloc(npoint * 2 * sizeof(double));
            if (!f_rho_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_rho_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_gamma_libxc = (double*) malloc(npoint * 3 * sizeof(double));
            if (!f_gamma_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_gamma_libxc\n");
                exit(EXIT_FAILURE);
            }
            f_tau_libxc = (double*) malloc(npoint * 2 * sizeof(double));
            if (!f_tau_libxc) {
                fprintf(stderr, "Failed to allocate memory for f_tau_libxc\n");
                exit(EXIT_FAILURE);
            }
            xc_func_init(&func, XC_MGGA_X_LTA, XC_POLARIZED);
            xc_mgga_exc_vxc(
                &func,
                npoint,
                h_rho,
                h_gamma,
                NULL,
                h_tau,
                f_libxc,
                f_rho_libxc,
                f_gamma_libxc,
                NULL,
                f_tau_libxc);
            xc_func_end(&func);
            for (int i = 0; i < npoint; i++) {
                int nan_found = 0;
                if (!isfinite(f_libxc[i])) nan_found = 1;
                if (!isfinite(f_rho_libxc[2 * i + 0])) nan_found = 1;
                if (!isfinite(f_rho_libxc[2 * i + 1])) nan_found = 1;
                if (!isfinite(f_gamma_libxc[3 * i + 0])) nan_found = 1;
                if (!isfinite(f_gamma_libxc[3 * i + 1])) nan_found = 1;
                if (!isfinite(f_gamma_libxc[3 * i + 2])) nan_found = 1;
                if (!isfinite(f_tau_libxc[2 * i + 0])) nan_found = 1;
                if (!isfinite(f_tau_libxc[2 * i + 1])) nan_found = 1;
                if (nan_found) {
                    f_libxc[i] = 0.0;
                    f_rho_libxc[2 * i + 0] = 0.0;
                    f_rho_libxc[2 * i + 1] = 0.0;
                    f_gamma_libxc[3 * i + 0] = 0.0;
                    f_gamma_libxc[3 * i + 1] = 0.0;
                    f_gamma_libxc[3 * i + 2] = 0.0;
                    f_tau_libxc[2 * i + 0] = 0.0;
                    f_tau_libxc[2 * i + 1] = 0.0;
                }
            }
            max_abs_f_error = 0.0;
            max_abs_f_rho_error = 0.0;
            max_rel_f_error = 0.0;
            max_rel_f_rho_error = 0.0;
            max_abs_f_gamma_error = 0.0;
            max_rel_f_gamma_error = 0.0;
            max_abs_f_tau_error = 0.0;
            max_rel_f_tau_error = 0.0;
            for (int point = 0; point < npoint; point++) {
                if (f_libxc[point] != 0.0) {
                    abs_error = fabs(f[point] - f_libxc[point]);
                    if (abs_error > max_abs_f_error) max_abs_f_error = abs_error;
                    rel_error = abs_error / fabs(f_libxc[point]);
                    if (rel_error > max_rel_f_error) max_rel_f_error = rel_error;
                }
                for (int sp = 0; sp < 2; sp++) {
                    if (f_rho_libxc[2 * point + sp] != 0.0) {
                        abs_error = fabs(f_rho[2*point+sp] - f_rho_libxc[2*point+sp]);
                        if (abs_error > max_abs_f_rho_error) max_abs_f_rho_error = abs_error;
                        rel_error = abs_error / fabs(f_rho_libxc[2*point+sp]);
                        if (rel_error > max_rel_f_rho_error) max_rel_f_rho_error = rel_error;
                    }
                }
                for (int c = 0; c < 3; c++) {
                    if (f_gamma_libxc[3 * point + c] != 0.0) {
                        abs_error = fabs(f_gamma[3*point+c] - f_gamma_libxc[3*point+c]);
                        if (abs_error > max_abs_f_gamma_error) max_abs_f_gamma_error = abs_error;
                        rel_error = abs_error / fabs(f_gamma_libxc[3*point+c]);
                        if (rel_error > max_rel_f_gamma_error) max_rel_f_gamma_error = rel_error;
                    }
                }
                for (int sp = 0; sp < 2; sp++) {
                    if (f_tau_libxc[2 * point + sp] != 0.0) {
                        abs_error = fabs(f_tau[2*point+sp] - f_tau_libxc[2*point+sp]);
                        if (abs_error > max_abs_f_tau_error) max_abs_f_tau_error = abs_error;
                        rel_error = abs_error / fabs(f_tau_libxc[2*point+sp]);
                        if (rel_error > max_rel_f_tau_error) max_rel_f_tau_error = rel_error;
                    }
                }
            }
            printf("UKS mGGA Maximum error of f:       %8.2e (absolute): %8.2e (relative)\n", max_abs_f_error, max_rel_f_error);
            printf("UKS mGGA Maximum error of f_rho:   %8.2e (absolute): %8.2e (relative)\n", max_abs_f_rho_error, max_rel_f_rho_error);
            printf("UKS mGGA Maximum error of f_gamma: %8.2e (absolute): %8.2e (relative)\n", max_abs_f_gamma_error, max_rel_f_gamma_error);
            printf("UKS mGGA Maximum error of f_tau:   %8.2e (absolute): %8.2e (relative)\n", max_abs_f_tau_error, max_rel_f_tau_error);
            free(f_libxc);
            free(f_rho_libxc);
            free(f_gamma_libxc);
            free(f_tau_libxc);
        #endif
            for (int point = 0; point < npoint; point++) {
                w = h_weights[point];
                f_gamma_aa = f_gamma[3 * point + 0];
                f_gamma_ab = f_gamma[3 * point + 1];
                f_gamma_bb = f_gamma[3 * point + 2];

                h_Vxc_grid_a[5 * point + 0] = w * f_rho[2 * point + 0];
                h_Vxc_grid_a[5 * point + 1] = w * (2 * f_gamma_aa * rho_x_a[point] + f_gamma_ab * rho_x_b[point]);
                h_Vxc_grid_a[5 * point + 2] = w * (2 * f_gamma_aa * rho_y_a[point] + f_gamma_ab * rho_y_b[point]);
                h_Vxc_grid_a[5 * point + 3] = w * (2 * f_gamma_aa * rho_z_a[point] + f_gamma_ab * rho_z_b[point]);
                h_Vxc_grid_a[5 * point + 4] = w * f_tau[2 * point + 0];

                h_Vxc_grid_b[5 * point + 0] = w * f_rho[2 * point + 1];
                h_Vxc_grid_b[5 * point + 1] = w * (2 * f_gamma_bb * rho_x_b[point] + f_gamma_ab * rho_x_a[point]);
                h_Vxc_grid_b[5 * point + 2] = w * (2 * f_gamma_bb * rho_y_b[point] + f_gamma_ab * rho_y_a[point]);
                h_Vxc_grid_b[5 * point + 3] = w * (2 * f_gamma_bb * rho_z_b[point] + f_gamma_ab * rho_z_a[point]);
                h_Vxc_grid_b[5 * point + 4] = w * f_tau[2 * point + 1];
            }

            free(h_rho);
            free(h_gamma);
            free(h_tau);
            free(f_gamma);
            free(f_tau);

            break;
        default:
            fprintf(stderr, "Invalid ansatz type\n");
            exit(EXIT_FAILURE);
    }

    /*
     * Build the bare per-point Exc density h_exc[point] = f * (ρ_α + ρ_β)
     * for use by cuestXCGridDerivativeCompute (no integration weights,
     * and no factor of 2 since both spins are summed explicitly).
     */
    double *h_exc_uks = (double*) malloc(npoint * sizeof(double));
    if (!h_exc_uks) {
        fprintf(stderr, "Failed to allocate memory for h_exc_uks\n");
        exit(EXIT_FAILURE);
    }
    Exc = 0.0;
    double integrated_density_a = 0.0;
    double integrated_density_b = 0.0;
    for (int point = 0; point < npoint; point++) {
        h_exc_uks[point] = f[point] * (rho_0_a[point] + rho_0_b[point]);
        Exc += h_weights[point] * h_exc_uks[point];
        integrated_density_a += h_weights[point] * rho_0_a[point];
        integrated_density_b += h_weights[point] * rho_0_b[point];
    }

    free(f);
    free(f_rho);
    free(h_weights);
    free(h_rho_a_derivsT);
    free(h_rho_b_derivsT);

    double *d_VxcGridA;
    if (cudaMalloc((void**)&d_VxcGridA, npoint * ncomponents * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(d_VxcGridA, h_Vxc_grid_a, npoint * ncomponents * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to device\n");
        exit(EXIT_FAILURE);
    }
    free(h_Vxc_grid_a);
    double *d_VxcGridB;
    if (cudaMalloc((void**)&d_VxcGridB, npoint * ncomponents * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(d_VxcGridB, h_Vxc_grid_b, npoint * ncomponents * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to device\n");
        exit(EXIT_FAILURE);
    }
    free(h_Vxc_grid_b);

    double *d_exc_uks;
    if (cudaMalloc((void**)&d_exc_uks, npoint * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(d_exc_uks, h_exc_uks, npoint * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to device\n");
        exit(EXIT_FAILURE);
    }
    free(h_exc_uks);

    /* Per-spin atomic gradient buffers and per-point grid-gradient
     * contributions.  The α and β contributions are accumulated
     * independently and combined on the host (atomic gradient) or with
     * a device-side add (grid gradient) below. */
    double *d_grad_atom_a, *d_grad_atom_b, *d_grad_grid_a, *d_grad_grid_b;
    if (cudaMalloc((void**)&d_grad_atom_a, numAtoms * 3 * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void**)&d_grad_atom_b, numAtoms * 3 * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void**)&d_grad_grid_a, npoint   * 3 * sizeof(double)) != cudaSuccess ||
        cudaMalloc((void**)&d_grad_grid_b, npoint   * 3 * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device buffer\n");
        exit(EXIT_FAILURE);
    }

    /* Compute the basis-center atomic gradient and the per-point
     * grid-gradient contribution for each spin. */
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCDERIVATIVECOMPUTE_PARAMETERS,
        &derivative_compute_parameters));
    checkCuestErrors(cuestXCDerivativeComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        ansatz,
        derivative_compute_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        noccA,
        d_CoccA,
        d_VxcGridA,
        d_grad_atom_a,
        d_grad_grid_a
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCDerivativeCompute(
        handle,
        xcIntPlan,
        ansatz,
        derivative_compute_parameters,
        variableBufferSize,
        temporaryWorkspace,
        noccA,
        d_CoccA,
        d_VxcGridA,
        d_grad_atom_a,
        d_grad_grid_a
    ));
    freeWorkspace(temporaryWorkspace);

    checkCuestErrors(cuestXCDerivativeComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        ansatz,
        derivative_compute_parameters,
        variableBufferSize,
        temporaryWorkspaceDescriptor,
        noccB,
        d_CoccB,
        d_VxcGridB,
        d_grad_atom_b,
        d_grad_grid_b
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCDerivativeCompute(
        handle,
        xcIntPlan,
        ansatz,
        derivative_compute_parameters,
        variableBufferSize,
        temporaryWorkspace,
        noccB,
        d_CoccB,
        d_VxcGridB,
        d_grad_atom_b,
        d_grad_grid_b
    ));
    freeWorkspace(temporaryWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCDERIVATIVECOMPUTE_PARAMETERS,
        derivative_compute_parameters));

    cudaFree(d_VxcGridA);
    cudaFree(d_VxcGridB);

    /* Sum the per-spin basis-center atomic gradient contributions on the host. */
    double *h_grad_uks = (double*) malloc(numAtoms * 3 * sizeof(double));
    double *h_grad_tmp = (double*) malloc(numAtoms * 3 * sizeof(double));
    if (!h_grad_uks || !h_grad_tmp) {
        fprintf(stderr, "Failed to allocate memory for h_grad_uks\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMemcpy(h_grad_uks, d_grad_atom_a, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_grad_tmp, d_grad_atom_b, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < numAtoms * 3; i++) {
        h_grad_uks[i] += h_grad_tmp[i];
    }

    /*
     * cuestXCGridDerivativeCompute consumes the *total* per-point grid
     * gradient (α + β) and produces the grid-center atomic gradient.
     * Stage through host memory to sum the two per-spin grid gradients
     * into d_grad_grid_a (a cuBLAS daxpy would also work, but is
     * avoided here to keep the example dependency-free).
     */
    {
        double *h_gg_a = (double*) malloc(npoint * 3 * sizeof(double));
        double *h_gg_b = (double*) malloc(npoint * 3 * sizeof(double));
        if (!h_gg_a || !h_gg_b) {
            fprintf(stderr, "Failed to allocate memory for grid gradient sum\n");
            exit(EXIT_FAILURE);
        }
        cudaMemcpy(h_gg_a, d_grad_grid_a, npoint * 3 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gg_b, d_grad_grid_b, npoint * 3 * sizeof(double), cudaMemcpyDeviceToHost);
        for (uint64_t i = 0; i < npoint * 3; i++) {
            h_gg_a[i] += h_gg_b[i];
        }
        cudaMemcpy(d_grad_grid_a, h_gg_a, npoint * 3 * sizeof(double), cudaMemcpyHostToDevice);
        free(h_gg_a);
        free(h_gg_b);
    }

    cuestXCGridDerivativeComputeParameters_t grid_derivative_compute_parameters_uks;
    checkCuestErrors(cuestParametersCreate(
        CUEST_XCGRIDDERIVATIVECOMPUTE_PARAMETERS,
        &grid_derivative_compute_parameters_uks));
    checkCuestErrors(cuestXCGridDerivativeComputeWorkspaceQuery(
        handle,
        xcIntPlan,
        grid_derivative_compute_parameters_uks,
        temporaryWorkspaceDescriptor,
        d_exc_uks,
        d_grad_grid_a,
        d_grad_atom_a
        ));
    temporaryWorkspace = allocateWorkspace(temporaryWorkspaceDescriptor);
    checkCuestErrors(cuestXCGridDerivativeCompute(
        handle,
        xcIntPlan,
        grid_derivative_compute_parameters_uks,
        temporaryWorkspace,
        d_exc_uks,
        d_grad_grid_a,
        d_grad_atom_a
    ));
    freeWorkspace(temporaryWorkspace);
    checkCuestErrors(cuestParametersDestroy(
        CUEST_XCGRIDDERIVATIVECOMPUTE_PARAMETERS,
        grid_derivative_compute_parameters_uks));

    /* Add the grid-center contribution to the running total. */
    if (cudaMemcpy(h_grad_tmp, d_grad_atom_a, numAtoms * 3 * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy device buffer to host\n");
        exit(EXIT_FAILURE);
    }
    for (uint64_t i = 0; i < numAtoms * 3; i++) {
        h_grad_uks[i] += h_grad_tmp[i];
    }

    printf("UKS XC gradient (Hartree/Bohr):\n");
    for (uint64_t a = 0; a < numAtoms; a++) {
        printf("  atom %3lu: %16.10f %16.10f %16.10f\n",
               (unsigned long) a,
               h_grad_uks[3*a+0], h_grad_uks[3*a+1], h_grad_uks[3*a+2]);
    }
    free(h_grad_uks);
    free(h_grad_tmp);

    cudaFree(d_grad_atom_a);
    cudaFree(d_grad_atom_b);
    cudaFree(d_grad_grid_a);
    cudaFree(d_grad_grid_b);
    cudaFree(d_exc_uks);

    /* Free Cocc matrices */
    if (cudaFree((void*) d_CoccA) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }
    if (cudaFree((void*) d_CoccB) != cudaSuccess) {
        fprintf(stderr, "cudaFree failed\n");
        exit(EXIT_FAILURE);
    }

    /*****************************************/
    /* Destroy cuEST handles and free memory */
    /*****************************************/

    free(variableBufferSize);

    /* The workspace descriptors are no longer needed. */
    free(persistentWorkspaceDescriptor);
    free(temporaryWorkspaceDescriptor);

    /* Destroy the XC integral plan handle. */
    checkCuestErrors(cuestXCIntPlanDestroy(xcIntPlan));
    freeWorkspace(persistentXCIntPlanWorkspace);

    /* Destroy the molecular grid handle. */
    checkCuestErrors(cuestMolecularGridDestroy(molecularGrid));
    freeWorkspace(persistentGridWorkspace);

    /* Destroy the AO basis handle. */
    checkCuestErrors(cuestAOBasisDestroy(basis));
    freeWorkspace(persistentBasisWorkspace);

    return;
}

int main(int argc, char **argv)
{
    /* Check that an xyz file and basis (gbs file) has been provided */
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <xyz_file_path> <gbs_file_path>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    const char* xyzFilePath = argv[1];
    const char* gbsFilePath = argv[2];

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

    /************************/
    /* cuEST AO Shell Setup */
    /************************/

    /* Use the AO shell helper to build the array of AO shells */
    AtomShellData_t* shellData = formAOShells(handle, xyzData, gbsFilePath, 1);

    // Do one run for each of the three ansätze.
    run_ansatz(handle, xyzData, shellData, CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_LDA);
    run_ansatz(handle, xyzData, shellData, CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_GGA);
    run_ansatz(handle, xyzData, shellData, CUEST_XCADVANCED_PARAMETERS_APPROXIMATION_METAGGA);

    /* Destroy the cuEST handle. */
    checkCuestErrors(cuestDestroy(handle));

    /* Free the XYZ data */
    freeParsedXYZFile(xyzData);

    /* Free all the shell data */
    freeAOShellData(shellData);
}
