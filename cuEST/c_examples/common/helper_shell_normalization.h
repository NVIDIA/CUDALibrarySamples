/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef COMMON_SHELL_NORMALIZATION
#define COMMON_SHELL_NORMALIZATION

#include <stddef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef CUESTAPI

/**
 * Computes the normalized coefficients of a contracted Gaussian shell for a given angular momentum and primitives.
 *
 * @param[in]  L                      Angular momentum quantum number (must be less than 9).
 * @param[in]  numPrimitives          Number of primitives (size of arrays).
 * @param[in]  exponents              Pointer to array of primitive Gaussian exponents (all positive).
 * @param[in]  coefficients           Pointer to array of input primitive coefficients.
 * @param[in]  normalization          Desired normalization factor (must be positive and non-zero).
 * @param[out] coefficientsNormalized Pointer to array where normalized coefficients will be stored.
 *
 * @return    CUEST_STATUS_SUCCESS if coefficients were successfully normalized.
 *            CUEST_STATUS_UNKNOWN_ERROR if the input is invalid (nonpositive exponent, L >= 9, or normalization <= 0).
 *
 */
static cuestStatus_t computeNormalizedCoefficients(
    size_t L,
    size_t numPrimitives,
    const double* const exponents,
    const double* const coefficients,
    double normalization,
    double* const coefficientsNormalized)
{
    /* Gaussian exponents must be positive */
    for (size_t n=0; n<numPrimitives; n++) {
        if (exponents[n] <= 0.0) {
            return CUEST_STATUS_UNKNOWN_ERROR;
        }
    }
    /* Can't normalize L=9 or higher */
    if (L >= 9) {
        return CUEST_STATUS_UNKNOWN_ERROR;
    }
    /* Can't normalize to a negative number or zero */
    if (normalization <= 0.0) {
        return CUEST_STATUS_UNKNOWN_ERROR;
    }

    double pi32 = pow(M_PI, 1.5);
    double twoL = pow(2.0, (double) L);

    size_t dfact = 1;
    for (size_t l = 1; l <= L; l++) {
        dfact *= 2*l - 1;
    }

    for (size_t index = 0; index < numPrimitives; index++) {
        coefficientsNormalized[index] = sqrt(twoL / (pi32 * (double) dfact) * pow(2.0 * exponents[index], (double) L + 1.5)) * coefficients[index];
    }

    double Q = 0.0;
    for (size_t index1 = 0; index1 < numPrimitives; index1++) {
        for (size_t index2 = 0; index2 < numPrimitives; index2++) {
            Q += pow(sqrt(4.0 * exponents[index1] * exponents[index2]) / (exponents[index1] + exponents[index2]), (double) L + 1.5) * coefficients[index1] * coefficients[index2];
        }
    }
    Q = pow(Q, -0.5);
    Q *= sqrt(normalization);

    for (size_t index = 0; index < numPrimitives; index++) {
        coefficientsNormalized[index] *= Q;
    }

    return CUEST_STATUS_SUCCESS;
}

#endif

#ifdef __cplusplus
} 
#endif

#endif /* COMMON_SHELL_NORMALIZATION */
