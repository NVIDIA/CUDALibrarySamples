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

#ifndef COMMON_HELPER_PCM
#define COMMON_HELPER_PCM

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This helper provides utility functions for constructing PCM cavity
 * parameters from molecular geometry data.
 */

/*
 * Returns 1.2 × Bondi atomic radius in bohr for the given uppercase element
 * symbol.  Based on Truhlar et al., J. Phys. Chem. A, 113, 5806-5812 (2009)
 * Table 12 and the CRC Handbook, 95th Edition, pp. 9-49.
 */
static double symbol_to_scaled_bondi_radius_bohr(const char *symbol)
{
    static const char *bondi_symbols_[] = {
        "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE",
        "NA", "MG", "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA",
        "SC", "TI", "V",  "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y",  "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
        "SB", "TE", "I",  "XE", "CS", "BA", "LA", "CE", "PR", "ND",
        "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
        "LU", "HF", "TA", "W",  "RE", "OS", "IR", "PT", "AU", "HG",
        "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
        "PA", "U",  "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM",
        "MD", "NO", "LR",
    };
    static const double bondi_radii_ang_[] = {
        1.10, 1.40, 1.81, 1.53, 1.92, 1.70, 1.55, 1.52, 1.47, 1.54,
        2.27, 1.73, 1.84, 2.10, 1.80, 1.80, 1.75, 1.88, 2.75, 2.31,
        2.15, 2.11, 2.07, 2.06, 2.05, 2.04, 2.00, 1.97, 1.96, 2.01,
        1.87, 2.11, 1.85, 1.90, 1.83, 2.02, 3.03, 2.49, 2.32, 2.23,
        2.18, 2.17, 2.16, 2.13, 2.10, 2.10, 2.11, 2.18, 1.93, 2.17,
        2.06, 2.06, 1.98, 2.16, 3.43, 2.68, 2.43, 2.42, 2.40, 2.39,
        2.38, 2.36, 2.35, 2.34, 2.33, 2.31, 2.30, 2.29, 2.27, 2.26,
        2.24, 2.23, 2.22, 2.18, 2.16, 2.16, 2.13, 2.13, 2.14, 2.23,
        1.96, 2.02, 2.07, 1.97, 2.02, 2.20, 3.48, 2.83, 2.47, 2.45,
        2.43, 2.41, 2.39, 2.43, 2.44, 2.45, 2.44, 2.45, 2.45, 2.45,
        2.46, 2.46, 2.46,
    };
    static const int n_ = (int)(sizeof(bondi_symbols_) / sizeof(bondi_symbols_[0]));
    for (int i = 0; i < n_; ++i) {
        if (strcmp(symbol, bondi_symbols_[i]) == 0)
            return 1.2 * bondi_radii_ang_[i] / 0.52917720859;
    }
    fprintf(stderr, "No Bondi radius defined for element %s\n", symbol);
    exit(EXIT_FAILURE);
}

/*
 * Returns the York-Karplus zeta exponent for a Lebedev grid of n angular
 * points.  J. Phys. Chem. A, 103, 11060-11079 (1999), Table 1.
 */
static double pcm_angular_points_to_zeta(uint64_t n)
{
    switch (n) {
        case   14: return 4.865;
        case   26: return 4.855;
        case   50: return 4.893;
        case  110: return 4.901;
        case  194: return 4.903;
        case  302: return 4.905;
        case  434: return 4.906;
        case  590: return 4.905;
        case  770: return 4.899;
        case  974: return 4.907;
        case 1202: return 4.907;
        default:
            fprintf(stderr, "No York-Karplus zeta for %llu angular points\n",
                    (unsigned long long) n);
            exit(EXIT_FAILURE);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* COMMON_HELPER_PCM */
