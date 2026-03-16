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

#ifndef BASIS_DEFINITION_H
#define BASIS_DEFINITION_H

#include <helper_gbs_parser.h>

/* Hydrogen */
static uint64_t H_shell_types[] = {0, 0, 1};
static uint64_t H_prim_counts[] = {3, 1, 1};
static uint64_t H_prim_offsets[] = {0, 3, 4};
static double   H_exponents[] = {
    13.0107010, 1.9622572, 0.44453796,
    0.12194962,
    0.8000000
};
static double   H_coefficients[] = {
    0.019682158, 0.13796524, 0.47831935,
    1.0,
    1.0
};
static AtomBasisSet_t H_basis = {
    3, H_shell_types, H_prim_counts,
    H_prim_offsets, H_exponents, H_coefficients
};

/* Oxygen */
static uint64_t O_shell_types[] = {0, 0, 0, 1, 1, 2};
static uint64_t O_prim_counts[] = {5, 1, 1, 3, 1, 1};
static uint64_t O_prim_offsets[] = {0, 5, 6, 7, 10, 11};
static double   O_exponents[] = {
    2266.1767785, 340.87010191, 77.363135167, 21.479644940, 6.6589433124,
    0.80975975668,
    0.25530772234,
    17.721504317, 3.8635505440, 1.0480920883,
    0.27641544411,
    1.2
};
static double   O_coefficients[] = {
   -0.0053431809926, -0.039890039230, -0.17853911985, -0.46427684959, -0.44309745172,
    1.0,
    1.0,
    0.043394573193, 0.23094120765, 0.51375311064,
    1.0,
    1.0
};
static AtomBasisSet_t O_basis = {
    6, O_shell_types, O_prim_counts,
    O_prim_offsets, O_exponents, O_coefficients
};

#endif /* BASIS_DEFINITION_H */
