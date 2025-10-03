/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#if !defined(__cplusplus)
#error "This file can only by include in C++ file because it uses templates"
#endif

#include <cuComplex.h>

#include "utils.hxx"

/**
 * Generates a 7 point 3D laplace stencil with a matrix size of nx * nx * nx.
 * STEP=1 only writes to csr_offsets and nnz_ only. This allows an allocation
 * of csr_columns and csr_values afterwards, which will be written in STEP=2.
 */
template <typename DATA_TYPE, int STEP>
static void generate_3D_7p_laplace(const int nx, int *csr_offsets, int *csr_columns,
                                   DATA_TYPE *csr_values, int *nnz_) {
    const int n  = nx * nx * nx;
    const int ny = nx, nz = nx;

    if (STEP == 1) {
        csr_offsets[0] = 0;
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    int nnz_per_row = 7;
                    if ((z == 0) || (z == (nz - 1)))
                        nnz_per_row--;
                    if ((y == 0) || (y == (ny - 1)))
                        nnz_per_row--;
                    if ((x == 0) || (x == (nx - 1)))
                        nnz_per_row--;
                    csr_offsets[z * ny * nx + y * nx + x + 1] = nnz_per_row;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            csr_offsets[i + 1] += csr_offsets[i];
        }
        *nnz_ = csr_offsets[n] - csr_offsets[0];
    } else if (STEP == 2) {
        for (int z = 0; z < nz; z++) {
            for (int y = 0; y < ny; y++) {
                for (int x = 0; x < nx; x++) {
                    const int row       = z * ny * nx + y * nx + x;
                    int       nnz_total = csr_offsets[row];
                    DATA_TYPE val       = -1.0;
                    DATA_TYPE conj_val  = val;
                    DATA_TYPE diag      = 16.0;
                    if (z > 0) {
                        csr_columns[nnz_total] = (z - 1) * nx * ny + (y)*nx + x;
                        csr_values[nnz_total]  = val;
                        nnz_total++;
                    }
                    if (y > 0) {
                        csr_columns[nnz_total] = (z)*nx * ny + (y - 1) * nx + x;
                        csr_values[nnz_total]  = val;
                        nnz_total++;
                    }
                    if (x > 0) {
                        csr_columns[nnz_total] = (z)*nx * ny + (y)*nx + x - 1;
                        csr_values[nnz_total]  = val;
                        nnz_total++;
                    }
                    csr_columns[nnz_total] = (z)*nx * ny + (y)*nx + x;
                    csr_values[nnz_total]  = diag;
                    nnz_total++;
                    if (x < nx - 1) {
                        csr_columns[nnz_total] = (z)*nx * ny + (y)*nx + x + 1;
                        csr_values[nnz_total]  = conj_val;
                        nnz_total++;
                    }
                    if (y < ny - 1) {
                        csr_columns[nnz_total] = (z)*nx * ny + (y + 1) * nx + x;
                        csr_values[nnz_total]  = conj_val;
                        nnz_total++;
                    }
                    if (z < nz - 1) {
                        csr_columns[nnz_total] = (z + 1) * nx * ny + (y)*nx + x;
                        csr_values[nnz_total]  = conj_val;
                        nnz_total++;
                    }
                }
            }
        }
    }
}

/**
 * Generates a 7 point 3D laplace stencil with a matrix size of nx * nx * nx.
 * The value in n_ is used for nx, and will later be overwritten by the actual
 * matrix size
 * (which will be nx * nx * nx).
 * nnz_ will be set to the number of non-zeroes of the generated matrix.
 * *csr_offsets_, *csr_columns_, and *csr_values_ will be allocated with
 * malloc() and
 * filled with the 7-point stencil matrix in this function.
 * The ownership of these pointers is transfered to the caller, which means they
 * need to
 * be deallocated with free().
 */
template <typename DATA_TYPE>
static void laplace_3d_7p(int *n_, int *nnz_, int **csr_offsets_, int **csr_columns_,
                          DATA_TYPE **csr_values_, const int nrhs) {
    int nnz = 1, n = 1;
    int nx = *n_;

    n = nx * nx * nx;

    int *csr_offsets = (int *)malloc((n + 1) * sizeof(int));

    generate_3D_7p_laplace<DATA_TYPE, 1>(nx, csr_offsets, NULL, NULL, &nnz);

    int       *csr_columns = (int *)malloc(nnz * sizeof(int));
    DATA_TYPE *csr_values  = (DATA_TYPE *)malloc(nnz * sizeof(DATA_TYPE));

    generate_3D_7p_laplace<DATA_TYPE, 2>(nx, csr_offsets, csr_columns, csr_values, &nnz);

    *csr_offsets_ = csr_offsets;
    *csr_columns_ = csr_columns;
    *csr_values_  = csr_values;
    *n_           = n;
    *nnz_         = nnz;
}
