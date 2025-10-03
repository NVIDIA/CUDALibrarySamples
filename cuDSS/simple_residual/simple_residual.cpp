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

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "cudss.h"
#include "laplace_generator.hxx"
#include "utils.hxx"

/*
    This example shows how to compute one of the most common
    accuracy estimates for solving a system of linear
    algebraic equations with a sparse matrix.
    The system is:
                        Ax = b,
    where:
        A is the sparse input matrix,
        b is the (dense) right-hand side vector (or a matrix),
        x is the (dense) solution vector (or a matrix).
    The accuracy is estimated by computing the relative residual
    norm
          || r || / (|| A ||_F * || x || + || b ||),
    where r = A * x - b is the residual.
    The host and device kernels are only for demonstration purposes
    and are not optimized for performance.
*/

#define CUDSS_EXAMPLE_FREE                                                               \
    do {                                                                                 \
        free(csr_offsets_h);                                                             \
        free(csr_columns_h);                                                             \
        free(csr_values_h);                                                              \
        free(x_values_h);                                                                \
        free(b_values_h);                                                                \
        cudaFree(csr_offsets_d);                                                         \
        cudaFree(csr_columns_d);                                                         \
        cudaFree(csr_values_d);                                                          \
        cudaFree(x_values_d);                                                            \
        cudaFree(b_values_d);                                                            \
    } while (0);

#define CUDA_CALL_AND_CHECK_AND_EXIT(call, msg)                                          \
    do {                                                                                 \
        cuda_error = call;                                                               \
        if (cuda_error != cudaSuccess) {                                                 \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n",  \
                   cuda_error);                                                          \
            CUDSS_EXAMPLE_FREE;                                                          \
            return -1;                                                                   \
        }                                                                                \
    } while (0);

#define CUDA_CALL_AND_CHECK(call, msg)                                                   \
    do {                                                                                 \
        cuda_error = call;                                                               \
        if (cuda_error != cudaSuccess) {                                                 \
            printf("Example FAILED: CUDA API returned error = %d, details: " #msg "\n",  \
                   cuda_error);                                                          \
        }                                                                                \
    } while (0);

#define CUDSS_CALL_AND_CHECK(call, status, msg)                                          \
    do {                                                                                 \
        status = call;                                                                   \
        if (status != CUDSS_STATUS_SUCCESS) {                                            \
            printf("Example FAILED: CUDSS call ended unsuccessfully with status = %d, "  \
                   "details: " #msg "\n",                                                \
                   status);                                                              \
            CUDSS_EXAMPLE_FREE;                                                          \
            return -2;                                                                   \
        }                                                                                \
    } while (0);

// Computes y = alpha * A * x + beta * y for A in CSR format
// and dense matrices x and y.
// A naive implementation that uses one thread per matrix row.
//
// To compute the residual r = A * x - b,
// we only need the case alpha = 1.0 and beta = -1.0.
// For a more performant implementation, consider using cuSPARSE,
// or writing a custom kernel.
// Another way to compute the residual is to set CUDSS_CONFIG_IR_N_STEPS to 1
// and call cudssExecute() with the phase CUDSS_PHASE_SOLVE_REFINEMENT.
template <int THREADS_PER_CTA, typename data_type, typename idx_type>
static __global__ void __launch_bounds__(THREADS_PER_CTA)
    simple_csr_spmm_ker(const int64_t m, const int64_t n, const int64_t num_rhs,
                        const idx_type *csr_offsets, const idx_type *csr_columns,
                        const data_type *csr_values, int idx_base, const double alpha,
                        int64_t ldx, const data_type *x, const double beta, int64_t ldy,
                        data_type *y) {
    // every index is computed as 0-base in this kernel
    const int64_t row = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    for (int64_t rhs_idx = 0; rhs_idx < num_rhs; ++rhs_idx) {
        data_type local_res = data_type{0.0};
        for (auto i = csr_offsets[row] - idx_base; i < csr_offsets[row + 1] - idx_base;
             ++i) {
            const auto mtx_val = csr_values[i];
            const auto col     = csr_columns[i] - idx_base;
            const auto x_val   = x[col + rhs_idx * ldx];
            local_res          = mtx_val * x_val + local_res;
        }
        const auto y_idx    = row + rhs_idx * ldy;
        const auto to_add_y = data_type{beta} * y[y_idx];
        y[y_idx]            = data_type{alpha} * local_res + to_add_y;
    }
}

// Computes y = y - A * x for a symmetric (lower- or upper-triangular) matrix A
// in CSR format.
// A naive sequential implementation.
// For brevity, this kernel is not performance-tuned.
template <typename data_type, typename idx_type>
static __global__ void __launch_bounds__(1)
    csr_symv_ker(int64_t n, int64_t nnz, idx_type *csr_offsets, idx_type *csr_columns,
                 data_type *csr_values, int lower, int index_base, int need_conj,
                 data_type *x, idx_type ldx, data_type *y, idx_type ldy) {

    const int64_t col = blockIdx.x;

    x += col * int64_t(ldx);
    y += col * int64_t(ldy);

    for (int64_t i = 0; i < n; i++) {
        data_type sum = data_type{0.0};
        for (idx_type j = csr_offsets[i] - index_base;
             j < csr_offsets[i + 1] - index_base; j++) {
            idx_type col_0based = csr_columns[j] - index_base;
            if (lower) {
                if (col_0based <= i) {
                    sum += csr_values[j] * x[col_0based];
                }
                if (col_0based < i) {
                    if (need_conj)
                        y[col_0based] -= csr_values[j] * x[i];
                    else
                        y[col_0based] -= cuConj(csr_values[j]) * x[i];
                }
            } else {
                if (col_0based >= i) {
                    sum += csr_values[j] * x[col_0based];
                }
                if (col_0based > i) {
                    data_type reg_a = csr_values[j];
                    if (need_conj)
                        reg_a = cuConj(reg_a);
                    y[col_0based] -= reg_a * x[i];
                }
            }
        }
        y[i] -= sum;
    }
}

template <typename data_type, typename idx_type>
static __global__ void __launch_bounds__(1)
    csr_frobenius_ker(int64_t n, int64_t nnz, const idx_type *csr_offsets,
                      const idx_type *csr_columns, const data_type *csr_values,
                      cudssMatrixType_t mtype, cudssMatrixViewType_t mview,
                      cudssIndexBase_t base, double *norm) {
    const int64_t row = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;

    if (row >= n) {
        return;
    }

    double local_sum_sq = 0.0;
    for (idx_type i = csr_offsets[row] - base; i < csr_offsets[row + 1] - base; i++) {
        const idx_type col = csr_columns[i] - base;
        if (col == row) {
            local_sum_sq += csr_values[i] * csr_values[i];
        } else if (mtype != CUDSS_MTYPE_GENERAL && mview == CUDSS_MVIEW_LOWER) {
            if (col < row) {
                local_sum_sq += 2.0 * csr_values[i] * csr_values[i];
            }
        } else if (mtype != CUDSS_MTYPE_GENERAL && mview == CUDSS_MVIEW_UPPER) {
            if (col > row) {
                local_sum_sq += 2.0 * csr_values[i] * csr_values[i];
            }
        } else {
            local_sum_sq += csr_values[i] * csr_values[i];
        }
    }
    atomicAdd(norm, local_sum_sq);
}

// This host function demonstrates how the relative residual norm can be
// computed
// for a given sparse matrix A (in CSR format), right-hand side b, and solution
// x.
// The relative residual norm is computed for each column of b and x separately.
// The relative residual norm is computed as:
//              || r || / (|| A ||_F * || x || + || b ||)
// where r = A * x - b is the residual.
// The kernels sued in this function are not optimized for performance.
// For a more performant implementation, consider using cuSPARSE and cuBLAS
// routines.
template <typename data_type, typename idx_type>
static bool relative_residual(const int64_t n, const int64_t nrhs, const int64_t nnz,
                              idx_type *csr_offsets, idx_type *csr_columns,
                              data_type *csr_values, cudssMatrixViewType_t mview,
                              cudssMatrixType_t mtype, cudssIndexBase_t base,
                              data_type *x, const int64_t ldx, data_type *b,
                              const int64_t ldb, cudaDataType compute_type,
                              double tol_max, double tol_l2, double &output_relres_max,
                              double &output_relres_l2) {

    cudaError_t cuda_error = cudaSuccess;

    bool is_valid = true;

    // We recommend to check that there are no device errors (incl. asynchronous)
    // before the residual computation starts.
    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    CUDA_CALL_AND_CHECK(cudaGetLastError(),
                        "cudaGetLastError at the start of relative_residual()");

    data_type *r_h = NULL, *x_h = NULL;
    r_h = (data_type *)malloc(nrhs * ldb * sizeof(data_type));
    x_h = (data_type *)malloc(nrhs * ldx * sizeof(data_type));
    if (!r_h || !x_h) {
        printf("Error: Host memory allocation failed for r_h and/or x_h in "
               "relative_residual()\n");
        if (r_h)
            free(r_h);
        if (x_h)
            free(x_h);
        return false;
    }

    data_type *r = NULL;
    CUDA_CALL_AND_CHECK(cudaMalloc(&r, nrhs * ldb * sizeof(data_type)),
                        "cudaMalloc for r");

    data_type *fro_norm = NULL;
    CUDA_CALL_AND_CHECK(cudaMalloc(&fro_norm, 1 * sizeof(double)),
                        "cudaMalloc for the fro_norm");
    CUDA_CALL_AND_CHECK(cudaMemset(fro_norm, 0, 1 * sizeof(double)),
                        "cudaMemset for the fro_norm");

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(r, b, nrhs * ldb * sizeof(data_type), cudaMemcpyDeviceToDevice),
        "cudaMemcpy D2D for r");

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(r_h, r, nrhs * ldb * sizeof(data_type), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H for r_h");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(x_h, x, nrhs * ldx * sizeof(data_type), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H for x_h");
    if (cuda_error != cudaSuccess) {
        printf("Error: cudaMemcpy D2H for r_h and/or x_h in relative_residual()\n");
        if (r_h)
            free(r_h);
        if (x_h)
            free(x_h);
        cudaFree(r);
        return false;
    }

    double *bnorms_max = NULL, *bnorms_l2 = NULL;
    bnorms_max = (double *)malloc(nrhs * sizeof(double));
    bnorms_l2  = (double *)malloc(nrhs * sizeof(double));

    double fro_norm_h = 0.0;

    if (!bnorms_max || !bnorms_l2) {
        printf("Error: Host memory allocation failed for bnorms arrays in "
               "relative_residual()\n");
        if (r_h)
            free(r_h);
        if (x_h)
            free(x_h);
        if (bnorms_max)
            free(bnorms_max);
        if (bnorms_l2)
            free(bnorms_l2);
        cudaFree(r);
        cudaFree(fro_norm);
        return false;
    }

    // Right now, r_h contains the original righthand side b on the host
    data_type *b_h = r_h;
    // Here we compute the max and l2 norms of the righthand side b on the host
    // for each column of b.
    // Alternatively, one could call cublas<t>nrm2()
    // to compute the l2 norm and cub::DeviceReduce to compute the max norm
    // on the device.
    for (int64_t j = 0; j < nrhs; j++) {
        double bnorm_max = 0.0, bnorm_l2sq = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double value = cuAbs(b_h[j * ldb + i]);
            if (value > bnorm_max)
                bnorm_max = value;
            bnorm_l2sq += value * value;
        }
        bnorms_max[j] = bnorm_max;
        bnorms_l2[j]  = sqrt(bnorm_l2sq / n);
    }

    csr_frobenius_ker<data_type, idx_type><<<n, 1>>>(
        n, nnz, csr_offsets, csr_columns, csr_values, mtype, mview, base, fro_norm);

    CUDA_CALL_AND_CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(&fro_norm_h, fro_norm, 1 * sizeof(double), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H for fro_norm_h");
    fro_norm_h = sqrt(fro_norm_h);
    printf("info: fro_norm_h (Frobenius norm of matrix A) = %3.3e\n", fro_norm_h);

    const int need_conj = (mtype == CUDSS_MTYPE_HERMITIAN || mtype == CUDSS_MTYPE_HPD);

    if (mtype == CUDSS_MTYPE_GENERAL) {
        constexpr int      THREADS_PER_CTA{128};
        const unsigned int grid = div_up(n, THREADS_PER_CTA);
        simple_csr_spmm_ker<THREADS_PER_CTA>
            <<<grid, THREADS_PER_CTA>>>(n, n, nrhs, csr_offsets, csr_columns, csr_values,
                                        base == CUDSS_BASE_ONE, -1., ldx, x, 1., ldb, r);
    } else if (mview == CUDSS_MVIEW_LOWER || mview == CUDSS_MVIEW_UPPER ||
               mview == CUDSS_MVIEW_FULL) {
        csr_symv_ker<data_type, idx_type><<<nrhs, 1>>>(
            n, nnz, csr_offsets, csr_columns, csr_values, (mview == CUDSS_MVIEW_LOWER),
            base != CUDSS_BASE_ZERO, need_conj, x, ldx, r, ldb);
    } else {
        printf("Error: unsupported case in relative_residual(): mtype = %d mview = "
               "%d\n",
               mtype, mview);
        is_valid = false;
    }

    CUDA_CALL_AND_CHECK(
        cudaMemcpy(r_h, r, nrhs * ldb * sizeof(data_type), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H for r_h");
    CUDA_CALL_AND_CHECK(
        cudaMemcpy(x_h, x, nrhs * ldx * sizeof(data_type), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H for x_h");

    // We compute a relative residual norm for each of the columns in the
    // righthand side (and solution) separately.
    for (int64_t j = 0; j < nrhs; j++) {
        int    has_nan    = 0;
        double max_err    = 0.0;
        double max_norm_x = 0.0, l2_norm_x = 0.0;
        double max_norm_res = 0.0, l2_norm_res = 0.0;
        for (int64_t i = 0; i < n; i++) {
            if (cuReal(x_h[j * ldx + i]) != cuReal(x_h[j * ldx + i]) ||
                cuImag(x_h[j * ldx + i]) != cuImag(x_h[j * ldx + i]))
                has_nan = 1;
            if (cuAbs(r_h[j * ldb + i]) > max_err)
                max_err = cuAbs(r_h[j * ldb + i]);
            if (cuAbs(x_h[j * ldx + i]) > max_norm_x)
                max_norm_x = cuAbs(x_h[j * ldx + i]);
            l2_norm_x   += cuAbs(x_h[j * ldx + i]) * cuAbs(x_h[j * ldx + i]);
            l2_norm_res += cuAbs(r_h[j * ldb + i]) * cuAbs(r_h[j * ldb + i]);
        }
        l2_norm_res  = sqrt(l2_norm_res / n);
        max_norm_res = max_err;

        if (has_nan) {
            printf("Error: nans detected\n");
            is_valid = false;
        }

        // The accuracy is estimated by computing the relative residual norm
        //              || r || / (|| A ||_F * || x || + || b ||)
        // Note: one of other commonly used alternatives is omitting the first term
        // in the denominator:
        //              || r || / || b ||
        // but this formula is less robust from the numerical linear algebra
        // standpoint.
        double residual_relative_norm_max =
            max_norm_res / (fro_norm_h * max_norm_x + bnorms_max[j]);
        double residual_relative_norm_l2 =
            l2_norm_res / (fro_norm_h * l2_norm_x + bnorms_l2[j]);

        printf("info: relative residual norms\n");
        printf("info: col = %ld residual_relative_norm_max = %3.3e "
               "residual_relative_norm_l2 = %3.3e\n",
               j, residual_relative_norm_max, residual_relative_norm_l2);
        printf("info: col = %ld details: max_norm_res = %3.3e max_norm_b = %3.3e "
               "fro_norm_A = %3.3e max_norm_x = %3.3e\n",
               j, max_norm_res, bnorms_max[j], fro_norm_h, max_norm_x);
        printf("info: col = %ld details: l2_norm_res = %3.3e l2_norm_b = %3.3e "
               "fro_norm_A = %3.3e l2_norm_x = %3.3e\n",
               j, l2_norm_res, bnorms_l2[j], fro_norm_h, l2_norm_x);

        // Optionally, we compare the residual relative norm with the given
        // tolerance
        if (tol_max != -1.0 && residual_relative_norm_max > tol_max) {
            printf("Error: relative max norm of the residual is too high: col = %ld "
                   "max_res "
                   "= %3.3e max_b = %3.3e relres_max = %3.3e (tol_max = %3.3e) \n",
                   j, max_norm_res, bnorms_max[j], residual_relative_norm_max, tol_max);
            is_valid = false;
        }
        if (tol_l2 != -1.0 && residual_relative_norm_l2 > tol_l2) {
            printf("Error: relative l2 norm of the residual is too high: col = %ld "
                   "l2_res = %3.3e l2_b = %3.3e relres_l2 = %3.3e (tol_l2 = %3.3e) \n",
                   j, l2_norm_res, bnorms_l2[j], residual_relative_norm_l2, tol_l2);
            is_valid = false;
        }

        if (residual_relative_norm_max > output_relres_max)
            output_relres_max = residual_relative_norm_max;
        if (residual_relative_norm_l2 > output_relres_l2)
            output_relres_l2 = residual_relative_norm_l2;
    }

    free(bnorms_max);
    free(bnorms_l2);

    free(r_h);
    free(x_h);

    cudaFree(r);
    cudaFree(fro_norm);

    if (cuda_error != cudaSuccess) {
        printf("Error: cuda_error = %d != cudaSuccess in relative_residual()\n",
               cuda_error);
        return false;
    } else
        return is_valid;
}

int main(int argc, char *argv[]) {
    printf("---------------------------------------------------------\n");
    printf("cuDSS example: solving a real linear system with a\n"
           "Laplace matrix, estimating accuracy of the solution\n");
    printf("---------------------------------------------------------\n");
    cudaError_t   cuda_error = cudaSuccess;
    cudssStatus_t status     = CUDSS_STATUS_SUCCESS;

    int nx = 10;
    int n  = nx; // will be changed in the laplace_3d_7p() function
    int nnz;
    int nrhs = 2;

    int    *csr_offsets_h = NULL;
    int    *csr_columns_h = NULL;
    double *csr_values_h  = NULL;
    double *x_values_h = NULL, *b_values_h = NULL;

    int    *csr_offsets_d = NULL;
    int    *csr_columns_d = NULL;
    double *csr_values_d  = NULL;
    double *x_values_d = NULL, *b_values_d = NULL;

    laplace_3d_7p<double>(&n, &nnz, &csr_offsets_h, &csr_columns_h, &csr_values_h, nrhs);

    printf("Info: n = %d [nx = %d], nnz = %d\n", n, nx, nnz);

    int ldx = n + 5;
    int ldb = n + n / 2;
    printf("Info: ldx = %d, ldb = %d\n", ldx, ldb);

    /* Allocate host memory for the sparse input matrix A,
       right-hand side x and solution b*/

    x_values_h = (double *)malloc(nrhs * ldx * sizeof(double));
    b_values_h = (double *)malloc(nrhs * ldb * sizeof(double));

    if (!csr_offsets_h || !csr_columns_h || !csr_values_h || !x_values_h || !b_values_h) {
        printf("Error: host memory allocation failed\n");
        return -1;
    }

    /* Note: In this sample we show how to compute the relative residual norm.
       A reasonable alternative  could be to take a vector x_ex to be the
       exact solution, and compute the so called manufactured righthand
       side as b = A * x_ex (by doing a simple sparse matrix - dense
       vector product), solve the system with righthand side set to b
       and compare the solution with x_ex.
     */
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            b_values_h[j * ldb + i] = 1. * (j + 1.);
        }
    }

    /* Allocate device memory for A, x and b */
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                                 "cudaMalloc for csr_offsets");
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                                 "cudaMalloc for csr_columns");
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMalloc(&csr_values_d, nnz * sizeof(double)),
                                 "cudaMalloc for csr_values");
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMalloc(&b_values_d, nrhs * ldb * sizeof(double)),
                                 "cudaMalloc for b_values");
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMalloc(&x_values_d, nrhs * ldx * sizeof(double)),
                                 "cudaMalloc for x_values");

    /* Copy host memory to device for A and b */
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMemcpy(csr_offsets_d, csr_offsets_h,
                                            (n + 1) * sizeof(int),
                                            cudaMemcpyHostToDevice),
                                 "cudaMemcpy for csr_offsets");
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMemcpy(csr_columns_d, csr_columns_h,
                                            nnz * sizeof(int), cudaMemcpyHostToDevice),
                                 "cudaMemcpy for csr_columns");
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMemcpy(csr_values_d, csr_values_h,
                                            nnz * sizeof(double), cudaMemcpyHostToDevice),
                                 "cudaMemcpy for csr_values");
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMemcpy(b_values_d, b_values_h,
                                            nrhs * ldb * sizeof(double),
                                            cudaMemcpyHostToDevice),
                                 "cudaMemcpy for b_values");

    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaStreamCreate(&stream), "cudaStreamCreate");

    /* Creating the cuDSS library handle */
    cudssHandle_t handle;

    CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

    /* (optional) Setting the custom stream for the library handle */
    CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status, "cudssSetStream");

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t   solverData;

    CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status, "cudssConfigCreate");
    CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status, "cudssDataCreate");

    /* Create matrix objects for the right-hand side b and solution x (as dense
     * matrices).
     */
    cudssMatrix_t x, b;

    int64_t nrows = n, ncols = n;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b, ncols, nrhs, ldb, b_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x, nrows, nrhs, ldx, x_values_d, CUDA_R_64F,
                                             CUDSS_LAYOUT_COL_MAJOR),
                         status, "cudssMatrixCreateDn for x");

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    // cudssMatrixType_t mtype     = CUDSS_MTYPE_SPD;
    // cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
    cudssMatrixType_t     mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;
    cudssIndexBase_t      base  = CUDSS_BASE_ZERO;
    CUDSS_CALL_AND_CHECK(cudssMatrixCreateCsr(&A, nrows, ncols, nnz, csr_offsets_d, NULL,
                                              csr_columns_d, csr_values_d, CUDA_R_32I,
                                              CUDA_R_64F, mtype, mview, base),
                         status, "cudssMatrixCreateCsr");

    /* Symbolic factorization */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b),
        status, "cudssExecute for analysis");

    /* Factorization */
    CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                      solverData, A, x, b),
                         status, "cudssExecute for factor");

    /* Solving */
    CUDSS_CALL_AND_CHECK(
        cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b),
        status, "cudssExecute for solve");

    /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A), status, "cudssMatrixDestroy for A");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b), status, "cudssMatrixDestroy for b");
    CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x), status, "cudssMatrixDestroy for x");
    CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status,
                         "cudssDataDestroy");
    CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status, "cudssConfigDestroy");
    CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

    CUDA_CALL_AND_CHECK_AND_EXIT(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    /* Print the solution and compare against the exact solution */
    CUDA_CALL_AND_CHECK_AND_EXIT(cudaMemcpy(x_values_h, x_values_d,
                                            nrhs * n * sizeof(double),
                                            cudaMemcpyDeviceToHost),
                                 "cudaMemcpy for x_values");

    /* Now, usually, in most applications the exact solution is not known.
       In this case, we recommend compute the relative residual norm
          || r || / (|| A ||_F * || x || + || b ||),
        where r = A * x - b is the residual.
     */

    int    passed  = 1;
    double tol_max = -1.0, tol_l2 = -1.0;
    double output_relres_max = 0.0, output_relres_l2 = 0.0;
    bool   is_valid =
        relative_residual(n, nrhs, nnz, csr_offsets_d, csr_columns_d, csr_values_d, mview,
                          mtype, base, x_values_d, ldx, b_values_d, ldb, CUDA_R_64F,
                          tol_max, tol_l2, output_relres_max, output_relres_l2);
    if (!is_valid) {
        printf("Example FAILED: relative_residual() returned status = false\n");
        passed = 0;
    }

    /* Release the data allocated on the user side */

    CUDSS_EXAMPLE_FREE;

    if (status == CUDSS_STATUS_SUCCESS && passed) {
        printf("Example PASSED\n");
        return 0;
    } else {
        printf("Example FAILED\n");
        return -1;
    }
}
