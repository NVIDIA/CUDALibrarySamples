/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include <cublas_v2.h>
#include <cusolverSp.h>
#include <cuComplex.h>

using ordinal_type = int;

#include "cudss.h"
#include "utils.hpp"

/// Creates CSR of B := A - shift*I assuming that A and B have
/// identical sparsity pattern.
static void apply_shift(const int m,
  const std::vector<cuDoubleComplex> &h_csrValA,
  std::vector<cuDoubleComplex> &h_csrValB,
  const std::vector<int> &diag_pos,
  const cuDoubleComplex shift) {
  std::copy(h_csrValA.begin(), h_csrValA.end(), h_csrValB.begin());

  // Apply shift to diagonal entries: B(i,i) := A(i,i) - shift
  for (int i = 0; i < m; i++) {
    h_csrValB[diag_pos[i]] = cuCsub(h_csrValB[diag_pos[i]], shift);
  }
}

/// Sets A := I (identity matrix).
static void eye(cublasHandle_t handle, int m, cuDoubleComplex *A, int lda) {
  CUDA_CHECK(cudaMemset(A, 0, lda * m * sizeof(cuDoubleComplex)));
  std::vector<cuDoubleComplex> h_ones(m, make_cuDoubleComplex(1.0, 0.0));
  cuDoubleComplex *d_ones = nullptr;
  size_t size_ones = m * sizeof(cuDoubleComplex);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ones), size_ones));
  CUDA_CHECK(cudaMemcpy(d_ones, h_ones.data(), size_ones, cudaMemcpyHostToDevice));
  CUBLAS_CHECK(cublasZcopy(handle, m, d_ones, 1, A, lda + 1));
  CUDA_CHECK(cudaFree(d_ones));
}

/// On exit, diag_pos[i] holds the index to the i-th diagonal
/// entry in the CSR values array.
static void find_diagonal_positions(
    const int m,
    const std::vector<int> &h_csrRowPtrA,
    const std::vector<int> &h_csrColIndA,
    std::vector<int> &diag_pos) {

  // Find indices where the diagonal entries are located.
  for (int i = 0; i < m; i++) {
    diag_pos[i] = -1;
    for (int j = h_csrRowPtrA[i]; j < h_csrRowPtrA[i+1]; j++) {
      if (h_csrColIndA[j] == i) {
        diag_pos[i] = j;
        break;
      }
    }
    if (diag_pos[i] == -1) {
      throw std::runtime_error("Error: Diagonal entry not found for row " + std::to_string(i));
    }
  }
}

/// Creates N equispaced points between start and end.
/// On exit, x contains N+1 points from start to end.
static void linspace(double start, double end, int N, std::vector<double> &x) {
  const double h = (end - start) / (double)N;
  for (int i = 0; i <= N; i++) {
    x[i] = start + i * h;
  }
}

/// Places quadrature points on the rectangular contour.
/// The contour goes from left_bottom_corner to right_upper_corner.
/// Nh is the number of horizontal points, Nv is the number of vertical points.
/// Total number of quadrature points is 2 * (Nh + Nv).
/// On exit, z contains the quadrature points.
static void place_quadrature_points(
    cuDoubleComplex left_bottom_corner,
    cuDoubleComplex right_upper_corner,
    int Nh,
    int Nv,
    std::vector<cuDoubleComplex> &z) {
  // Extract the corners of the box
  //
  //     (a1,b2) --------- (a2, b2)
  //        |                 |
  //     (a1,b1) --------- (a2, b1)
  //
  // where left_bottom_corner = (a1 + j*b1), and
  //       right_upper_corner = (a2 + j*b2)
  //
  const double a1 = left_bottom_corner.x;
  const double b1 = left_bottom_corner.y;
  const double a2 = right_upper_corner.x;
  const double b2 = right_upper_corner.y;

  std::vector<double> x(Nh + 1);
  std::vector<double> y(Nv + 1);
  linspace(a1, a2, Nh, x);
  linspace(b1, b2, Nv, y);

  // Quadrature points along (a1,b1) -> (a2,b1)
  for (int j = 0; j <= Nh; j++) {
    z[j] = make_cuDoubleComplex(x[j], b1);
  }

  // Quadrature points along (a2,b1) -> (a2,b2).
  for (int j = 1; j <= Nv; j++) {
    z[Nh + 1 + j - 1] = make_cuDoubleComplex(a2, y[j]);
  }

  // Quadrature points along (a2,b2) -> (a1,b2).
  for (int j = 1; j <= Nh; j++) {
    z[Nh + 1 + Nv + j - 1] = make_cuDoubleComplex(x[Nh - j], b2);
  }

  // Quadrature points along (a1,b2) -> (a1,b1).
  for (int j = 1; j < Nv; j++) {
    z[Nh + 1 + Nv + Nh + j - 1] = make_cuDoubleComplex(a1, y[Nv - j]);
  }
}

/// Returns the sum of the diagonal entries of the dense m-by-m matrix d_X.
static cuDoubleComplex compute_trace(
  cublasHandle_t cublasH,
  const int m, const cuDoubleComplex *d_X,
  cuDoubleComplex *d_diagX, std::vector<cuDoubleComplex> &h_diagX) {
  // Copy the diagonal of the X to the host.
  CUBLAS_CHECK(cublasZcopy(cublasH, m, d_X, m + 1, d_diagX, 1));
  CUDA_CHECK(cudaMemcpy(h_diagX.data(), d_diagX, m * sizeof(cuDoubleComplex),
                        cudaMemcpyDeviceToHost));
  // Sum the entries.
  cuDoubleComplex trace = make_cuDoubleComplex(0.0, 0.0);
  for (int j = 0; j < m; j++) {
    trace = cuCadd(trace, h_diagX[j]);
  }
  return trace;
}


static void approximate_eigenvalue_count_cudss(
  cusparseIndexBase_t base,
  const int m, const int nnzA,
  int *d_csrRowPtrA, int *d_csrColIndA, cuDoubleComplex *d_csrValA,       // CSR of A
  cuDoubleComplex left_bottom_corner, cuDoubleComplex right_upper_corner, // box
  int *num_eigs) {                                                        // on exit, the number
                                                                          // of eigenvalues
  // Create cuBLAS handle.
  cublasHandle_t cublasH = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublasH));

  // Initialize cuDSS.
  cudssHandle_t cudss = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  CUDSS_CHECK(cudssCreate(&cudss));
  CUDSS_CHECK(cudssConfigCreate(&config));
  CUDSS_CHECK(cudssDataCreate(cudss, &data));

  // Configure cuDSS to use CUDSS_ALG_1 for robustness. Alternatively,
  // CUDSS_ALG_DEFAULT yields a faster, but maybe less accurate
  // computation.
  cudssAlgType_t reordering_alg = CUDSS_ALG_1;
  CUDSS_CHECK(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG,
    &reordering_alg, sizeof(cudssAlgType_t)));

  // Set data type and index base.
  constexpr cudaDataType_t dtype = CUDA_C_64F;
  cudssIndexBase_t indexBase = (base == CUSPARSE_INDEX_BASE_ZERO) ?
    CUDSS_BASE_ZERO : CUDSS_BASE_ONE;

  // Host copy of A.
  std::vector<int> h_csrRowPtrA(m + 1);
  std::vector<int> h_csrColIndA(nnzA);
  std::vector<cuDoubleComplex> h_csrValA(nnzA);
  CUDA_CHECK(cudaMemcpy(h_csrRowPtrA.data(), d_csrRowPtrA,
    (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_csrColIndA.data(), d_csrColIndA,
    nnzA * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_csrValA.data(), d_csrValA,
    nnzA * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

  //
  // Step 1: Place the quadrature points on the contour.
  //
  // More points can increase the resolution.
  const int Nh = 100;
  const int Nv = 100;
  int N = 2 * (Nh + Nv); // total number of quadrature points
  std::vector<cuDoubleComplex> z(N); // the quadrature points
  place_quadrature_points(left_bottom_corner, right_upper_corner, Nh, Nv, z);

  //
  // Step 2: Prepare the computation of B_k := A - z[k]*I
  //         by finding the positions of the diagonal entries of A.
  //         Recall that we assume that all diagonal entries
  //         of A are non-zero.
  //
  int nnzB = nnzA; // assume that A has non-zeros on the diagonal
  std::vector<cuDoubleComplex> h_csrValB(nnzB);
  std::vector<int> diag_pos(m);
  find_diagonal_positions(m, h_csrRowPtrA, h_csrColIndA, diag_pos);

  //
  // Step 3: Compute the symbolic factorization of B_k := A - z[k]*I.
  //         The symbolic factorization is shared across all k.
  //
  cudssMatrix_t obj_B, obj_X, obj_I;
  CUDSS_CHECK(cudssMatrixCreateCsr(&obj_B, m, m, nnzA, d_csrRowPtrA, nullptr,
                                   d_csrColIndA, d_csrValA, CUDA_R_32I,
                                   dtype, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                                   indexBase));
  // Create right-hand side matrix as the identity matrix.
  cuDoubleComplex *d_I = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_I),
    m * m * sizeof(cuDoubleComplex)));
  eye(cublasH, m, d_I, m);
  CUDSS_CHECK(cudssMatrixCreateDn(&obj_I, m, m, m, d_I, dtype,
                                  CUDSS_LAYOUT_COL_MAJOR));
  // Create solution matrix.
  cuDoubleComplex *d_X = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X),
    m * m * sizeof(cuDoubleComplex)));
  CUDSS_CHECK(cudssMatrixCreateDn(&obj_X, m, m, m, d_X, dtype,
                                  CUDSS_LAYOUT_COL_MAJOR));
  // Analyze.
  CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_ANALYSIS, config, data,
                           obj_B, obj_X, obj_I));

  //
  // Step 4: Evaluate the integrand on every quadrature point.
  //
  std::vector<cuDoubleComplex> h_integrandValues(N);

  // Allocate device and host workspace for diagonal entries of solution X.
  cuDoubleComplex *d_diagX = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_diagX), m * sizeof(cuDoubleComplex)));
  std::vector<cuDoubleComplex> h_diagX(m);

  for (int k = 0; k < N; k++) {
    // Form B_k := A - z[k]*I. Since we assume that A has non-zeros on
    // the diagonal, we can apply the shift to the diagonal entries
    // directly and do not have to address any fill-in.
    apply_shift(m, h_csrValA, h_csrValB, diag_pos, z[k]);
    CUDA_CHECK(cudaMemcpy(d_csrValA, h_csrValB.data(),
                          nnzA * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice));

    // Factorize B_k := A - z[k]*I.
    // Alternatively, instead of computing a new factorization for
    // every point, refactorizations can be used for a faster computation,
    // though potentially at the expense of some accuracy. The impact on the
    // accuracy depends on the data and the quadrature points on the box.
    CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_FACTORIZATION, config, data,
                             obj_B, obj_X, obj_I));

    // Solve B_k * X = I for X using the computed factorization of B_k.
    // If successful, X = inv(B_k).
    CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_SOLVE, config, data,
                             obj_B, obj_X, obj_I));

    // Compute -trace(X).
    cuDoubleComplex trace = compute_trace(cublasH, m, d_X, d_diagX, h_diagX);
    h_integrandValues[k] = cuCmul(trace, make_cuDoubleComplex(-1.0, 0.0));
  }

  //
  // Step 5: Compute the integral using the trapezoidal rule.
  //         ∫f(z)dz ≈ (f(z[k]) + f(z[k+1]))/2 * (z[k+1] - z[k]).
  //
  cuDoubleComplex integral = make_cuDoubleComplex(0.0, 0.0);
  for (int k = 0; k < N-1; k++) {
    cuDoubleComplex avg = cuCadd(h_integrandValues[k], h_integrandValues[k+1]);
    avg = cuCmul(avg, make_cuDoubleComplex(0.5, 0.0));
    cuDoubleComplex dz = cuCsub(z[k+1], z[k]);
    integral = cuCadd(integral, cuCmul(avg, dz));
   }
  // Handle wraparound for closed contour.
  cuDoubleComplex avg = cuCadd(h_integrandValues[N-1], h_integrandValues[0]);
  avg = cuCmul(avg, make_cuDoubleComplex(0.5, 0.0));
  cuDoubleComplex dz = cuCsub(z[N-1], z[0]);
  integral = cuCadd(integral, cuCmul(avg, dz));

  // Number of eigenvalues = 1/(2*pi*i) * integral.
  *num_eigs = (int)round(integral.y / (2.0 * M_PI));

  // Free resources.
  CUDA_CHECK(cudaFree(d_diagX));
  CUDA_CHECK(cudaFree(d_I));
  CUDA_CHECK(cudaFree(d_X));
  CUDSS_CHECK(cudssMatrixDestroy(obj_I));
  CUDSS_CHECK(cudssMatrixDestroy(obj_B));
  CUDSS_CHECK(cudssMatrixDestroy(obj_X));
  CUDSS_CHECK(cudssDataDestroy(cudss, data));
  CUDSS_CHECK(cudssConfigDestroy(config));
  CUDSS_CHECK(cudssDestroy(cudss));
  CUBLAS_CHECK(cublasDestroy(cublasH));
}


int main(int argc, char *argv[]) {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  /*
   * A = |  1+i   1+i    0      0   |
   *     |   0    1-2i  -1-i    0   |
   *     |   0     0     1-2i  1-i  |
   *     | -2+2i   0      0     i   |
   * CSR of A is 0-based
   *
   * To keep the transition example short, A has nonzero
   * diagonal entries. If this is not the case, fill-in is created
   * when forming B := A - z*I. In this case, the CSR of B
   * can be constructed by either using the routine shift_diagonal
   * in csreigvsi2cuDSS_double.cpp or by adding explicit zero values
   * on the diagonal entries to the CSR of A.
   */
  const int m = 4;
  const int nnzA = 8;
  std::vector<int> csrRowPtrA = {0, 2, 4, 6, 8};
  std::vector<int> csrColIndA = {0, 1, 1, 2, 2, 3, 0, 3};
  std::vector<cuDoubleComplex> csrValA = { {1.0, 1.0}, {1.0, 1.0},
    {1.0, -2.0}, {-1.0, -1.0}, {1.0, -2.0}, {1.0, -1.0}, {-2.0, 2.0}, {0.0, 1.0} };
  cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;

  //
  // Device copies.
  //
  int *d_csrRowPtrA = nullptr;
  int *d_csrColIndA = nullptr;
  cuDoubleComplex *d_csrValA = nullptr;

  //
  // Allocate on device.
  //
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrValA), sizeof(cuDoubleComplex)*nnzA));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrColIndA), sizeof(int)*nnzA));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrRowPtrA), sizeof(int)*(m+1)));

  //
  // Copy A to device.
  //
  CUDA_CHECK(cudaMemcpyAsync(d_csrValA, csrValA.data(), sizeof(cuDoubleComplex)*nnzA,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_csrColIndA, csrColIndA.data(), sizeof(int)*nnzA,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_csrRowPtrA, csrRowPtrA.data(), sizeof(int)*(m+1),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  //
  // Define the box in which the number of eigenvalues are counted.
  //
  cuDoubleComplex left_bottom_corner = make_cuDoubleComplex(-1.0, -3.0);
  cuDoubleComplex right_upper_corner = make_cuDoubleComplex(1.0, 1.0);
  std::cout << "Box is defined by the bottom left corner [" 
    << left_bottom_corner.x  << " + " << left_bottom_corner.y << "i] "
    << "and the top right corner [" 
    << right_upper_corner.x << " + " << right_upper_corner.y << "i] " << std::endl;
  int num_eigs = 0; // Number of eigenvalues in the box

  //
  // Approximate eigenvalue count in box using cusolverSp.
  //
  {
    cusolverSpHandle_t cusolverH = nullptr;
    cusparseMatDescr_t descrA = nullptr;
    CUSOLVER_CHECK(cusolverSpCreate(&cusolverH));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, base));

    std::cout << "Compute eigenvalue count in box with cusolverSp\n";
    CUSOLVER_CHECK(cusolverSpZcsreigsHost(
      cusolverH, m, nnzA, descrA, csrValA.data(), csrRowPtrA.data(), csrColIndA.data(),
      left_bottom_corner, right_upper_corner, &num_eigs));
    std::cout << "Number of eigenvalues = " << num_eigs << std::endl;

    // Free resources.
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descrA));
    CUSOLVER_CHECK(cusolverSpDestroy(cusolverH));
  }

  //
  // Approximate eigenvalue count in box using cuDSS.
  //
  {
    std::cout << "Compute eigenvalue count in box with cuDSS\n";
    approximate_eigenvalue_count_cudss(
      base, m, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA,
      left_bottom_corner, right_upper_corner, &num_eigs);
    std::cout << "Number of eigenvalues = " << num_eigs << std::endl;
  }

  // Free resources.
  CUDA_CHECK(cudaFree(d_csrRowPtrA));
  CUDA_CHECK(cudaFree(d_csrColIndA));
  CUDA_CHECK(cudaFree(d_csrValA));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}