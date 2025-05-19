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

#include <cublas_v2.h>
#include <cusparse.h>

using ordinal_type = int;

#include "cudss.h"
#include "utils.hpp"


static void normalize_vector(cublasHandle_t cublasH, int m, double *d_x) {
  double norm = 0.0;
  CUBLAS_CHECK(cublasDnrm2(cublasH, m, d_x, 1, &norm));
  if (norm != 0.0) {
    const double alpha = 1.0 / norm;
    CUBLAS_CHECK(cublasDscal(cublasH, m, &alpha, d_x, 1));
  }
}

/// Creates CSR of B, where B = A - mu * I.
static void shift_diagonal(cusparseIndexBase_t base,
  std::vector<int> &csrRowPtrA,
  std::vector<int> &csrColIndA, std::vector<double> &csrValA,
  double mu,
  std::vector<int> &csrRowPtrB, std::vector<int> &csrColIndB,
  std::vector<double> &csrValB, int &nnzB) {
  int nnzA = csrValA.size();
  int m = csrRowPtrA.size()-1;

  int offset = (base == CUSPARSE_INDEX_BASE_ZERO) ? 0 : 1;

  // Compute how much fill-in is created by applying the shift.
  int fill = csrRowPtrA.size()-1;
  int j = 0;
  for (int row = 0; row < csrRowPtrA.size()-1; row++) {
    int nz = csrRowPtrA[row+1]-csrRowPtrA[row];
    for (int idx = 0; idx < nz; idx++) {
      int col = csrColIndA[j];
      if (row + offset == col) {
        fill--;
      }
      j++;
    }
  }

  // Resize.
  nnzB = nnzA + fill;
  csrRowPtrB.resize(csrRowPtrA.size());
  csrColIndB.resize(nnzB);
  csrValB.resize(nnzB);

  // Convert to matrix market format and apply shift to diagonal.
  using ijx_type = coo_t<double>;
  std::vector<ijx_type> mmA;
  csr_to_coo<double>(m, nnzA, csrRowPtrA, csrColIndA, csrValA, mmA);
  for (int i = 0; i < m; i++) {
    mmA.push_back(ijx_type(i+offset, i+offset, -mu));
  }
  std::sort(mmA.begin(), mmA.end(), std::less<ijx_type>());

  // Convert B.
  coo_to_csr(m, nnzB, mmA, csrRowPtrB, csrColIndB, csrValB, offset == 0);
}

static void approximate_eigenpair_cudss(
  cusparseIndexBase_t base,
  const int m, const int nnzA,
  int *d_csrRowPtrA, int *d_csrColIndA, double *d_csrValA, // CSR of A
  double mu0, double *d_x0,                                // initial guess
  int maxite, double tol,                                  // stopping criteria
  double *d_mu, double *d_x,                               // on exit, the computed eigenpair
  cudaStream_t stream) {
  double one = 1.0;
  double zero = 0.0;
  constexpr cudaDataType_t dtype = CUDA_R_64F;

  double mu = mu0;

  // Init cuBlas (norm computation).
  cublasHandle_t cublasH = nullptr;
  CUBLAS_CHECK(cublasCreate(&cublasH));

  // Init cuSparse (sparse matrix-vector multiplication).
  cusparseHandle_t cusparseH = nullptr;
  CUSPARSE_CHECK(cusparseCreate(&cusparseH));

  // Initialize cuDSS.
  cudssHandle_t cudss = nullptr;
  cudssConfig_t config = nullptr;
  cudssData_t data = nullptr;
  CUDSS_CHECK(cudssCreate(&cudss));
  CUDSS_CHECK(cudssConfigCreate(&config));
  CUDSS_CHECK(cudssDataCreate(cudss, &data));

  // Host copy of A.
  std::vector<int> csrRowPtrA(m+1);
  std::vector<int> csrColIndA(nnzA);
  std::vector<double> csrValA(nnzA);

  //
  // CSR of B := A - mu0 * I
  // Initialization is deferred until fill-in is determined.
  //
  int nnzB = -1;
  std::vector<int> csrRowPtrB;
  std::vector<int> csrColIndB;
  std::vector<double> csrValB;
  // Device copies.
  int *d_csrRowPtrB = nullptr;
  int *d_csrColIndB = nullptr;
  double *d_csrValB = nullptr;

  //
  // Prepare computation of y = A * x through cusparseSpMV.
  //
  cusparseConstSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  double *d_y = nullptr;
  double *d_buffer = nullptr;
  size_t bufferSize = 0;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(double)*m));
  CUSPARSE_CHECK(cusparseCreateConstCsr(&matA, m, m, nnzA,
    d_csrRowPtrA, d_csrColIndA, d_csrValA, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I, base, dtype));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, m, d_x, dtype));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, m, d_y, dtype));
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
    cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX,
    &zero, vecY, dtype, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  CUDA_CHECK(cudaMalloc(&d_buffer, bufferSize));
  CUSPARSE_CHECK(cusparseSpMV_preprocess(
    cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX,
    &zero, vecY, dtype, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

  //
  // Preprocessing Step 1: Form B := (A - mu0 * I).
  //
  CUDA_CHECK(cudaMemcpyAsync(csrValA.data(), d_csrValA, sizeof(double)*nnzA,
             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(csrColIndA.data(), d_csrColIndA, sizeof(int)*nnzA,
             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(csrRowPtrA.data(), d_csrRowPtrA, sizeof(int)*(m+1),
             cudaMemcpyDeviceToHost, stream));
  shift_diagonal(base, csrRowPtrA, csrColIndA, csrValA, mu0,
                 csrRowPtrB, csrColIndB, csrValB, nnzB);

  //
  // Preprocessing Step 2: The initial vector must have unit norm.
  //
  normalize_vector(cublasH, m, d_x0);

  //
  // Preprocessing Step 3: Transfer host data to device
  //
  // Allocate on device.
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrValB), sizeof(double)*nnzB));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrColIndB), sizeof(int)*nnzB));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrRowPtrB), sizeof(int)*(m+1)));

  // Copy B and x0 to device.
  CUDA_CHECK(cudaMemcpyAsync(d_csrValB, csrValB.data(), sizeof(double)*nnzB,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_csrColIndB, csrColIndB.data(), sizeof(int)*nnzB,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_csrRowPtrB, csrRowPtrB.data(), sizeof(int)*(m+1),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  //
  // Step 1: Compute LU factorization of B := A - mu0*I.
  //

  // Create matrix objects.
  cudssMatrix_t obj_B, obj_x, obj_x0;
  cudssIndexBase_t indexBase = (base == CUSPARSE_INDEX_BASE_ZERO) ? 
    CUDSS_BASE_ZERO : CUDSS_BASE_ONE;
  CUDSS_CHECK(cudssMatrixCreateCsr(&obj_B, m, m, nnzB, d_csrRowPtrB, nullptr,
                                   d_csrColIndB, d_csrValB, CUDA_R_32I,
                                   dtype, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL,
                                   indexBase));
  CUDSS_CHECK(cudssMatrixCreateDn(&obj_x, m, 1, m, d_x, dtype, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&obj_x0, m, 1, m, d_x0, dtype, CUDSS_LAYOUT_COL_MAJOR));

  // Analyze
  CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_ANALYSIS, config, data, obj_B, obj_x, obj_x0));

  // Factorize
  CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_FACTORIZATION, config, data, obj_B, obj_x, obj_x0));

  //
  // Step 2: Inverse iteration
  //
  bool converged = false;
  int iter = 0;
  while(!converged && iter++ < maxite) {
    // Solve B * x = x0 for x using the LU factorization of B computed before.
    CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_SOLVE, config, data, obj_B, obj_x, obj_x0));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Normalize x := x / ||x||_2.
    normalize_vector(cublasH, m, d_x);

    // Compute y = A * x.
    CUSPARSE_CHECK(cusparseSpMV(cusparseH, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &one, matA, vecX, &zero, vecY, dtype, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer));

    // Approximate eigenvalue mu = x**T * (A * x) = x**T * y.
    CUBLAS_CHECK(cublasDdot(cublasH, m, d_x, 1, d_y, 1, &mu));

    // Compute norm of residual |A*x - mu*x| = |y - mu*x| for convergence check.
    double norm = 0.0;
    double neg_mu = -mu;
    CUBLAS_CHECK(cublasDaxpy(cublasH, m, &neg_mu, d_x, 1, d_y, 1));
    CUBLAS_CHECK(cublasDnrm2(cublasH, m, d_y, 1, &norm));
    converged = norm < tol;

    // Use current eigenvector as the right-hand side in the next iteration.
    CUDA_CHECK(cudaMemcpyAsync(d_x0, d_x, m * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Copy mu to d_mu.
  CUDA_CHECK(cudaMemcpyAsync(d_mu, &mu, sizeof(double),cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Free resources
  CUDSS_CHECK(cudssMatrixDestroy(obj_B));
  CUDSS_CHECK(cudssMatrixDestroy(obj_x0));
  CUDSS_CHECK(cudssMatrixDestroy(obj_x));
  CUDA_CHECK(cudaFree(d_csrRowPtrB));
  CUDA_CHECK(cudaFree(d_csrColIndB));
  CUDA_CHECK(cudaFree(d_csrValB));
  CUDA_CHECK(cudaFree(d_y));
  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
  CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
  CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  CUSPARSE_CHECK(cusparseDestroy(cusparseH));
  CUDSS_CHECK(cudssDataDestroy(cudss, data));
  CUDSS_CHECK(cudssConfigDestroy(config));
  CUDSS_CHECK(cudssDestroy(cudss));
}

void set_zero(int m, double *d_x, double *d_mu) {
  CUDA_CHECK(cudaMemset(d_x, 0, sizeof(double)*m));
  CUDA_CHECK(cudaMemset(d_mu, 0, sizeof(double)));
}

void copy_eigenpair_to_host(
  int m, const double *d_mu, const double *d_x,
  void *h_mu, void *h_x) {
  CUDA_CHECK(cudaMemcpy(h_mu, d_mu, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_x, d_x, m*sizeof(double), cudaMemcpyDeviceToHost));
}

void approximate_eigenpair_cusolversp(
  cusparseIndexBase_t base,
  const int m, const int nnzA,
  int *d_csrRowPtrA, int *d_csrColIndA, double *d_csrValA, // CSR of A
  double mu0, double *d_x0,                                // initial guess
  int maxite, double tol,                                  // stopping criteria
  double *d_mu, double *d_x,                               // on exit, the computed eigenpair
  cudaStream_t stream) {
  cusolverSpHandle_t cusolverspH = nullptr;
  CUSOLVER_CHECK(cusolverSpCreate(&cusolverspH));

  // Set up A.
  cusparseMatDescr_t descrA = nullptr;
  CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
  CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, base));

  // Compute eigenpair.
  CUSOLVER_CHECK(cusolverSpDcsreigvsi(cusolverspH, m, nnzA, descrA,
    d_csrValA, d_csrRowPtrA, d_csrColIndA,
    mu0, d_x0, maxite, tol, d_mu, d_x));

  CUSPARSE_CHECK(cusparseDestroyMatDescr(descrA));
  CUSOLVER_CHECK(cusolverSpDestroy(cusolverspH));
}


int main(int argc, char *argv[]) {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  /*
   *      | 1   2   0   0 |
   *  A = | 2   0   3   0 |
   *      | 0   3   5   6 |
   *      | 0   0   6   0 |
   *  CSR of A is 0-based
   *
   *  Since cusolverSpDcsreigvsi only supports general matrices,
   *  the symmetry of A is not exploited.
   */
  const int m = 4;
  const int nnzA = 8;
  std::vector<int> csrRowPtrA = {0, 2, 4, 7, 8};
  std::vector<int> csrColIndA = {0, 1, 0, 2, 1, 2, 3, 2};
  std::vector<double> csrValA = {1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 6.0, 6.0};
  cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;

  //
  // Device copies.
  //
  int *d_csrRowPtrA = nullptr;
  int *d_csrColIndA = nullptr;
  double *d_csrValA = nullptr;

  //
  // Allocate on device.
  //
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrValA), sizeof(double)*nnzA));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrColIndA), sizeof(int)*nnzA));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_csrRowPtrA), sizeof(int)*(m+1)));

  //
  // Copy A to device.
  //
  CUDA_CHECK(cudaMemcpyAsync(d_csrValA, csrValA.data(), sizeof(double)*nnzA,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_csrColIndA, csrColIndA.data(), sizeof(int)*nnzA,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_csrRowPtrA, csrRowPtrA.data(), sizeof(int)*(m+1),
                             cudaMemcpyHostToDevice, stream));

  //
  // Stopping conditions.
  //
  int maxite = 10;   // maximum number of iterations
  double tol = 1e-6; // convergence tolerance - must not be less than |mu0|*eps


  //
  // Host copy of the initial guess (mu0, x0)
  // and the eigenpair (mu, x) that will be computed.
  //
  // The initial guess mu0 must be chosen such (A - mu0*I) is not singular.
  //
  double mu0 = -5.0, mu;
  std::vector<double> x0(m), x(m);
  randomize_vector(m, x0); // x0 must not be the zero vector

  //
  // Device copies.
  //
  double *d_x0 = nullptr;
  double *d_mu = nullptr;
  double *d_x = nullptr;

  //
  // Allocate on device and copy x0 to device.
  //
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x0), sizeof(double)*m));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double)*m));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_mu), sizeof(double)));
  CUDA_CHECK(cudaMemcpyAsync(d_x0, x0.data(), sizeof(double)*m,
                             cudaMemcpyHostToDevice, stream));

  //
  // Solve A*x = lambda*x with cusolverSpDcsreigvsi
  //
  std::cout << "Compute eigenpair with cusolverSp\n";
  approximate_eigenpair_cusolversp(base, m, nnzA,
    d_csrRowPtrA, d_csrColIndA, d_csrValA,
    mu0, d_x0, maxite, tol, d_mu, d_x, stream);
  copy_eigenpair_to_host(m, d_mu, d_x, &mu, x.data());
  std::cout << "computed eigenvalue mu=" << mu << " and eigenvector x\n";
  show_vector("x", m, x);

  // Reset x and mu before solving with cuDSS.
  set_zero(m, d_x, d_mu);

  //
  // Find an eigenpair (mu, x) satisfying A*x = mu*x with cuDSS
  //
  std::cout << "Compute eigenpair with cuDSS\n";
  approximate_eigenpair_cudss(base, m, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA,
    mu0, d_x0, maxite, tol, d_mu, d_x, stream);
  copy_eigenpair_to_host(m, d_mu, d_x, &mu, x.data());
  std::cout << "computed eigenvalue mu=" << mu << " and eigenvector x\n";
  show_vector("x", m, x);

  //
  // Free resources
  //
  CUDA_CHECK(cudaFree(d_x0));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_mu));

  CUDA_CHECK(cudaFree(d_csrRowPtrA));
  CUDA_CHECK(cudaFree(d_csrColIndA));
  CUDA_CHECK(cudaFree(d_csrValA));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}