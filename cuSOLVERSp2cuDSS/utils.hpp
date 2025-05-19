/*
 * Copyright 2024 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef CUSOLVERSP2CUDSS_UTILS_HPP
#define CUSOLVERSP2CUDSS_UTILS_HPP

/// std headers
#include <type_traits>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <numeric>
#include <complex>
#include <random>
#include <chrono>
#include <cfloat>

#if !defined(DISABLE_CUSPARSE_DEPRECATED)
#define DISABLE_CUSPARSE_DEPRECATED 1
#endif

#include <cuComplex.h>
#include <cuda_runtime.h> 
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusolverRf.h>
#include "cudss.h"

int parse_cmdline(int argc, char **argv, const std::string &option) {
  char **beg = argv, **end = argv+argc;
  char **iter = std::find(beg, end, option);

  return (iter != end);
}

int parse_cmdline(int argc, char **argv, const std::string &option, char** value) {
  char **beg = argv, **end = argv+argc;
  char **iter = std::find(beg, end, option);

  *value = (iter != end && ++iter != end) ? *iter : nullptr;    

  return (*value != nullptr);;
}

struct timer_measurement_t {
  std::string _label;
  cudaStream_t _stream;
  bool _enabled;
  
  std::chrono::time_point<std::chrono::high_resolution_clock> _beg, _end;

  timer_measurement_t(const std::string label, const cudaStream_t stream, const bool enabled) {
    _label = label;
    _stream = stream;
    _enabled = enabled;

    if (_enabled) {
      if (cudaSuccess == cudaStreamSynchronize(_stream)) {             
        _beg = std::chrono::high_resolution_clock::now();
      } else {
        std::cerr << "Error: failed to cudaStreamSynchronize, timer_measurement_t\n";
      }
    }
  }
  ~timer_measurement_t() {
    if (_enabled) {
      if (cudaSuccess == cudaStreamSynchronize(_stream)) {
        _end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _beg);
        std::cout << "-- " << _label << " = " << (duration.count()/1000.0) << " seconds" << "\n";
      } else {
        std::cerr << "Error: failed to cudaStreamSynchronize, ~timer_measurement_t\n";
      }
    }
  }
};


template<typename T>
struct arith_traits_t;

template<> struct arith_traits_t<float> {
  using value_type = float;
  using magnitude_type = float;
  using cuda_value_type = float;
  static constexpr cudaDataType_t cuda_data_type = CUDA_R_32F;
  static constexpr bool is_complex = false;
  static constexpr float epsilon = FLT_EPSILON;
};
template<> struct arith_traits_t<double> {
  using value_type = double;
  using magnitude_type = double;
  using cuda_value_type = double;
  static constexpr cudaDataType_t cuda_data_type = CUDA_R_64F;
  static constexpr bool is_complex = false;
  static constexpr double epsilon = DBL_EPSILON;    
};
template<> struct arith_traits_t<std::complex<float>> {
  using value_type = std::complex<float>;
  using magnitude_type = float;
  using cuda_value_type = cuComplex;
  static constexpr cudaDataType_t cuda_data_type = CUDA_C_32F;  
  static constexpr bool is_complex = true;
  static constexpr float epsilon = FLT_EPSILON;  
};
template<> struct arith_traits_t<std::complex<double>> {
  using value_type = std::complex<double>;
  using magnitude_type = double;
  using cuda_value_type = cuDoubleComplex;
  static constexpr cudaDataType_t cuda_data_type = CUDA_C_64F;    
  static constexpr bool is_complex = true;
  static constexpr double epsilon = DBL_EPSILON;  
};

template<typename T>
inline typename std::enable_if<!arith_traits_t<T>::is_complex, T>::type
conjugate(const T &val) {
  return val;
}

template<typename T>
inline typename std::enable_if<arith_traits_t<T>::is_complex, T>::type
conjugate(const T &val) {
  return std::conj(val);
}

template<typename T>
inline typename std::enable_if<!arith_traits_t<T>::is_complex, void>::type
randomize_vector(const ordinal_type m,
                 std::vector<T> &x) {
  std::mt19937 gen(0 /* rd */); /// for reproduceable test, fix the seed
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::generate(x.begin(), x.end(), [&]() { return T(dist(gen)); });  
}

template<typename T>
inline typename std::enable_if<arith_traits_t<T>::is_complex, void>::type
randomize_vector(const ordinal_type m,
                 std::vector<T> &x) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::generate(x.begin(), x.end(), [&]() { return T(dist(gen), dist(gen)); });  
}

template<typename value_type>
double compute_norm(std::vector<value_type> &x) {
  double sum(0);
  for (auto &val : x) {
    sum += std::real(val*conjugate(val));
  }
  return std::sqrt(sum);
}

template<typename value_type>
void compute_b_is_Ax(const ordinal_type m, const ordinal_type n, 
                     std::vector<ordinal_type> &ap,
                     std::vector<ordinal_type> &aj,
                     std::vector<value_type> &ax,
                     std::vector<value_type> &x,
                     std::vector<value_type> &b) {
  /// error check
  assert(ap.size() == (m+1));
  assert(aj.size() == ap[m]);
  assert(ax.size() == ap[m]);
  assert(x.size() == m);
  assert(b.size() == m);

  /// b = Ax
  for (ordinal_type i=0;i<m;++i) {
    const ordinal_type kbeg = ap[i], kend = ap[i+1];
    value_type tmp(0);
    for (ordinal_type k=kbeg;k<kend;++k) {
      const value_type aa = ax[k];
      const value_type xx = x[aj[k]];
      tmp += aa*xx;
    }
    b[i] = tmp;
  }
}

template<typename value_type>
int compute_r_is_Ax_minus_b(const ordinal_type m, const ordinal_type n, 
                            std::vector<ordinal_type> &ap,
                            std::vector<ordinal_type> &aj,
                            std::vector<value_type> &ax,
                            std::vector<value_type> &x,
                            std::vector<value_type> &b,
                            std::vector<value_type> &r) {
  /// error check
  assert(r.size() == m);

  /// r = Ax
  compute_b_is_Ax(m, n,
                  ap, aj, ax,
                  x,
                  r);

  /// r = r - b
  std::transform(r.begin(), r.end(), b.begin(),
                 r.begin(), std::minus<value_type>());
  
  return 0;
}

template<typename value_type>
void perturb_diag_A(const ordinal_type m, const ordinal_type n, 
                    std::vector<ordinal_type> &ap,
                    std::vector<ordinal_type> &aj,
                    std::vector<value_type> &ax,
                    std::vector<value_type> &x) {
  /// error check
  assert(ap.size() == (m+1));
  assert(aj.size() == ap[m]);
  assert(ax.size() == ap[m]);
  assert(x.size() == m);

  /// diag(A) += x
  for (ordinal_type i=0;i<m;++i) {
    const ordinal_type kbeg = ap[i], kend = ap[i+1];
    value_type tmp(0);
    for (ordinal_type k=kbeg;k<kend;++k) {
      const ordinal_type j = aj[k];
      if (i == j) {
        ax[k] += x[i];
      }
    }
  }
}



template<typename value_type>
struct coo_t {
  ordinal_type _i, _j;
  value_type _x;
  coo_t() = default;
  coo_t(const coo_t &b) = default;
  coo_t(const ordinal_type i, const ordinal_type j, const value_type x)
    : _i(i), _j(j), _x(x) {}

  /// compare less, equality for indices only
  bool operator<(const coo_t &y) const {
    const auto r_val = (_i - y._i);
    return (r_val == 0 ? _j < y._j : r_val < 0);
  }
  bool operator==(const coo_t &y) const {
    return (_i == y._i) && (_j == y._j);
  }    
  bool operator!=(const coo_t &y) const {
    return !(*this == y);
  } 
};


template<typename T>
inline typename std::enable_if<!arith_traits_t<T>::is_complex>::type
read_value_from_file(std::ifstream &file, ordinal_type &i, ordinal_type &j, T &x) {
  file >> i >> j >> x;
}

template<typename T>
inline typename std::enable_if<arith_traits_t<T>::is_complex>::type
read_value_from_file(std::ifstream &file, ordinal_type &i, ordinal_type &j, T &x) {
  typename T::value_type a, b;
  file >> i >> j >> a >> b;
  x = T(a,b);
}

template<typename value_type>
int coo_to_csr(const ordinal_type m,
               const ordinal_type nnz,
               std::vector<coo_t<value_type>> &mm,
               std::vector<ordinal_type> &ap,
               std::vector<ordinal_type> &aj,
               std::vector<value_type> &ax,
               bool useZeroBasedIndexing = true) {
  ap.resize(m+1);
  aj.resize(nnz);
  ax.resize(nnz);

  using ijx_type = coo_t<value_type>;

  // Adjust indexing depending on zero-based or one-based indexing.
  int offset = useZeroBasedIndexing ? 0 : 1;

  ordinal_type icnt = 0;
  ordinal_type jcnt = 0;
  ijx_type prev = mm[0];
  
  ap[icnt++] = offset;
  aj[jcnt] = prev._j;
  ax[jcnt++] = prev._x;
  
  for (auto it = (mm.begin() + 1); it < (mm.end()); ++it) {
    const ijx_type aij = (*it);
    
    if (aij._i != prev._i)
      ap[icnt++] = jcnt + offset;
    
    if (aij == prev) {
      aj[jcnt-1] = aij._j;
      ax[jcnt-1] += aij._x;
    } else {
      aj[jcnt] = aij._j;
      ax[jcnt++] = aij._x;
    }
    prev = aij;
  }
  ap[icnt++] = jcnt + offset;

  return 0;
}

template<typename value_type>
int csr_to_coo(const ordinal_type m,
               const ordinal_type nnz,
               std::vector<ordinal_type> &ap,
               std::vector<ordinal_type> &aj,
               std::vector<value_type> &ax,
               std::vector<coo_t<value_type>> &mm) {
  mm.resize(nnz);

  // Adjust indexing depending on zero-based or one-based indexing.
  int offset = 0;
  if (ap.size() > 0) {
    offset = (ap[0] == 0) ? 0 : 1;
  }

  using ijx_type = coo_t<value_type>;  
  for (ordinal_type i=0,cnt=0;i<m;++i) {
    const ordinal_type kbeg = ap[i], kend = ap[i+1];
    for (ordinal_type k=kbeg;k<kend;++k) {
      const ordinal_type j = aj[k-offset]-offset;
      const value_type x = ax[k-offset];
      mm[cnt++] = ijx_type(i + offset, j + offset, x);
    }
  }
  
  return 0;
}

template<typename value_type>
void show_csr(const std::string label,
              const ordinal_type m,
              const ordinal_type n,
              const std::vector<ordinal_type> &ap,
              const std::vector<ordinal_type> &aj,
              const std::vector<value_type> &ax) {
  std::cout << "-- sparse matrix: " << label << " [" << m << " x " << n << "]\n";

  std::cout << "   ap = [ ";
  for (auto &p : ap) {
    std::cout << p << " ";
  }
  std::cout << "]\n";  

  std::cout << "   aj = [ ";
  for (auto &j : aj) {
    std::cout << j << " ";
  }
  std::cout << "]\n";
  
  std::cout << "   ax = [ ";
  for (auto &x : ax) {
    std::cout << x << " ";
  }
  std::cout << "]\n";
}

template<typename value_type>
void show_vector(const std::string label,
                 const ordinal_type m,
                 const std::vector<value_type> &x) {
  std::cout << "-- vector: " << label << " [" << m << "]\n";
  for (ordinal_type i=0;i<m;++i) {
    std::cout << "   " << x[i] << "\n";
  }
}

template<typename value_type>
int read_matrixmarket(std::string filename,
                      ordinal_type &m,
                      ordinal_type &n,
                      std::vector<ordinal_type> &ap,
                      std::vector<ordinal_type> &aj,
                      std::vector<value_type> &ax,
                      const int verbose) {
  std::ifstream file;
  file.open(filename);
  
  // reading mm header
  ordinal_type nnz;
  bool symmetry = false, hermitian = false;
  {
    std::string header;
    std::getline(file, header);
    while (file.good()) {
      char c = file.peek();
      if (c == '%' || c == '\n') {
        file.ignore(256, '\n');
        continue;
      }
      break;
    }
    std::transform(header.begin(), header.end(), header.begin(), ::tolower);
    symmetry = (header.find("symmetric") != std::string::npos || header.find("hermitian") != std::string::npos);
    hermitian = (header.find("hermitian") != std::string::npos);

    file >> m >> n >> nnz;
  }

  if (verbose) {
    std::cout << "-- read matrix market\n";
    std::cout << (symmetry ? "symmetric" :"general") << " " << (hermitian ? "hermitian" : "") << "\n";
    std::cout << m << " " << n << " " << nnz << "\n";  
  }
  
  // read data into coo format  
  using ijx_type = coo_t<value_type>;
  constexpr ordinal_type mm_base = 1;  
  std::vector<ijx_type> mm;
  mm.reserve(nnz * (symmetry ? 2 : 1));
  for (ordinal_type i = 0; i < nnz; ++i) {
    ordinal_type row(0), col(0);
    value_type val;
    
    read_value_from_file(file, row, col, val);
    
    row -= mm_base;
    col -= mm_base;
    
    mm.push_back(ijx_type(row, col, val));
    if (symmetry && row != col) {
      const value_type conj_val = conjugate(val);
      mm.push_back(ijx_type(col, row, hermitian ? conj_val : val));
    }
  }
  std::sort(mm.begin(), mm.end(), std::less<ijx_type>());
  nnz = mm.size();

  if (verbose) {
    for (auto &ijx : mm) 
      std::cout << "   " << ijx._i << " " << ijx._j << " " << ijx._x << "\n"; 
  }
  coo_to_csr(m, nnz, mm, ap, aj, ax);

  return 0;
}


// CUDA API error checking
#define CUDA_CHECK(err)                                             \
  do {                                                              \
    cudaError_t err_ = (err);                                       \
    if (err_ != cudaSuccess) {                                      \
      printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                       \
    }                                                               \
  } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                             \
  do {                                                                  \
    cusolverStatus_t err_ = (err);                                      \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                              \
      printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cusolver error");                       \
    }                                                                   \
  } while (0)

// cusparse API error checking
#define CUSPARSE_CHECK(err)                                             \
  do {                                                                  \
    cusparseStatus_t err_ = (err);                                      \
    if (err_ != CUSPARSE_STATUS_SUCCESS) {                              \
      printf("cusparse error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cusparse error");                       \
    }                                                                   \
  } while (0)

// cudss API error checking
#define CUDSS_CHECK(err)                                                \
  do {                                                                  \
    cudssStatus_t err_ = (err);                                         \
    if (err_ != CUDSS_STATUS_SUCCESS) {                                 \
      printf("cudss error %d at %s:%d\n", err_, __FILE__, __LINE__);    \
      throw std::runtime_error("cudss error");                          \
    }                                                                   \
  } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

///
/// Unified X interface to cusolver APIs
///

cusolverStatus_t cusolverSpXcsrcholBufferInfo(cusolverSpHandle_t handle,
                                              int n,
                                              int nnzA,
                                              const cusparseMatDescr_t descrA,
                                              const float *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              csrcholInfo_t info,
                                              size_t *internalDataInBytes,
                                              size_t *workspaceInBytes) {
  return cusolverSpScsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes);
}
cusolverStatus_t cusolverSpXcsrcholBufferInfo(cusolverSpHandle_t handle,
                                              int n,
                                              int nnzA,
                                              const cusparseMatDescr_t descrA,
                                              const double *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              csrcholInfo_t info,
                                              size_t *internalDataInBytes,
                                              size_t *workspaceInBytes) {
  return cusolverSpDcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes);
}

cusolverStatus_t cusolverSpXcsrcholBufferInfo(cusolverSpHandle_t handle,
                                              int n,
                                              int nnzA,
                                              const cusparseMatDescr_t descrA,
                                              const cuComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              csrcholInfo_t info,
                                              size_t *internalDataInBytes,
                                              size_t *workspaceInBytes) {
  return cusolverSpCcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes);
}
cusolverStatus_t cusolverSpXcsrcholBufferInfo(cusolverSpHandle_t handle,
                                              int n,
                                              int nnzA,
                                              const cusparseMatDescr_t descrA,
                                              const cuDoubleComplex *csrValA,
                                              const int *csrRowPtrA,
                                              const int *csrColIndA,
                                              csrcholInfo_t info,
                                              size_t *internalDataInBytes,
                                              size_t *workspaceInBytes) {
  return cusolverSpZcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, workspaceInBytes);
}


cusolverStatus_t cusolverSpXcsrcholFactor(cusolverSpHandle_t handle,
                                          int n,
                                          int nnzA,
                                          const cusparseMatDescr_t descrA,
                                          const float *csrValA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA,
                                          csrcholInfo_t info,
                                          void *pBuffer) {
  return cusolverSpScsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrcholFactor(cusolverSpHandle_t handle,
                                          int n,
                                          int nnzA,
                                          const cusparseMatDescr_t descrA,
                                          const double *csrValA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA,
                                          csrcholInfo_t info,
                                          void *pBuffer) {
  return cusolverSpDcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}
cusolverStatus_t cusolverSpXcsrcholFactor(cusolverSpHandle_t handle,
                                          int n,
                                          int nnzA,
                                          const cusparseMatDescr_t descrA,
                                          const cuComplex *csrValA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA,
                                          csrcholInfo_t info,
                                          void *pBuffer) {
  return cusolverSpCcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}
cusolverStatus_t cusolverSpXcsrcholFactor(cusolverSpHandle_t handle,
                                          int n,
                                          int nnzA,
                                          const cusparseMatDescr_t descrA,
                                          const cuDoubleComplex *csrValA,
                                          const int *csrRowPtrA,
                                          const int *csrColIndA,
                                          csrcholInfo_t info,
                                          void *pBuffer) {
  return cusolverSpZcsrcholFactor(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBuffer);
}


cusolverStatus_t cusolverSpXcsrcholSolve(cusolverSpHandle_t handle,
                                         int n,
                                         const float *b,
                                         float *x,
                                         csrcholInfo_t info,
                                         void *pBuffer) {
  return cusolverSpScsrcholSolve(handle, n, b, x, info, pBuffer);
}
cusolverStatus_t cusolverSpXcsrcholSolve(cusolverSpHandle_t handle,
                                         int n,
                                         const double *b,
                                         double *x,
                                         csrcholInfo_t info,
                                         void *pBuffer) {
  return cusolverSpDcsrcholSolve(handle, n, b, x, info, pBuffer);
}
cusolverStatus_t cusolverSpXcsrcholSolve(cusolverSpHandle_t handle,
                                         int n,
                                         const cuComplex *b,
                                         cuComplex *x,
                                         csrcholInfo_t info,
                                         void *pBuffer) {
  return cusolverSpCcsrcholSolve(handle, n, b, x, info, pBuffer);
}
cusolverStatus_t cusolverSpXcsrcholSolve(cusolverSpHandle_t handle,
                                         int n,
                                         const cuDoubleComplex *b,
                                         cuDoubleComplex *x,
                                         csrcholInfo_t info,
                                         void *pBuffer) {
  return cusolverSpZcsrcholSolve(handle, n, b, x, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrlsvchol(cusolverSpHandle_t handle,
                                       int m,
                                       int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const float *csrVal,
                                       const int *csrRowPtr,
                                       const int *csrColInd,
                                       const float *b,
                                       float tol,
                                       int reorder,
                                       float *x,
                                       int *singularity) {
  return cusolverSpScsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

cusolverStatus_t cusolverSpXcsrlsvchol(cusolverSpHandle_t handle,
                                       int m,
                                       int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const double *csrVal,
                                       const int *csrRowPtr,
                                       const int *csrColInd,
                                       const double *b,
                                       double tol,
                                       int reorder,
                                       double *x,
                                       int *singularity) {
  return cusolverSpDcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}

cusolverStatus_t cusolverSpXcsrlsvchol(cusolverSpHandle_t handle,
                                       int m,
                                       int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const cuComplex *csrVal,
                                       const int *csrRowPtr,
                                       const int *csrColInd,
                                       const cuComplex *b,
                                       float tol,
                                       int reorder,
                                       cuComplex *x,
                                       int *singularity) {
  return cusolverSpCcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity);
}  

cusolverStatus_t cusolverSpXcsrlsvchol(cusolverSpHandle_t handle,
                                       int m,
                                       int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const cuDoubleComplex *csrVal,
                                       const int *csrRowPtr,
                                       const int *csrColInd,
                                       const cuDoubleComplex *b,
                                       double tol,
                                       int reorder,
                                       cuDoubleComplex *x,
                                       int *singularity) {
  return cusolverSpZcsrlsvchol(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, b, tol, reorder, x, singularity); 
}

cusolverStatus_t cusolverSpXcsrlsvluHost(cusolverSpHandle_t handle,
                                         int n,
                                         int nnzA,
                                         const cusparseMatDescr_t descrA,
                                         const float * csrValA,
                                         const int * csrRowPtrA,
                                         const int * csrColIndA,
                                         const float * b,
                                         float tol,
                                         int reorder,
                                         float * x,
                                         int * singularity) {
  return cusolverSpScsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

cusolverStatus_t cusolverSpXcsrlsvluHost(cusolverSpHandle_t handle,
                                         int n,
                                         int nnzA,
                                         const cusparseMatDescr_t descrA,
                                         const double * csrValA,
                                         const int * csrRowPtrA,
                                         const int * csrColIndA,
                                         const double * b,
                                         double tol,
                                         int reorder,
                                         double * x,
                                         int * singularity) {
  return cusolverSpDcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

cusolverStatus_t cusolverSpXcsrlsvluHost(cusolverSpHandle_t handle,
                                         int n,
                                         int nnzA,
                                         const cusparseMatDescr_t descrA,
                                         const cuComplex * csrValA,
                                         const int * csrRowPtrA,
                                         const int * csrColIndA,
                                         const cuComplex * b,
                                         float tol,
                                         int reorder,
                                         cuComplex * x,
                                         int * singularity) {
  return cusolverSpCcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

cusolverStatus_t cusolverSpXcsrlsvluHost(cusolverSpHandle_t handle,
                                         int n,
                                         int nnzA,
                                         const cusparseMatDescr_t descrA,
                                         const cuDoubleComplex * csrValA,
                                         const int * csrRowPtrA,
                                         const int * csrColIndA,
                                         const cuDoubleComplex * b,
                                         double tol,
                                         int reorder,
                                         cuDoubleComplex * x,
                                         int * singularity) {
  return cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, tol, reorder, x, singularity);
}

cusolverStatus_t cusolverSpXcsrluExtractHost(cusolverSpHandle_t handle,
                                             int * P,
                                             int * Q,
                                             const cusparseMatDescr_t descrL,
                                             float * csrValL,
                                             int * csrRowPtrL,
                                             int * csrColIndL,
                                             const cusparseMatDescr_t descrU,
                                             float * csrValU,
                                             int * csrRowPtrU,
                                             int * csrColIndU,
                                             csrluInfoHost_t info,
                                             void * pBuffer) {
  return cusolverSpScsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluExtractHost(cusolverSpHandle_t handle,
                                             int * P,
                                             int * Q,
                                             const cusparseMatDescr_t descrL,
                                             double * csrValL,
                                             int * csrRowPtrL,
                                             int * csrColIndL,
                                             const cusparseMatDescr_t descrU,
                                             double * csrValU,
                                             int * csrRowPtrU,
                                             int * csrColIndU,
                                             csrluInfoHost_t info,
                                             void * pBuffer) {
  return cusolverSpDcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluExtractHost(cusolverSpHandle_t handle,
                                             int * P,
                                             int * Q,
                                             const cusparseMatDescr_t descrL,
                                             cuComplex * csrValL,
                                             int * csrRowPtrL,
                                             int * csrColIndL,
                                             const cusparseMatDescr_t descrU,
                                             cuComplex * csrValU,
                                             int * csrRowPtrU,
                                             int * csrColIndU,
                                             csrluInfoHost_t info,
                                             void * pBuffer) {
  return cusolverSpCcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluExtractHost(cusolverSpHandle_t handle,
                                             int * P,
                                             int * Q,
                                             const cusparseMatDescr_t descrL,
                                             cuDoubleComplex * csrValL,
                                             int * csrRowPtrL,
                                             int * csrColIndL,
                                             const cusparseMatDescr_t descrU,
                                             cuDoubleComplex * csrValU,
                                             int * csrRowPtrU,
                                             int * csrColIndU,
                                             csrluInfoHost_t info,
                                             void * pBuffer) {
  return cusolverSpZcsrluExtractHost(handle, P, Q, descrL, csrValL, csrRowPtrL, csrColIndL, descrU, csrValU, csrRowPtrU, csrColIndU, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluBufferInfoHost(cusolverSpHandle_t handle,
                                                int n,
                                                int nnzA,
                                                const cusparseMatDescr_t descrA,
                                                const float * csrValA,
                                                const int * csrRowPtrA,
                                                const int * csrColIndA,
                                                csrluInfoHost_t info,
                                                size_t * internalDataInBytes,
                                                size_t * bufferSizeInBytes) {
  return cusolverSpScsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, bufferSizeInBytes);
}

cusolverStatus_t cusolverSpXcsrluBufferInfoHost(cusolverSpHandle_t handle,
                                                int n,
                                                int nnzA,
                                                const cusparseMatDescr_t descrA,
                                                const double * csrValA,
                                                const int * csrRowPtrA,
                                                const int * csrColIndA,
                                                csrluInfoHost_t info,
                                                size_t * internalDataInBytes,
                                                size_t * bufferSizeInBytes) {
  return cusolverSpDcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, bufferSizeInBytes);
}

cusolverStatus_t cusolverSpXcsrluBufferInfoHost(cusolverSpHandle_t handle,
                                                int n,
                                                int nnzA,
                                                const cusparseMatDescr_t descrA,
                                                const cuComplex * csrValA,
                                                const int * csrRowPtrA,
                                                const int * csrColIndA,
                                                csrluInfoHost_t info,
                                                size_t * internalDataInBytes,
                                                size_t * bufferSizeInBytes) {
  return cusolverSpCcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, bufferSizeInBytes);
}

cusolverStatus_t cusolverSpXcsrluBufferInfoHost(cusolverSpHandle_t handle,
                                                int n,
                                                int nnzA,
                                                const cusparseMatDescr_t descrA,
                                                const cuDoubleComplex * csrValA,
                                                const int * csrRowPtrA,
                                                const int * csrColIndA,
                                                csrluInfoHost_t info,
                                                size_t * internalDataInBytes,
                                                size_t * bufferSizeInBytes) {
  return cusolverSpZcsrluBufferInfoHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, internalDataInBytes, bufferSizeInBytes);
}

cusolverStatus_t cusolverSpXcsrluFactorHost(cusolverSpHandle_t handle,
                                            int n,
                                            int nnzA,
                                            const cusparseMatDescr_t descrA,
                                            const float * csrValA,
                                            const int * csrRowPtrA,
                                            const int * csrColIndA,
                                            csrluInfoHost_t info,
                                            float pivot_threshold,
                                            void * pBuffer) {
  return cusolverSpScsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluFactorHost(cusolverSpHandle_t handle,
                                            int n,
                                            int nnzA,
                                            const cusparseMatDescr_t descrA,
                                            const double * csrValA,
                                            const int * csrRowPtrA,
                                            const int * csrColIndA,
                                            csrluInfoHost_t info,
                                            double pivot_threshold,
                                            void * pBuffer) {
  return cusolverSpDcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluFactorHost(cusolverSpHandle_t handle,
                                            int n,
                                            int nnzA,
                                            const cusparseMatDescr_t descrA,
                                            const cuComplex * csrValA,
                                            const int * csrRowPtrA,
                                            const int * csrColIndA,
                                            csrluInfoHost_t info,
                                            float pivot_threshold,
                                            void * pBuffer) {
  return cusolverSpCcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluFactorHost(cusolverSpHandle_t handle,
                                            int n,
                                            int nnzA,
                                            const cusparseMatDescr_t descrA,
                                            const cuDoubleComplex * csrValA,
                                            const int * csrRowPtrA,
                                            const int * csrColIndA,
                                            csrluInfoHost_t info,
                                            double pivot_threshold,
                                            void * pBuffer) {
  return cusolverSpZcsrluFactorHost(handle, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info, pivot_threshold, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluSolveHost(cusolverSpHandle_t handle,
                                           int n,
                                           const float * b,
                                           float * x,
                                           csrluInfoHost_t info,
                                           void * pBuffer) {
  return cusolverSpScsrluSolveHost(handle, n, b, x, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluSolveHost(cusolverSpHandle_t handle,
                                           int n,
                                           const double * b,
                                           double * x,
                                           csrluInfoHost_t info,
                                           void * pBuffer) {
  return cusolverSpDcsrluSolveHost(handle, n, b, x, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluSolveHost(cusolverSpHandle_t handle,
                                           int n,
                                           const cuComplex * b,
                                           cuComplex * x,
                                           csrluInfoHost_t info,
                                           void * pBuffer) {
  return cusolverSpCcsrluSolveHost(handle, n, b, x, info, pBuffer);
}

cusolverStatus_t cusolverSpXcsrluSolveHost(cusolverSpHandle_t handle,
                                           int n,
                                           const cuDoubleComplex * b,
                                           cuDoubleComplex * x,
                                           csrluInfoHost_t info,
                                           void * pBuffer) {
  return cusolverSpZcsrluSolveHost(handle, n, b, x, info, pBuffer);
}

#endif
