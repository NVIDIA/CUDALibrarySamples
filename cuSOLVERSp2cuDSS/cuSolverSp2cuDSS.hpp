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

#ifndef CUSOLVERSP2CUDSS_HPP
#define CUSOLVERSP2CUDSS_HPP 

using ordinal_type = int;

#include "utils.hpp"

void show_usage(const std::string &exec_name) {
  std::cout << "usage: " << exec_name << " --solver <cudss;cusolver> --file <filename> --verbose\n";
  std::cout << "  --solver : select a linear solver; cudss, cusolver\n";
  std::cout << "  --file : sparse matrix input as a matrix market format\n";
  std::cout << "  --single-api,-s : use a single api for linear solve if available; \n";        
  std::cout << "  --timer,-t : enable timer to measure solver phases\n";  
  std::cout << "  --verbose,-v : verbose flag\n";
  std::cout << "  --help,-h : print usage\n";  
}

template<typename value_type>
int driver(int argc, char * argv[]) {
  using cuda_value_type = typename arith_traits_t<value_type>::cuda_value_type;
  using magnitude_type = typename arith_traits_t<value_type>::magnitude_type;
  constexpr cudaDataType_t cuda_data_type = arith_traits_t<value_type>::cuda_data_type;
  constexpr magnitude_type epsilon = arith_traits_t<value_type>::epsilon;
  
  ///
  /// Parse command line input
  /// 
  int use_cudss(0);
  int use_timer(0);
  int use_single_api(0);
  std::string filename("test.mtx");
  int verbose(0);  
  
  /// parse for solver type, filename and verbose
  {
    char *opt_val(nullptr);
    if (parse_cmdline(argc, argv, "--help") || parse_cmdline(argc, argv, "-h")) {
      show_usage(argv[0]);
      return 0;      
    }
    if (parse_cmdline(argc, argv, "--solver", &opt_val) && opt_val != nullptr) {
      use_cudss = std::string(opt_val) == "cudss";
    }
    if (parse_cmdline(argc, argv, "--file", &opt_val) && opt_val != nullptr) {
      filename = std::string(opt_val);
    }
    
    if (parse_cmdline(argc, argv, "--single-api") || parse_cmdline(argc, argv, "-s")) {
      use_single_api = 1;
    }
    if (parse_cmdline(argc, argv, "--timer") || parse_cmdline(argc, argv, "-t")) {
      use_timer = 1;
    }
    if (parse_cmdline(argc, argv, "--verbose") || parse_cmdline(argc, argv, "-v")) {
      verbose = 1;
    }
  }
  
  std::cout << "-- commandline input\n";
  std::cout << "   solver: " << (use_cudss ? "cudss, hpd" : "cusolversp, cholesky") << "\n";
  std::cout << "   filename: " << filename << "\n";
  std::cout << "   mode: " << (use_single_api ? "single api for linear-solve" : "phase-separated api e.g., analysis, factorize, solve") << "\n";  
  std::cout << "   timer: " << (use_timer ? "enabled" : "disabled") << "\n";  
  std::cout << "   verbose: " << verbose << "\n";

  try {
    
    ///
    /// Step 0: Initialize cuda, cusolver, cusparse, cudss, etc.
    ///

    /// CUDA init/finalization
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto cuda_finalize = [&]() {
      CUDA_CHECK(cudaStreamDestroy(stream));
    };
    
    /// CUSOLVERSP init/finalization
    cusolverSpHandle_t cusolversp;
    cusparseHandle_t cusparse;
    cusparseMatDescr_t descriptor;    
    csrcholInfo_t chol_info;

    CUSOLVER_CHECK(cusolverSpCreate(&cusolversp));
    CUSOLVER_CHECK(cusolverSpCreateCsrcholInfo(&chol_info));
    
    CUSPARSE_CHECK(cusparseCreate(&cusparse));
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descriptor));        

    auto cusolver_finalize = [&]() {
      CUSPARSE_CHECK(cusparseDestroyMatDescr(descriptor));      
      CUSPARSE_CHECK(cusparseDestroy(cusparse));
      
      CUSOLVER_CHECK(cusolverSpDestroyCsrcholInfo(chol_info));      
      CUSOLVER_CHECK(cusolverSpDestroy(cusolversp));
    };

    /// CUDSS init/finalization
    cudssHandle_t cudss;
    cudssConfig_t config;
    cudssData_t data;
    
    CUDSS_CHECK(cudssCreate(&cudss));
    CUDSS_CHECK(cudssConfigCreate(&config));
    CUDSS_CHECK(cudssDataCreate(cudss, &data));
    
    auto cudss_finalize = [&]() {
      CUDSS_CHECK(cudssDataDestroy(cudss, data));
      CUDSS_CHECK(cudssConfigDestroy(config));
      CUDSS_CHECK(cudssDestroy(cudss));
    };

    /// Assign stream to cusolversp, cusparse and cudss
    CUSOLVER_CHECK(cusolverSpSetStream(cusolversp, stream));
    CUSPARSE_CHECK(cusparseSetStream(cusparse, stream));                                                          
    CUDSS_CHECK(cudssSetStream(cudss, stream));

    /// Set matrix descriptor; in this example, we use general matrix type and zero base index
    CUSPARSE_CHECK(cusparseSetMatType(descriptor, CUSPARSE_MATRIX_TYPE_GENERAL));                                           
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descriptor, CUSPARSE_INDEX_BASE_ZERO));    

    ///
    /// Step 1: Set problem Ax = b
    ///
    
    /// Input csr sparse matrix
    ordinal_type m(0), n(0);
    std::vector<ordinal_type> h_ap, h_aj;
    std::vector<value_type> h_ax;

    /// Right hand side and solution vector
    std::vector<value_type> h_b, h_x;      

    /// Read A from matrix market file
    read_matrixmarket<value_type>(filename, m, n, h_ap, h_aj, h_ax, verbose);
    const ordinal_type nnz = h_ax.size();

    if (verbose) {
      show_csr("A", m, n, h_ap, h_aj, h_ax);
    }
    
    /// Randomize solution vector
    h_x.resize(m);
    randomize_vector(m, h_x);

    /// Compute right hand side vector b <- Ax
    h_b.resize(m);
    compute_b_is_Ax(m, n,
                    h_ap, h_aj, h_ax,
                    h_x, 
                    h_b);

    if (verbose) {
      show_vector("x", m, h_x);      
      show_vector("b", m, h_b);
    }

    ///
    /// Step 2: Fill-reduce ordering
    ///

    /// Permutation and inverse permutation array
    std::vector<ordinal_type> h_perm, h_peri;
    h_perm.resize(m);
    h_peri.resize(m);

    /// Explicit reordering for cusolver; cudss does not need this process
    if (!use_cudss) {
      CUSOLVER_CHECK(cusolverSpXcsrmetisndHost(cusolversp, m, nnz, descriptor,                                           
                                               h_ap.data(), h_aj.data(),
                                               NULL, /* default setting. */                                           
                                               h_perm.data()));   

      for (ordinal_type i=0;i<m;++i) {
        h_peri[h_perm[i]] = i;
      }

      if (verbose) {
        std::cout << "-- perm = [ ";
        for (auto &p : h_perm) 
          std::cout << p << " ";
        std::cout << "] \n";
        std::cout << "-- peri = [ ";
        for (auto &p : h_peri) 
          std::cout << p << " ";
        std::cout << "] \n";
      }
      
      using ijx_type = coo_t<value_type>;
      std::vector<ijx_type> h_mm;
      csr_to_coo(m, nnz, h_ap, h_aj, h_ax, h_mm);
      for (auto &val : h_mm) {
        val._i = h_perm[val._i];
        val._j = h_perm[val._j];
      }
      std::sort(h_mm.begin(), h_mm.end(), std::less<ijx_type>());      
      coo_to_csr(m, nnz, h_mm, h_ap, h_aj, h_ax);

      if (verbose) {
        show_csr("P A P^T", m, n, h_ap, h_aj, h_ax);        
      }
      
      std::vector<value_type> h_tmp = h_b;
      std::transform(h_perm.begin(), h_perm.end(), h_b.begin(), 
                     [&h_tmp](int idx) { return h_tmp[idx]; });

      if (verbose) {
        show_vector("P b", m, h_b);              
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    ///
    /// Step 3: Transfer host data to device
    ///
    
    /// allocate on device
    ordinal_type *d_ap, *d_aj;
    value_type *d_ax, *d_x, *d_b;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ap), sizeof(ordinal_type)*h_ap.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aj), sizeof(ordinal_type)*h_aj.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ax), sizeof(value_type)*h_ax.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(value_type)*h_x.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(value_type)*h_b.size()));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    /// clean-up 
    auto cuda_free_memory = [&]() {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ap)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_aj)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_ax)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_x)));
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_b)));
    };

    /// transfer memory to device
    CUDA_CHECK(cudaMemcpyAsync(d_ap, h_ap.data(), sizeof(ordinal_type)*h_ap.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_aj, h_aj.data(), sizeof(ordinal_type)*h_aj.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ax, h_ax.data(), sizeof(value_type)*h_ax.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b.data(), sizeof(value_type)*h_b.size(), cudaMemcpyHostToDevice, stream));    

    /// set zeros on x 
    CUDA_CHECK(cudaMemsetAsync(d_x, 0, sizeof(value_type)*h_x.size(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    ///
    /// Step 4: Solve the linear system using a direct solver 
    ///
    if (use_cudss) {
      ///
      /// Step 4.a.0: Create matrix objects
      ///
      cudssMatrix_t obj_A, obj_x, obj_b;  
      CUDSS_CHECK(cudssMatrixCreateCsr(&obj_A,
                                       m, n, nnz,
                                       d_ap, nullptr,
                                       d_aj, d_ax, 
                                       CUDA_R_32I,
                                       cuda_data_type,
                                       CUDSS_MTYPE_HPD,
                                       CUDSS_MVIEW_FULL,
                                       CUDSS_BASE_ZERO));
      CUDSS_CHECK(cudssMatrixCreateDn(&obj_x, m, 1, m, d_x, cuda_data_type, CUDSS_LAYOUT_COL_MAJOR));
      CUDSS_CHECK(cudssMatrixCreateDn(&obj_b, m, 1, m, d_b, cuda_data_type, CUDSS_LAYOUT_COL_MAJOR));

      CUDA_CHECK(cudaStreamSynchronize(stream));

      ///
      /// Step 4.a.1: Analyze
      ///
      {
        std::unique_ptr<timer_measurement_t> timer(new timer_measurement_t("cudss:analyze", stream, use_timer));
        CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_ANALYSIS, config, data, obj_A, obj_x, obj_b));
      }
      ///
      /// Step 4.a.2: Factorize
      ///
      {
        std::unique_ptr<timer_measurement_t> timer(new timer_measurement_t("cudss:factorize", stream, use_timer));        
        CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_FACTORIZATION, config, data, obj_A, obj_x, obj_b));
      }
      ///
      /// Step 4.a.3: Solve
      ///
      {
        std::unique_ptr<timer_measurement_t> timer(new timer_measurement_t("cudss:solve", stream, use_timer));                
        CUDSS_CHECK(cudssExecute(cudss, CUDSS_PHASE_SOLVE, config, data, obj_A, obj_x, obj_b));            
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));      
    
      ///
      /// Step 4.a.4: Free objects
      /// 
      CUDSS_CHECK(cudssMatrixDestroy(obj_A));      
      CUDSS_CHECK(cudssMatrixDestroy(obj_b));
      CUDSS_CHECK(cudssMatrixDestroy(obj_x));
    } else {
      ///
      /// Step 4.b.0: cusolversp does not exploit matrix objects
      ///

      if (use_single_api) {
        ///
        /// Step 4.b.1-4: Solve the linear system via sparse cholesky factorization
        ///
        std::unique_ptr<timer_measurement_t> timer(new timer_measurement_t("cusolversp::linear-solve", stream, use_timer));
        const magnitude_type tol(epsilon*1e3);
        int singularity(0), reorder(0);            
        CUSOLVER_CHECK(cusolverSpXcsrlsvchol(cusolversp,
                                             m, nnz,
                                             descriptor,
                                             reinterpret_cast<const cuda_value_type*>(d_ax),
                                             reinterpret_cast<const int*>(d_ap),
                                             reinterpret_cast<const int*>(d_aj),
                                             reinterpret_cast<const cuda_value_type*>(d_b),
                                             tol, reorder,
                                             reinterpret_cast<cuda_value_type*>(d_x),
                                             &singularity));        
      } else {
        ///
        /// Step 4.b.1: Analyze fills
        ///
        {
          std::unique_ptr<timer_measurement_t> timer(new timer_measurement_t("cusolversp::analyze", stream, use_timer));                        
          CUSOLVER_CHECK(cusolverSpXcsrcholAnalysis(cusolversp,
                                                    m, nnz, descriptor,
                                                    d_ap, d_aj,
                                                    chol_info));
        }
        
        ///
        /// Step 4.b.2: Factorize
        ///
        size_t not_used, workspace_in_bytes;
        CUSOLVER_CHECK(cusolverSpXcsrcholBufferInfo(cusolversp,
                                                    m, nnz, descriptor,
                                                    reinterpret_cast<cuda_value_type*>(d_ax),
                                                    d_ap, d_aj,
                                                    chol_info,
                                                    &not_used, &workspace_in_bytes));
        
        void *d_workspace;
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_in_bytes));
        CUDA_CHECK(cudaStreamSynchronize(stream));      
        {
          std::unique_ptr<timer_measurement_t> timer(new timer_measurement_t("cusolversp::factorize", stream, use_timer));
          CUSOLVER_CHECK(cusolverSpXcsrcholFactor(cusolversp,
                                                  m, nnz, descriptor,
                                                  reinterpret_cast<cuda_value_type*>(d_ax),
                                                  d_ap, d_aj,
                                                  chol_info,
                                                  d_workspace));
        }
        
        ///
        /// Step 4.b.3: Solve
        ///
        {
          std::unique_ptr<timer_measurement_t> timer(new timer_measurement_t("cusolversp::solve", stream, use_timer));
          CUSOLVER_CHECK(cusolverSpXcsrcholSolve(cusolversp,
                                                 m,
                                                 reinterpret_cast<cuda_value_type*>(d_b),
                                                 reinterpret_cast<cuda_value_type*>(d_x),
                                                 chol_info,
                                                 d_workspace));
        }        
        
        ///
        /// Step 4.a.4: Free workspace
        ///
        CUDA_CHECK(cudaStreamSynchronize(stream));      
        CUDA_CHECK(cudaFree(d_workspace));
      }
    }

    /// 
    /// Step 5: Transfer data from device to host
    /// 
    CUDA_CHECK(cudaMemcpyAsync(h_x.data(), d_x, sizeof(value_type)*h_x.size(), cudaMemcpyDeviceToHost, stream));        
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /// permute solution; x := P^{-1} x
    if (!use_cudss) {
      std::vector<value_type> h_tmp = h_x;
      std::transform(h_peri.begin(), h_peri.end(), h_x.begin(), 
                     [&h_tmp](ordinal_type idx) { return h_tmp[idx]; });
      h_x = h_tmp;      
    }
    
    ///
    /// Step 6: Compute residual and check the solution
    ///
    std::vector<value_type> h_r(h_x.size());
    compute_r_is_Ax_minus_b(m, n,
                            h_ap, h_aj, h_ax,
                            h_x, 
                            h_b,
                            h_r);
    if (verbose) {
      show_vector("r", m, h_r);      
    }

    const double norm_A = compute_norm(h_ax);
    const double norm_r = compute_norm(h_r);

    std::cout << "-- |A| = " << norm_A << ", "
              << "|Ax-b| = " << norm_r << ", "
              << "|Ax-b|/|A| = " << (norm_r / norm_A) << "\n";


    ///
    /// Step 7: Finalize cuda instances and memory
    ///
    cuda_free_memory();

    cusolver_finalize();
    cudss_finalize();
    cuda_finalize();            
  } catch (std::exception &e) {
    std::cerr << "Error: exception is caught: \n" << e.what() << "\n";
  }

  return 0;
}

#endif
