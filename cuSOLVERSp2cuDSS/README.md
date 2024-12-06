# Transition examples from cuSOLVERSp/Rf to cuDSS  

## Description

cuSOLVERSp/Rf are deprecated and will be removed in a future major release.
To continue using the sparse direct methods provided from these packages,
please use [cuDSS](https://developer.nvidia.com/cudss) library instead
for better performance and support. Code samples in this directory demonstrate the
transition procedure from cuSOLVERSp/Rf to cuDSS. 

Note: while the cuDSS team plans to extend the coverage, it is not planned to cover
the 100% of functionalities provided from cuSolverSp/Rf.

Reference:
* [cuSOLVER Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
* [cuDSS Documentation](https://docs.nvidia.com/cuda/cudss/index.html)
* [cuDSS Examples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuDSS)

# Deprecated cuSOLVERSp/Rf APIs and its transition to cuDSS

A brief summary of deprecated APIs and its transition to cuDSS are illustrated in the below.
For more detailed and complete transition of the APIs, we recommend to check the example
codes `cuSolverSp2cuDSS.hpp` and `cuSolverRf2cuDSS.hpp`.

```{verbatim}
+-----------------+-------------------------------------------+---------------------------------------------------------+
| Workflow        | cuSolverSp/Rf (deprecated)                | cuDSS (new)                                             |
+=================+===========================================+=========================================================+
| Initialize      | cusolverSpCreate                          | cudssCreate                                             |
|                 | cusolverSpCreateCsr{chol,lu}Info[Host]    | cudssConfigCreate                                       |
|                 | cusparseCreateMatDescr                    | cudssDataCreate                                         |
|                 | cusolverRfCreate                          |                                                         |
+-----------------+-------------------------------------------+---------------------------------------------------------+
| Input matrix    | cusparseSetMatType                        | cudssMatrixCreateCsr                                    |
| handling        | cusparseSetMatIndexBase                   | cudssMatrixCreateDn                                     |
|                 |                                           | cudssMatrixDestroy                                      |
+-----------------+-------------------------------------------+---------------------------------------------------------+
| Reorder &       | cusolverSpXcsrmetisndHost                 | cudssConfigSet(config, CUDSS_REORDERING_ALG, ...)       |
| Analyze         | cusolverSpXcsr{chol,lu}Analysis[Host]     | cudssExecute(cudss, CUDSS_PHASE_ANALYSIS, ...)          |
+-----------------+-------------------------------------------+---------------------------------------------------------+
| Factorize       | cusolverSpXcsr{chol,lu}BufferInfo[Host]'  | cudssExecute(cudss, CUDSS_PHASE_FACTORIZATION, ...)     |
|                 | cusolverSpXcsr{chol,lu}Factor[Host]'      |                                                         |
+-----------------+-------------------------------------------+---------------------------------------------------------+
| Solve           | cusolverSpXcsr{chol,lu}Solve[Host]        | cudssExecute(cudss, CUDSS_PHASE_SOLVE, ...)             |
+-----------------+-------------------------------------------+---------------------------------------------------------+
| Refactorize &   | cusolverRfCreate                          | cudssMatrixSetValues                                    |
| Solve           | cusolverSpXcsrluNnzHost                   | cudssExecute(cudss, CUDSS_PHASE_REFACTORIZATION, ...)   |
|                 | cusolverSpXcsrluExtractHost               | cudssExecute(cudss, CUDSS_PHASE_SOLVE, ...)             |
|                 | cusolverRfSetMatrixFormat                 |                                                         |
|                 | cusolverRfSetupHost                       |                                                         |
|                 | cusolverRfAnalyze                         |                                                         |
|                 | cusolverRfResetValues                     |                                                         |
|                 | cusolverRfRefactor                        |                                                         |
|                 | cusolverRfSolve                           |                                                         |
+-----------------+-------------------------------------------+---------------------------------------------------------+
| Finalize        | cusolverRfDestroy                         | cudssDataDestroy                                        |
|                 | cusparseDestroyMatDescr                   | cudssDataDestroy                                        |
|                 | cusolverSpDestroyCsr{chol,lu}Info[Host]   | cudssConfigDestroy                                      |
|                 | cusolverSpDestroy                         | cudssDestroy                                            |
+-----------------+-------------------------------------------+---------------------------------------------------------+
```

# Building examples

This example is compatible to cuDSS 0.3.0 and CUDA Toolkit 12.x and higher.
Download and install the packages from

* [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
* [developer.nvidia.com/cudss-downloads](https://developer.nvidia.com/cudss-downloads).


To demonstrate the cmake build procedure, we use the following environment variables.

```sh
    $ export CUDSS_ROOT=/cudss/install/path
    $ export SAMPLE_SOURCE_PATH=/cudalibrarysamples/cuSOLVERSp2cuDSS
    $ export SAMPLE_BUILD_PATH=/your/sample-build/path
```

Configure and build the examples using cmake.

```sh
    $ mkdir -p ${SAMPLE_BUILD_PATH}
    $ cd ${SAMPLE_BUILD_PATH}
    $ cmake -DCMAKE_BUILD_TYPE=Release -DCUDSS_ROOT=${CUDSS_ROOT} -S ${SAMPLE_SOURCE_PATH}
    $ make
```

# Running examples

After successful sample build, it generates the executables in your build directory.

* `cuSolverRf2cuDSS_<float,double,scomplex,dcomplex>`: sparse Cholesky factorization using cuSOLVERSp and cuDSS.
* `cuSolverRf2cuDSS_double`: sparse LU factorization and refactorization using cuSOLVERSp/Rf and cuDSS.

The usage of each sample code can be listed with a `--help` command line argument.
An example usage is shown in the below.

```{verbatime}
    $ ./cuSolverSp2cuDSS_double -h
    usage: ./cuSolverSp2cuDSS_double --solver <cudss;cusolver> --file <filename> --timer --verbose 
      --solver : select a linear solver; cudss, cusolver
      --file : sparse matrix input as a matrix market format
      --single-api,-s : use a single api for linear solve if available; 
      --timer,-t : enable timer to measure solver phases
      --verbose,-v : verbose flag
      --help,-h : print usage
```

Run the example code with a matrix market input file provided in this sample. 

Outputs from cuSolverSp2cuDSS:

```{verbatim}
    $ ./cuSolverSp2cuDSS_double --solver cusolver --file ${SAMPLE_SOURCE_PATH}/test_real.mtx 
    -- commandline input
       solver: cusolversp, cholesky
       filename: test_real.mtx
       mode: phase-separated api e.g., analysis, factorize, solve
       timer: disabled
       verbose: 0
    -- read matrixmarket file
    -- |A| = 35.6371, |Ax-b| = 1.36877e-15, |Ax-b|/|A| = 3.84087e-17

    $ ./cuSolverSp2cuDSS_double --solver cudss --file ${SAMPLE_SOURCE_PATH}/test_real.mtx 
    -- commandline input
       solver: cudss, hpd
       filename: test_real.mtx
       mode: phase-separated api e.g., analysis, factorize, solve
       timer: disabled
       verbose: 0
    -- read matrixmarket file
    -- |A| = 35.6371, |Ax-b| = 2.60134e-15, |Ax-b|/|A| = 7.29952e-17
```

Outputs from cuSolverRf2cuDSS:

```{verbatim}
    $ ./cuSolverRf2cuDSS_double --solver cusolver --file ${SAMPLE_SOURCE_PATH}/test_real.mtx 
    -- commandline input
       solver: cusolversp, lu host
       filename: test_real.mtx
       timer: disabled
       verbose: 0
    -- read matrixmarket file
    -- |A| = 35.6371, |Ax-b| = 3.39663e-15, |Ax-b|/|A| = 9.53117e-17
    -- the entries of A are modified
    -- A is refactorized
    -- |A| = 35.9739, |Ax-b| = 5.42989e-15, |Ax-b|/|A| = 1.5094e-16
    
    $ ./cuSolverRf2cuDSS_double --solver cudss --file ${SAMPLE_SOURCE_PATH}/test_real.mtx 
    -- commandline input
       solver: cudss, general
       filename: test_real.mtx
       timer: disabled
       verbose: 0
    -- read matrixmarket file
    -- |A| = 35.6371, |Ax-b| = 9.15513e-16, |Ax-b|/|A| = 2.56899e-17
    -- the entries of A are modified
    -- A is refactorized
    -- |A| = 35.9739, |Ax-b| = 2.67377e-15, |Ax-b|/|A| = 7.43253e-17
 ```
 