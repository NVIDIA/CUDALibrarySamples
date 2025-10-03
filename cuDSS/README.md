# cuDSS Library

## Description

This folder demonstrates main cuDSS APIs usage.

Note: samples are updated to work with the latest version of cuDSS available (currently: 0.7.0) and might be incompatible with older versions.
For further details, please check the release notes in the official documentation.

[cuDSS Documentation](https://docs.nvidia.com/cuda/cudss/index.html)

## cuDSS Samples

* [Simple workflow for solving a real-valued sparse linear system](simple/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system without any advanced features

* [Simple workflow for solving a complex-valued sparse linear system](simple_complex/)

    The sample demonstrates how to use cuDSS for solving a complex sparse linear system without any advanced features

* [Hybrid host/device memory mode](simple_hybrid_memory_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the hybrid host/device memory mode

* [Multi-gpu multi-node (MGMN) mode](simple_mgmn_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multi-GPU multi-node mode (available since cudss 0.3.0)

    Note: building MGMN mode samples require extra options for the communication backend at build and runtime, and must be run with mpirun

* [Getter/setter APIs for cudssConfig, cudssData and cudssHandle objects](simple_get_set/)

    The sample extends the previous code to demonstrate how extra settings can be applied to solving sparse linear
    systems and how to retrieve extra information from the solver

* [Device memory handler APIs for cudssDeviceMemHandler_t defined device memory pools or allocators](simple_memory_handler/)

    The sample demonstrates how to use cuDSS device memory handler APIs (available since cudss 0.2.0) for providing user-defined device memory pools
    or allocators

* [Non-uniform batch API](simple_batch/)

    The sample demonstrates how to use cuDSS batch APIs (available since cudss 0.4.0) for solving a non-uniform batch of linear systems where matrices and righthand sides
    can all be different

* [Hybrid execution mode](simple_hybrid_execute_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the hybrid execute mode (available since cudss 0.5.0)

* [Multi-threaded (MT) mode](simple_multithreaded_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multithreading (MT) mode (available since cudss 0.5.0)

* [Sparse matrix interactions](simple_sparse_matrix_helpers/)

    The sample demonstrates how to interact with sparse matrices in cuDSS

* [Dense matrix interactions](simple_dense_matrix_helpers/)

    The sample demonstrates how to interact with dense matrices in cuDSS

* [Batch matrix interactions](simple_batch_sparse_matrix_helpers/)

    The sample demonstrates how to interact with batched systems in cuDSS (available since cudss 0.5.0)

* [Batch system solving](simple_uniform_batch/)

    The sample demonstrates how to use cuDSS for solving a uniform batched system (available since cudss 0.6.0)

* [Extracting reordering information](simple_reordering_phase/)

    The sample demonstrates how to use cuDSS to extract the reordering information it uses internally immediately after it is generated (available since cudss 0.6.0)

* [Multi-gpu multi-node (MGMN) mode with a distributed matrix](simple_mgmn_distributed_matrix/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multi-GPU multi-node mode with a distributed system matrix (available since cudss 0.6.0)

    Note: building MGMN mode samples require extra options for the communication backend at build and runtime, and must be run with mpirun

* [Multi-gpu single-node (MG) mode](simple_mg_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multi-GPU mode (without any distributed communication backend) (available since cudss 0.7.0)

* [Schur complement computation](simple_schur_complement/)

    The sample demonstrates how to use cuDSS for computing the Schur complement matrix (and how it can be used for solving the full system) (available since cudss 0.7.0)

* [Residual computation](simple_residual/)

    The sample demonstrates how one can compute the relative residual norm to estimate accuracy of cuDSS (available since cudss 0.7.0)

* [Helper for testing the communication layer library](test_communication_layer/)

    The sample demonstrates how one can check functional health of a distributed communication backend and the associated communication layer library of cuDSS (available since cudss 0.7.0)

* [Helper for testing the threading layer library](test_threading_layer/)

    The sample demonstrates how one can check functional health of a threading backend and the associated threading layer library of cuDSS (available since cudss 0.7.0)
