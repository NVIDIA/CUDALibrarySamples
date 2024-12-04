# cuDSS Library

## Description

This folder demonstrates main cuDSS APIs usage.

Note: samples are updated to work with the latest version of cuDSS available (currently: 0.4.0) and might be imcompatible with older versions. For further details, please check the release notes in the official documentation.

[cuDSS Documentation](https://docs.nvidia.com/cuda/cudss/index.html)

## cuDSS Samples

* [Simple workflow for solving a real-valued sparse linear system](simple/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system without any advanced features

* [Simple workflow for solving a complex-valued sparse linear system](simple_complex/)

    The sample demonstrates how to use cuDSS for solving a complex sparse linear system without any advanced features

* [Simple workflow for solving a real-valued sparse linear system using the hybrid host/device memory mode](simple_hybrid_memory_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the hybrid host/device memory mode

* [Simple workflow for solving a real-valued sparse linear system using the MGMN mode](simple_mgmn_mode/)

    The sample demonstrates how to use cuDSS for solving a real sparse linear system using the multi-GPU multi-node mode (available since cudss 0.3.0)

    Note: building MGMN mode samples require extra options for the communication backend at build and runtime, and must be run with mpirun

* [Getter/setter APIs for cudssConfig, cudssData and cudssHandle objects](get_set/)

    The sample extends the previous code to demonstrate how extra settings can be applied to solving sparse linear
    systems and how to retrieve extra information from the solver

* [Device memory handler APIs for cudssDeviceMemHandler_t defined device memory pools or allocators](memory_handler/)

    The sample demonstrates how to use cuDSS device memory handler APIs (available since cudss 0.2.0) for providing user-defined device memory pools
    or allocators

* [Non-uniform batch API](simple_batch/)

    The sample demonstrates how to use cuDSS batch APIs (available since cudss 0.4.0) for solving a non-uniform batch of linear systems where matrices and righthand sides
    can all be different
