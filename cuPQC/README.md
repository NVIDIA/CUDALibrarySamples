# cuPQC Library - API Examples

All examples are shipped within [cuPQC Software Development Kit](https://developer.nvidia.com/cupqc-downloads).

## Description

This folder demonstrates how to use the libraries stored in the cuPQC SDK: cuPQC and cuHash.

* [cuPQC download page](https://developer.nvidia.com/cupqc-downloads)
* [cuPQC API documentation](https://docs.nvidia.com/cuda/cupqc/index.html)

## Requirements

* [cuPQC SDK](https://developer.nvidia.com/cupqc-downloads)
* [See cuPQC SDK requirements](https://docs.nvidia.com/cuda/cupqc/requirements.html)
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build
Download and expand the cuPQC SDK then use the MakeFile located in this directory. Make sure that you set the `CUPQC_DIR` to the location of your expanded cuPQC SDK folder.

```
export CUPQC_DIR=<your_path_to_cupqc>
make
// Run
./example_ml_kem
./example_ml_dsa
./example_sha2
./example_sha3
./example_poseidon2
./example_merkle
```

## Examples
There is a ML-KEM and a ML-DSA example in this directory, these demonstrate the usage for the cuPQC library, and requires `libcupqc.a`. 
There are also SHA2, SHA3, Poseidon2 and Merkle Tree examples that demonstrate the usage of the cuHash library, these require `libcuhash.a`. 
For the detailed descriptions of the cuPQC API please visit [cuPQC SDK Docs](https://docs.nvidia.com/cuda/cupqc/index.html) section of the cuPQC documentation.

