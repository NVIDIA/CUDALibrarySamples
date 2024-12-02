# cuPQC Library - API Examples

All examples are shipped within [cuPQC package](https://developer.nvidia.com/cupqc-downloads).

## Description

This folder demonstrates cuPQC APIs usage.

* [cuPQC download page](https://developer.nvidia.com/cupqc-downloads)
* [cuPQC API documentation](https://docs.nvidia.com/cuda/cupqc/index.html)

## Requirements

* [cuPQC package](https://developer.nvidia.com/cupqc-downloads)
* [See cuPQC requirements](https://docs.nvidia.com/cuda/cupqc/requirements.html)
* Linux system with installed NVIDIA drivers
* NVIDIA GPU of Volta (SM70) or newer architecture

## Build
Download and expand the cuPQC package then use the MakeFile located in this directory. Make sure that you set the `CUPQC_DIR` to the location of your expanded cuPQC package.

```
export CUPQC_DIR=<your_path_to_cupqc>
make
// Run
./example_ml_kem
./example_ml_dsa
```

## Examples

There is a ML-KEM and a ML-DSA example in this directory. 
For the detailed descriptions of the examples please visit [Examples](https://docs.nvidia.com/cuda/cupqc/examples.html) section of the cuCPQ documentation.

