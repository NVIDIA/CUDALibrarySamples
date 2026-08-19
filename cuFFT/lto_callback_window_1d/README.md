# cuFFT LTO callback examples

## Description

This directory contains examples demonstrating the use of cuFFT callbacks with both legacy and LTO (Link-Time Optimization) approaches. Two different use cases are provided: windowing and zero-padding.

### Windowing Examples

In the windowing examples, we apply a low-pass filter to a batch of signals in the frequency domain. Specifically, the sample code creates a forward (R2C, Real-To-Complex) plan and an inverse (C2R, Complex-To-Real) plan. The low-pass filter is implemented via a **load callback** in the inverse plan, where values under a certain threshold (specified by the user as a window dimension) are loaded normally, and every other value is zeroed.

Three versions of the windowing code are provided:
* `r2c_c2r_windowing_lto_callback_example.cpp` contains the sample code using a load callback with LTO to compute the window function. The LTO callback is compiled offline using nvcc.
* `r2c_c2r_windowing_lto_nvrtc_callback_example.cpp` contains the sample code using a load callback with LTO to compute the window function. The LTO callback is compiled at runtime using NVRTC.
* `r2c_c2r_windowing_legacy_callback_example.cu` contains the sample code using a 'legacy' (non-LTO) load callback to compute the window function. The callback does not use LTO and requires separate device linking against the cuFFT static library.

### Padding Examples

In the padding examples, we demonstrate the use of **both load and store callbacks** to implement zero-padding. The load callback is applied on the forward (R2C) plan to pad the input with zeros, and the store callback is applied on the inverse (C2R) plan to truncate the output to the original size and normalize.

Two versions of the padding code are provided:
* `r2c_c2r_padding_lto_callback_example.cpp` contains the sample code using LTO callbacks to implement zero-padding. The LTO callback is compiled offline using nvcc.
* `r2c_c2r_padding_legacy_callback_example.cu` contains the sample code using legacy callbacks to implement zero-padding. The callback does not use LTO and requires separate device linking against the cuFFT static library.

Other source files included:
* `r2c_c2r_windowing_lto_callback_device.cu` contains the callback device function used in the windowing LTO and LTO + NVRTC examples.
* `r2c_c2r_padding_lto_load_callback_device.cu` contains the load callback device function used in the padding LTO example.
* `r2c_c2r_padding_lto_store_callback_device.cu` contains the store callback device function used in the padding LTO example.
* `r2c_c2r_windowing_reference.cu` contains the code used as reference for the windowing samples. The reference computes the window function using a separate kernel, rather than callbacks.
* `r2c_c2r_padding_reference.cu` contains the code used as reference for the padding samples. The reference computes the zero-padding using separate kernels, rather than callbacks.
* `nvrtc_helper.h` contains the required code to do runtime compilation of the LTO callback using NVRTC.
* `common.cpp` and `common.h` include some helper functions, like methods to perform the initialization of the signal in the time domain.

## Supported SM Architectures

All GPUs supported by the CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux
The sample should support Windows with minimal changes to the build system

## Supported CPU Architecture

x86_64

## CUDA APIs involved
- [cufftCreate](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftcreate)
- [cufftXtSetJITCallback](https://docs.nvidia.com/cuda/cufft/index.html#cufftxtsetjitcallback)
- [cufftMakePlan1d](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftmakeplan1d)
- [cufftExecR2C](https://docs.nvidia.com/cuda/cufft/index.html#functions-cufftexecr2c-and-cufftexecd2z)
- [cufftExecC2R](https://docs.nvidia.com/cuda/cufft/index.html#functions-cufftexecc2r-and-cufftexecz2d)
- [cufftDestroy](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftdestroy)

## Requirements
- CUDA Toolkit 12.6 Update 2 or newer; specifically, [cuFFT](https://docs.nvidia.com/cuda/cufft/index.html#cufft-callback-routines), [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html) and [nvJitLink](https://docs.nvidia.com/cuda/nvjitlink/index.html).
- [cuFFT Callback Routines with LTO support](https://docs.nvidia.com/cuda/cufft/index.html#lto-load-and-store-callback-routines)

## Building the examples on Linux

### Make
```
# Set path to CUDA Toolkit, default is /usr/local/cuda
export CUDA_PATH=/PATH/TO/CUDA

# Set target architecture if desired, default is SM50 with PTX JIT to any newer architecture
export CUDA_ARCH=TARGET_ARCH

make
```
**NOTE** Refer to the Makefile to configure other parameters of the build, like the compiler used.

### CMake
```
mkdir my_cmake_build && cd my_cmake_build
cmake ..
make
```

## Running the examples

### Windowing examples
```
./bin/r2c_c2r_windowing_lto_callback_example
./bin/r2c_c2r_windowing_lto_nvrtc_callback_example
./bin/r2c_c2r_windowing_legacy_callback_example
```

### Padding examples
```
./bin/r2c_c2r_padding_lto_callback_example
./bin/r2c_c2r_padding_legacy_callback_example
```

Sample of output (windowing)

```
$ ./bin/r2c_c2r_windowing_lto_callback_example 
Transforming signal cufftExecR2C
Transforming signal cufftExecC2R
Transforming reference cufftExecR2C
Transforming reference cufftExecC2R
L2 error: 0.000000e+00
```

Sample of output (padding)

```
$ ./bin/r2c_c2r_padding_lto_callback_example 
Transforming signal cufftExecR2C with padding load callback
Transforming signal cufftExecC2R with padding store callback
Transforming reference cufftExecR2C
Transforming reference cufftExecC2R
L2 error: 0.000000e+00
```

## Troubleshooting
### I am getting an error with the message: "error: ‘cufftXtSetJITCallback’ was not declared in this scope".
Please, make sure you are using cuFFT from CUDA Toolkit 12.6 Update 2 or newer. LTO callbacks are not available on older versions of cuFFT.

### I am getting an error with the message: "undefined reference to 'cufftXtSetJITCallback'".
Similarly to the error above, please ensure you are using cuFFT from CUDA Toolkit 12.6 Update 2 or newer when linking the code.

### My existing callback works fine without LTO (i.e. using the old API `cufftXtSetCallback`) but planning fails with `CUFFT_INTERNAL_ERROR` or similar when attempting to use LTO callbacks via `cufftXtSetJITCallback`.
There could be several reasons for this; the most likely cause is that inlining the callback function in the cuFFT kernel causes the kernel to run out of resources. Please, contact us with your feedback or use case so we can investigate further. In future updates, we will improve the logging of runtime linking to help users debug their application.

