# cuFFT LTO callback examples

## Description

In this example, we apply a low-pass filter to a batch of signals in the frequency domain. Specifically, the sample code creates a forward (R2C, Real-To-Complex) plan and an inverse (C2R, Complex-To-Real) plan. The low-pass filter is implemented via a load callback in the inverse plan, where values under a certain threshold (specified by the user as a window dimension) are loaded normally, and every other value is zeroed.

Three versions of the code are provided:
* `r2c_c2r_lto_callback_example.cpp` contains the sample code using a load callback with LTO to compute the window function. The LTO callback is compiled offline using nvcc.
* `r2c_c2r_lto_nvrtc_callback_example.cpp` contains the sample code using a load callback with LTO to compute the window function. The LTO callback is compiled at runtime using NVRTC.
* `r2c_c2r_callback_example.cu` contains the sample code using a 'standard' (non-LTO) load callback to compute the window function. The callback does not use LTO and requires separate device linking against the cuFFT static library.

Other source files included:
* `r2c_c2r_lto_callback_device.cu` contains the callback device function used in the LTO and LTO + NVRTC examples.
* `r2c_c2r_reference.cu` contains the code used as reference for the samples. The reference computes the window function using a separate kernel, rather than callbacks.
* `nvrtc_helper.h` contains the required code to do runtime compilation of the LTO callback using NVRTC.
* `common.cpp` and `common.h` include some helper functions, like methods to perform the initialization of the signal in the time domain..

## Supported SM Architectures

All GPUs supported by CUDA Toolkit (https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux
The sample should support Windows with minimal changes to the build system

## Supported CPU Architecture

x86_64

## CUDA APIs involved
- [cufftCreate](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftcreate)
- [cufftXtSetJITCallback]()
- [cufftMakePlan1d](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftmakeplan1d)
- [cufftExecR2C](https://docs.nvidia.com/cuda/cufft/index.html#functions-cufftexecr2c-and-cufftexecd2z)
- [cufftExecC2R](https://docs.nvidia.com/cuda/cufft/index.html#functions-cufftexecc2r-and-cufftexecz2d)
- [cufftDestroy](https://docs.nvidia.com/cuda/cufft/index.html#function-cufftdestroy)

## Requirements
- CUDA Toolkit 12.2 or newer.
- [cuFFT EA with LTO callback support](https://developer.nvidia.com/cufftea) (version 11.1.0.XX or similar).

The examples assume that the cuFFT library binaries in the CUDA Toolkit 'lib64' folder have been replaced with the cuFFT EA with LTO binaries. Similarly, the `cufftXt.h` header file located in the CUDA Toolkit 'include' folder should be replaced with the `cufftXt.h` header shipped with the cuFFT EA with LTO package.

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
```
./bin/r2c_c2r_lto_callback_example
./bin/r2c_c2r_lto_nvrtc_callback_example
./bin/r2c_c2r_callback_example
```

**NOTE** Using NVRTC to do runtime compilation of the callback with LTO will print an error similar to `error: Binary format for key='0', ident='' is not recognized`. In general, it is safe to ignore this error. This is a known issue of the nvJitLink library (on which cuFFT with LTO callbacks depends) and will be fixed in a future release.

Sample of output

```
$ ./bin/r2c_c2r_lto_callback_example 
Transforming signal cufftExecR2C
Transforming signal cufftExecC2R
Transforming reference cufftExecR2C
Transforming reference cufftExecC2R
L2 error: 0.000000e+00
```

## Troubleshooting
### I am getting an error with the message: "error: ‘cufftXtSetJITCallback’ was not declared in this scope".
Please, make sure you are including the correct `cufftXt.h` header, shipped with the cuFFT LTO preview package. This header can be dropped-in as replacement in the CUDA Toolkit 'include' folder.

### I am getting an error with the message: "undefined reference to 'cufftXtSetJITCallback'".
Similarly to the error above, please ensure you are linking against the correct version of `libcufft.so` / `libcufft_static.a` shipped in the cuFFT LTO preview package. The libraries can be dropped-in as replacement in the CUDA Toolkit 'lib64' folder.

### My existing callback works fine without LTO (i.e. using the old API `cufftXtSetCallback`) but planning fails with `CUFFT_INTERNAL_ERROR` or similar when attempting to use LTO callbacks via `cufftXtSetJITCallback`.
There could be several reasons for this; the most likely cause is that inlining the callback function in the cuFFT kernel causes the kernel to run out of resources. Please, contact us with your feedback or use case so we can investigate further. In future updates, we will improve the logging of runtime linking to help users debug their application.

