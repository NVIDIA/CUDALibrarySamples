# CUDA Library Samples

The **CUDA Library Samples** repository contains various examples that demonstrate the use of GPU-accelerated libraries in CUDA. These libraries enable high-performance computing in a wide range of applications, including math operations, image processing, signal processing, linear algebra, and compression. The samples included cover:

- **Math and Image Processing Libraries**
- **cuBLAS** (Basic Linear Algebra Subprograms)
- **cuTENSOR** (Tensor Linear Algebra)
- **cuSPARSE** (Sparse Matrix Operations)
- **cuSOLVER** (Dense and Sparse Solvers)
- **cuFFT** (Fast Fourier Transform)
- **cuRAND** (Random Number Generation)
- **NPP** (Image and Video Processing)
- **nvJPEG** (JPEG Encode/Decode)
- **nvCOMP** (Data Compression)
- **and more...**

## About

The CUDA Library Samples are provided by NVIDIA Corporation as Open Source software, released under the 3-clause "New" BSD license. These examples showcase how to leverage GPU-accelerated libraries for efficient computation across various fields.

For more information on the available libraries and their uses, visit [GPU Accelerated Libraries](https://developer.nvidia.com/gpu-accelerated-libraries).

## Library Examples

Explore the examples of each CUDA library included in this repository:

- [cuBLAS - GPU-accelerated basic linear algebra (BLAS) library](cuBLAS/)
- [cuBLASLt - Lightweight BLAS library](cuBLASLt/)
- [cuBLASMp - Multi-process BLAS library](cuBLASMp/)
- [cuBLASDx - Device-side BLAS extensions](MathDx/cuBLASDx/)
- [cuDSS - GPU-accelerated linear solvers](cuDSS/)
- [cuFFT - Fast Fourier Transforms](cuFFT/)
- [cuFFTMp - Multi-process FFT](cuFFTMp/)
- [cuFFTDx - Device-side FFT extensions](MathDx/cuFFTDx/)
- [cuRAND - Random number generation](cuRAND/)
- [cuSOLVER - Dense and sparse direct solvers](cuSOLVER/)
- [cuSOLVERMp - Multi-process solvers](cuSOLVERMp/)
- [cuSOLVERSp2cuDSS - Transition example to cuDSS](cuSOLVERSp2cuDSS/)
- [cuSPARSE - BLAS for sparse matrices](cuSPARSE/)
- [cuSPARSELt - Lightweight BLAS for sparse matrices](cuSPARSELt/)
- [cuTENSOR - Tensor linear algebra library](cuTENSOR/)
- [cuTENSORMg - Multi-GPU tensor linear algebra](cuTENSORMg/)
- [NPP - GPU-accelerated image, video, and signal processing functions](NPP/)
- [NPP+ - C++ extensions for NPP](NPP+/)
- [nvJPEG - High-performance JPEG encode/decode](nvJPEG/)
- [nvJPEG2000 - JPEG2000 encoding/decoding](nvJPEG2000/)
- [nvTIFF - TIFF encoding/decoding](nvTIFF/)
- [nvCOMP - Data compression and decompression](nvCOMP/)

Each sample provides a practical use case for how to apply these libraries in real-world scenarios, showcasing the power and flexibility of CUDA for a wide variety of computational needs.

## Additional Resources

For more information and documentation on CUDA libraries, please visit:

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Developer Zone](https://developer.nvidia.com/)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

## License

The CUDA Library Samples are distributed under the 3-clause "New" BSD license. For more details, refer to the license terms below:

## Copyright

Copyright (c) 2022-2024 NVIDIA CORPORATION AND AFFILIATES.  All rights reserved.

```
  Redistribution and use in source and binary forms, with or without modification, are permitted
  provided that the following conditions are met:
      * Redistributions of source code must retain the above copyright notice, this list of
        conditions and the following disclaimer.
      * Redistributions in binary form must reproduce the above copyright notice, this list of
        conditions and the following disclaimer in the documentation and/or other materials
        provided with the distribution.
      * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
        to endorse or promote products derived from this software without specific prior written
        permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
  STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```