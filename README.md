[![License](https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)

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

The CUDA Library Samples are provided by NVIDIA Corporation as Open Source software, released under the Apache 2.0 License. These examples showcase how to leverage GPU-accelerated libraries for efficient computation across various fields.

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
- [cuPQC - Post-Quantum Cryptography device library](cuPQC/)
- [cuRAND - Random number generation](cuRAND/)
- [cuSOLVER - Dense and sparse direct solvers](cuSOLVER/)
- [cuSOLVERMp - Multi-process solvers](cuSOLVERMp/)
- [cuSOLVERSp2cuDSS - Transition example from cuSOLVERSp/Rf to cuDSS](cuSOLVERSp2cuDSS/)
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

## Contributing

We welcome contributions to **CUDA Library Samples**. To contribute to **CUDA Library Samples** and make pull requests,
follow the guidelines outlined in the [Contributing](./CONTRIBUTING.md) document.

## License

The CUDA Library Samples are distributed under the Apache 2.0 License. For more details, refer to the LICENSE.md file.

The old code that was originally distributed under the 3-clause "New" BSD license is available at bsd3_main branch and is no longer maintained.
