# Canny Edge Detection with NPP+ in Python

This sample demonstrates how to use the nppiFilterCannyBorder_8u_C1R_Ctx function from NVIDIA's NPP+ (NVIDIA Performance Primitives) for Canny edge detection. The function requires a single-channel 8-bit grayscale image as input. Color images should be converted using nppiColorToGray() or nppiRGBToGray() prior to processing.

The project compares performance between NPP+ and OpenCV implementations of Canny edge detection across various resolutions, focusing on execution time and scalability, particularly when using PyTorch tensors on the GPU.

## Overview

The benchmark tests the performance of Canny edge detection using:
1. OpenCV's implementation (with CPU-GPU transfer costs)
2. NPP+ implementation (optimized with PyTorch tensors)

The tests are run on images ranging from 320x180 (0.06 MP) to 5120x2880 (14.75 MP) with 1000 iterations per test to ensure statistical reliability. 
GPU configuration NVIDIA RTX A6000
NPP+ V0.10.0

## Results Summary

### Performance Comparison - Run #1 (1000 iterations)
+---+------------+------------+------------------------------+--------------------------+--------------------+
|   | Resolution | Megapixels | OpenCV (CPU + transfer) (ms) | NPP (GPU optimized) (ms) |      Speedup       |
+---+------------+------------+------------------------------+--------------------------+--------------------+
| 0 |  320x180   |   0.0576   |      0.2127704620361328      |   0.06665971196070314    | 3.19x              |
| 1 |  640x360   |   0.2304   |      0.511603593826294       |    0.0730036159530282    | 7.00x              |
| 2 |  800x600   |    0.48    |      0.7419486045837402      |   0.07595692797377705    | 9.76x              |
| 3 |  1280x720  |   0.9216   |      1.183044672012329       |   0.08218614405393601    | 14.39x             |
| 4 | 1920x1080  |   2.0736   |      1.8175594806671143      |   0.12558924829214813    | 14.47x             |
| 5 | 2560x1440  |   3.6864   |      2.936678409576416       |   0.18150592005252839    | 16.17x             |
| 6 | 3840x2160  |   8.2944   |       6.66689395904541       |   0.34022380778193473    | 19.59x             |
| 7 | 5120x2880  |  14.7456   |      12.06825590133667       |    0.5401814390420914    | 22.34x             |
+---+------------+------------+------------------------------+--------------------------+--------------------+

### Performance Comparison - Run #2 (1000 iterations)
+---+------------+------------+------------------------------+--------------------------+--------------------+
|   | Resolution | Megapixels | OpenCV (CPU + transfer) (ms) | NPP (GPU optimized) (ms) |      Speedup       |
+---+------------+------------+------------------------------+--------------------------+--------------------+
| 0 |  320x180   |   0.0576   |     0.20667243003845215      |   0.06370281605422497    | 3.24x              |
| 1 |  640x360   |   0.2304   |      0.5194730758666992      |   0.07735942399874329    | 6.71x              |
| 2 |  800x600   |    0.48    |      0.8309378623962402      |   0.07778342404961586    | 10.68x             |
| 3 |  1280x720  |   0.9216   |      1.1930086612701416      |   0.08126364810019732    | 14.68x             |
| 4 | 1920x1080  |   2.0736   |      1.9724104404449463      |   0.12417126397043467    | 15.88x             |
| 5 | 2560x1440  |   3.6864   |      2.9341330528259277      |   0.18224953599274157    | 16.09x             |
| 6 | 3840x2160  |   8.2944   |      6.796700954437256       |   0.34163942405581477    | 19.89x             |
| 7 | 5120x2880  |  14.7456   |      11.827062368392944      |    0.5242507843375206    | 22.55x |
+---+------------+------------+------------------------------+--------------------------+--------------------+

*Note: OpenCV time includes both CPU processing and GPU->CPU->GPU transfer costs, as required when data is already on the GPU. NPP+ time is measured with data already on the GPU using PyTorch tensors for optimal performance.*

### Key Findings

1. **Crossover Point**:
   - In Runs #1 and #2, NPP+ is faster than OpenCV at all tested resolutions

2. **Performance Scaling**:
   - OpenCV processing time (including transfer costs) scales approximately linearly with pixel count
   - NPP+ processing time scales more efficiently at higher resolutions
   - At 5K resolution, NPP+ is 22.34x-22.55x faster than OpenCV across different runs

3. **Data Transfer Impact**:
   - Data transfer costs are a significant bottleneck for OpenCV when data is already on the GPU
   - This bottleneck becomes more pronounced at higher resolutions

4. **Reproducibility**:
   - With 1000 iterations, results show good consistency across runs
   - The performance advantage of NPP+ over OpenCV increases with resolution in all runs
   - Minor variability is expected due to system load fluctuations and the inherent variability in GPU operations

5. **PyTorch Tensor Optimization**:
   - Using PyTorch tensors directly with NPP+ provides significant performance improvements
   - This optimization is particularly beneficial for applications that already have data on the GPU

## Best Practices for Using PyTorch Tensors with NPP+

1. **Ensure Contiguous Memory Layout**:
   ```python
   torch_img = torch.from_numpy(img).cuda().contiguous()
   ```

   ```

## Implementation Details

- **OpenCV**: Uses the standard Canny implementation with L2 gradient and thresholds of 36/128
- **NPP+**: Uses optimized parameters from Bayesian optimization (thresholds of 72/256) with L1 gradient
- Each test includes 5 warmup iterations and 1000 measurement iterations to ensure reliable results
- PyTorch CUDA events are used for accurate GPU timing of NPP+ operations
- Time measurements include all relevant processing and data transfer costs

## Conclusion

NVIDIA NPP+ provides significant performance benefits for Canny edge detection at higher resolutions, especially when the CPU-GPU data conversion overhead can be minimized. For applications processing high-resolution images (4K and above), NPP+ offers substantial speedups over OpenCV.

The data transfer overhead is a significant factor at lower resolutions but becomes less important as resolution increases. For real-time applications working with high-resolution images, NPP+ is the recommended choice.

Using PyTorch tensors directly with NPP+ can further improve performance by avoiding unnecessary CPU-GPU transfers. This approach is particularly beneficial for applications that already have data on the GPU, such as deep learning pipelines.

## Running the Benchmark

The benchmark consists of the following main script:

```bash
# Run the benchmark
python3 cannyEdgeDetector_combine.py
```

The script performs the following operations:
1. Loads and resizes the input image to various resolutions
2. Uploads the image data to the GPU
3. Measures OpenCV performance including GPU->CPU->GPU transfer costs
4. Measures NPP+ performance with data already on the GPU
5. Calculates and reports performance statistics
6. Saves the results to a CSV file

## Requirements
- NVIDIA GPU with CUDA
- NPP+ (latest version)
- Python packages:
   - numpy
   - pandas
   - matplotlib
   - tabulate
   - torch
   - opencv-python