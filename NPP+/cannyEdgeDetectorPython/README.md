# Canny Edge Detection with NPP+ in Python

This sample demonstrates how to use the nppiFilterCannyBorder_8u_C1R_Ctx function from NVIDIA's NPP+ (NVIDIA Performance Primitives) for Canny edge detection. The function requires a single-channel 8-bit grayscale image as input. Color images should be converted using nppiColorToGray() or nppiRGBToGray() prior to processing.

The project compares performance between NPP+ and OpenCV implementations of Canny edge detection across various resolutions, focusing on execution time and scalability, particularly when using PyTorch tensors on the GPU.


## Canny Edge Detection Benchmark: OpenCV vs NVIDIA NPP+

This benchmark evaluates the performance of Canny edge detection using two implementations:

- **OpenCV**: CPU-based with GPU-CPU-GPU data transfer overhead
- **NVIDIA NPP+ (v0.10.0)**: GPU-accelerated, integrated with PyTorch tensors for optimized in-GPU processing

The benchmark is tailored for high-resolution image analysis (up to 5K) and aims to assess execution time and scalability.

---

## 🚀 Benchmark Overview

The script processes a sample image across multiple resolutions:
- **Resolutions tested**: 320x180 to 5120x2880 (0.06 MP to 14.75 MP)
- **Iterations**: 5 warm-up + 1000 timed iterations per run
- **GPU Used**: NVIDIA RTX A6000
- **NPP+ Used**: v0.10.0

### Techniques Compared
1. **OpenCV** Canny detection with CPU processing + memory transfer overhead
2. **NPP+** Canny detection directly on PyTorch tensors in GPU memory

---

## 📊 Results Summary

### Run #1 (1000 iterations)

| # | Resolution | Megapixels | OpenCV (CPU + transfer) (ms) | NPP (GPU optimized) (ms) | Speedup |
|---|------------|------------|------------------------------|--------------------------|---------|
| 0 | 320x180    | 0.0576     | 0.2127704620361328           | 0.06665971196070314      | 3.19x   |
| 1 | 640x360    | 0.2304     | 0.511603593826294            | 0.0730036159530282       | 7.00x   |
| 2 | 800x600    | 0.48       | 0.7419486045837402           | 0.07595692797377705      | 9.76x   |
| 3 | 1280x720   | 0.9216     | 1.183044672012329            | 0.08218614405393601      | 14.39x  |
| 4 | 1920x1080  | 2.0736     | 1.8175594806671143           | 0.12558924829214813      | 14.47x  |
| 5 | 2560x1440  | 3.6864     | 2.936678409576416            | 0.18150592005252839      | 16.17x  |
| 6 | 3840x2160  | 8.2944     | 6.66689395904541             | 0.34022380778193473      | 19.59x  |
| 7 | 5120x2880  | 14.7456    | 12.06825590133667            | 0.5401814390420914       | 22.34x  |

### Run #2 (1000 iterations)

| # | Resolution | Megapixels | OpenCV (CPU + transfer) (ms) | NPP (GPU optimized) (ms) | Speedup |
|---|------------|------------|------------------------------|--------------------------|---------|
| 0 | 320x180    | 0.0576     | 0.20667243003845215          | 0.06370281605422497      | 3.24x   |
| 1 | 640x360    | 0.2304     | 0.5194730758666992           | 0.07735942399874329      | 6.71x   |
| 2 | 800x600    | 0.48       | 0.8309378623962402           | 0.07778342404961586      | 10.68x  |
| 3 | 1280x720   | 0.9216     | 1.1930086612701416           | 0.08126364810019732      | 14.68x  |
| 4 | 1920x1080  | 2.0736     | 1.9724104404449463           | 0.12417126397043467      | 15.88x  |
| 5 | 2560x1440  | 3.6864     | 2.9341330528259277           | 0.18224953599274157      | 16.09x  |
| 6 | 3840x2160  | 8.2944     | 6.796700954437256            | 0.34163942405581477      | 19.89x  |
| 7 | 5120x2880  | 14.7456    | 11.827062368392944           | 0.5242507843375206       | 22.55x  |
``

📌 *OpenCV times include full CPU-GPU-CPU transfers, whereas NPP+ timings reflect fully in-GPU execution.*

---

## 🔍 Key Insights

- **NPP+ outperforms OpenCV** consistently across all resolutions
- **Performance gap widens** with higher image resolutions
- **NPP+ scales efficiently**, especially beneficial for 4K+ workflows
- **PyTorch tensor integration** enhances speed by eliminating transfer overhead

---

## 💡 Best Practices for NPP+ with PyTorch

Ensure tensors are GPU-ready and memory-contiguous:
```python
torch_img = torch.from_numpy(img).cuda().contiguous()
```

---

## 🧪 How It Works

1. Load and resize a sample image to various resolutions
2. Upload grayscale images to GPU
3. Measure OpenCV performance (including transfer cost)
4. Measure NPP+ performance with in-GPU data
5. Calculate performance statistics and save results as CSV

Run the benchmark:
```bash
python3 cannyEdgeDetector_combine.py
```

---

## ⚙️ Requirements

- NVIDIA GPU with CUDA support
- NVIDIA Performance Primitives (NPP+) v0.10.0+
- Python libraries:
  - `torch`
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `tabulate`

---

## ✅ Conclusion

NPP+ provides substantial speedups over OpenCV for Canny edge detection, particularly for high-resolution images where memory transfers become a bottleneck. By combining NPP+ with PyTorch tensors, real-time or large-scale image processing pipelines can benefit from minimal latency and higher throughput.

For high-resolution vision applications (e.g., 4K+), NPP+ is the preferred method.


