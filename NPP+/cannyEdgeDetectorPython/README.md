# NPP+ Canny Edge Detector Demo (Python)

This demo showcases how to use **NVIDIA Performance Primitives (NPP+)** for high-performance Canny edge detection on the GPU using Python and PyTorch tensors.

---

## 🚀 Overview

This example demonstrates:
- Efficient use of `nppiFilterCannyBorder_8u_C1R_Ctx` from NPP+
- Direct GPU memory access with PyTorch tensors (no unnecessary transfers)
- Real-time edge detection across high-resolution images

### Highlights

- Resolutions tested: 320x180 to 5120x2880 (up to 5K)
- GPU: NVIDIA RTX A6000
- Optimized for in-GPU execution (no CPU bottlenecks)
- Parameters: 5 warm-up + 1000 timed iterations

---

## 🛠️ How to Run

### 1. Install Dependencies

```bash
pip install torch opencv-python numpy pandas tabulate
```

### 2. Set Library Path

Ensure your `LD_LIBRARY_PATH` includes the directory with `libnpp_plus_if.so`:
```bash
export LD_LIBRARY_PATH=/path/to/libnppif:$LD_LIBRARY_PATH
```

### 3. Run the Example

```bash
python3 cannyEdgeDetector.py
```

This script will:
- Load a color image
- Convert and resize it to multiple resolutions
- Run the NPP+ Canny edge detection directly on GPU
- Benchmark and log performance
- Save outputs and CSV results

---
## ✅ Images Input and Output
Input Image
![Teapot_Resize_800x600](/NPP+/Teapot_resolutions/Teapot_Resize_800x600.png)

Canny Edge detected
![out_npp_800x600](/NPP+/Teapot_resolutions/out_npp_800x600.pngg)

---
## 💡 Tips for PyTorch Users

Ensure your tensors are compatible with CUDA and contiguous for optimal performance:
```python
torch_img = torch.from_numpy(img).cuda().contiguous()
```

---

## 📦 Requirements

- NVIDIA GPU with CUDA support
- CUDA 12.8 or later
- NPP+ libraries (libnppif.so)
- Python 3.x
- Python packages:
  - `torch`
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `tabulate`

---
## 📊 Performance Summary

| # | Resolution | Megapixels | NPP Time (ms)         |
|---|------------|------------|-----------------------|
| 0 | 320x180    | 0.0576     | 0.0452                |
| 1 | 640x360    | 0.2304     | 0.0483                |
| 2 | 800x600    | 0.48       | 0.0548                |
| 3 | 1280x720   | 0.9216     | 0.0630                |
| 4 | 1920x1080  | 2.0736     | 0.1046                |
| 5 | 2560x1440  | 3.6864     | 0.1543                |
| 6 | 3840x2160  | 8.2944     | 0.2986                |
| 7 | 5120x2880  | 14.7456    | 0.5008                |

---
## ✅ Output

You’ll find:
- Edge-detected images in `/Teapot_resolutions`
- `performance_results.csv` with benchmark statistics


---

## 📍 Notes

- Optimized for pipelines where data is already in GPU (e.g., deep learning)
- Ideal for large-scale, real-time image processing on high-res input