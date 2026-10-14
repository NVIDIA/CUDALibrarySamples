# 🧮 Image Euclidean Distance Transform (EDT) using NVIDIA NPP

This repository demonstrates how to use the **NVIDIA Performance Primitives (NPP)** library to compute Euclidean distance transforms and Voronoi diagrams on grayscale images using the GPU.

---

## 📌 Overview

This example loads raw `.raw` grayscale images, performs the following operations using NPP:

- Truncated Euclidean Distance Transform (EDT)
- True (floating-point) Distance Transform
- Voronoi Diagram Generation

The results are saved as output raw files, and CUDA streams are used for efficient execution.

---

## 🚀 Features

- GPU-accelerated distance transform using `nppiDistanceTransformPBA_*` functions
- Support for multiple output types: `16u`, `32f`, `16s`
- Batch processing and stream-based execution
- Demonstrates use of `NppStreamContext`

---

## 📷 Example Output

### Input Image  
`Dolphin1_313x317_8u.raw`  
![Input Image](/NPP/distanceTransform/dolphin1_Input_319x319_8u.jpg)

### Distance Transform  
`DistanceTransformTrue_Dolphin1_319x319_16u.raw`  
![Distance Transform](/NPP/distanceTransform/DistanceTransformTrue_Dolphin1_319x319_16u.jpg)

---

## 🧠 Key Concepts

- GPU Image Processing
- Distance Transform
- Voronoi Diagrams
- CUDA Streams
- Raw Image Format Handling

---

## 💻 Supported Platforms

| Category            | Support                                 |
|---------------------|------------------------------------------|
| GPU Architectures   | SM 7.5, SM 8.0 and Above                 |
| OS                  | Linux, Windows                           |
| CPU Architecture    | x86_64                                   |
| CUDA Toolkit        | Requires CUDA 11.2+                      |

---

## 🛠️ Build Instructions

### Linux
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### Windows
```bash
$ mkdir build
$ cd build
$ cmake -DCMAKE_GENERATOR_PLATFORM=x64 ..
# Open distanceTransform.sln in Visual Studio and build
```

---

## ▶️ Run Instructions

```bash
./distanceTransform
```

Expected Output:
```
NPP Library Version 12.4.0
CUDA Driver  Version: 12.9
CUDA Runtime Version: 12.9

Input file load succeeded.
Input file load succeeded.
Done!

```

---

## 📦 Dependencies

- [NPP Library](https://docs.nvidia.com/cuda/npp/index.html)
- CUDA Toolkit ≥ 11.x
- CMake

---

## 🧾 License

This sample is released under the NVIDIA Software License Agreement. Refer to the `LICENSE` file for more details.

## NPP workspace (CUDA 13.4)

The sample creates a workspace (`nppiWorkspaceCreate`) and probes `nppiDistanceTransformPBA_8u16u_C1R_Ctx` with a `NULL` scratch buffer on a small image at start-up. If NPP accepts it, the workspace supplies the scratch buffers; if it returns `NPP_NULL_POINTER_ERROR` (CUDA 13.4 does; later NPP builds accept `NULL`) the sample allocates them as before. It prints which path is in use. `NppStreamContext` is zero-initialized so `hWorkspace` is zero when unused.
