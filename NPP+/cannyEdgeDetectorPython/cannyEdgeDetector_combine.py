# / Copyright 2021-2025 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# 
# NOTICE TO LICENSEE: 
# 
# The source code and/or documentation ("Licensed Deliverables") are 
# subject to NVIDIA intellectual property rights under U.S. and 
# international Copyright laws. 
# 
# The Licensed Deliverables contained herein are PROPRIETARY and 
# CONFIDENTIAL to NVIDIA and are being provided under the terms and 
# conditions of a form of NVIDIA software license agreement by and 
# between NVIDIA and Licensee ("License Agreement") or electronically 
# accepted by Licensee.  Notwithstanding any terms or conditions to 
# the contrary in the License Agreement, reproduction or disclosure 
# of the Licensed Deliverables to any third party without the express 
# written consent of NVIDIA is prohibited. 
# 
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE 
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE 
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND. 
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED 
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, 
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE. 
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE 
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY 
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY 
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS 
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
# OF THESE LICENSED DELIVERABLES. 
# 
# U.S. Government End Users.  These Licensed Deliverables are a 
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 
# 1995), consisting of "commercial computer software" and "commercial 
# computer software documentation" as such terms are used in 48 
# C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government 
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and 
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all 
# U.S. Government End Users acquire the Licensed Deliverables with 
# only those rights set forth herein. 
# 
# Any use of the Licensed Deliverables in individual and commercial 
# software must include, in the user documentation and internal 
# comments to the code, the above Disclaimer and U.S. Government End 
# Users Notice. 
# /

# importing the libraries
import cv2
import os
import time
import torch
import ctypes
# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
from tabulate import tabulate
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from ctypes import c_int, c_int16, c_ubyte, POINTER, c_void_p, sizeof

# Input image path and output directory
input_image_path = "Teapot.jpg"
output_dir = "Teapot_resolutions"
os.makedirs(output_dir, exist_ok=True)

# Canny threshold parameters
thresh_weak_cv2, thresh_strong_cv2 = 36, 128  # OpenCV uses L2 gradient
thresh_weak_npp, thresh_strong_npp = 72, 256   # NPP optimized parameters with L2s gradient

# Number of iterations for performance measurement
iterations = 1000
warmup_iterations = 5

# Define resolutions to test
# Format: (width, height, name)
resolutions = [
    (320, 180, "320x180"),
    (640, 360, "640x360"),
    (800, 600, "800x600"), 	# Original resolution
    (1280, 720, "1280x720"),  
    (1920, 1080, "1920x1080"),
    (2560, 1440, "2560x1440"),
    (3840, 2160, "3840x2160"),  # 4K
    (5120, 2880, "5120x2880"),  # 5K
]

class CannyEdgeDetection:
    class NppiSize(ctypes.Structure):
        _fields_ = [("width", c_int), ("height", c_int)]

    class NppiPoint(ctypes.Structure):
        _fields_ = [("x", c_int), ("y", c_int)]

    class NppStreamContext(ctypes.Structure):
        _fields_ = [
            ("hStream", c_void_p),
            ("nCudaDeviceId", c_int),
            ("nMultiProcessorCount", c_int),
            ("nMaxThreadsPerMultiProcessor", c_int),
            ("nMaxThreadsPerBlock", c_int),
            ("nSharedMemPerBlock", c_int),
            ("nCudaDevAttrComputeCapabilityMajor", c_int),
            ("nCudaDevAttrComputeCapabilityMinor", c_int),
            ("nStreamFlags", c_int)
        ]

    def __init__(self):
        self.npp_i_lib = self._load_npp_library()
        self._setup_buffer_size_function()
        self._setup_canny_function()

    def _load_npp_library(self):
        #npp_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nppif64_12.dll" # for windows user
        #npp_lib = ctypes.cdll.LoadLibrary(npp_path)
        npp_lib = ctypes.CDLL('libnpp_plus_if.so') # for linux user
        return npp_lib

    def _setup_buffer_size_function(self):
        self.get_buffer_size_func = self.npp_i_lib.nppiFilterCannyBorderGetBufferSize
        self.get_buffer_size_func.restype = c_int
        self.get_buffer_size_func.argtypes = [
            CannyEdgeDetection.NppiSize,  # oSizeROI
            POINTER(c_int)                # bufferSize
        ]

    def _setup_canny_function(self):
        self.canny_func = self.npp_i_lib.nppiFilterCannyBorder_8u_C1R_Ctx
        self.canny_func.restype = c_int
        self.canny_func.argtypes = [
            POINTER(c_ubyte),   # pSrc
            c_int,              # nSrcStep
            CannyEdgeDetection.NppiSize,  # oSrcSize
            CannyEdgeDetection.NppiPoint, # oSrcOffset
            POINTER(c_ubyte),   # pDst
            c_int,              # nDstStep
            CannyEdgeDetection.NppiSize,  # oSizeROI
            c_int,              # eFilterType
            c_int,              # eMaskSize
            c_int16,            # nLowThreshold
            c_int16,            # nHighThreshold
            c_int,              # eNorm
            c_int,              # eBorderType
            c_void_p,           # pBuffer
            CannyEdgeDetection.NppStreamContext  # nppStreamCtx
        ]

    def _prepare_input(self, image_tensor):
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.cuda()
        if image_tensor.dtype != torch.uint8:
            image_tensor = (image_tensor * 255).byte()
        return image_tensor

    def _get_buffer_size(self, roi):
        buffer_size = c_int(0)
        status = self.get_buffer_size_func(roi, ctypes.byref(buffer_size))
        if status != 0:
            raise RuntimeError(f"Failed to get buffer size, status: {status}")
        return buffer_size.value

    def _setup_stream_context(self):
        stream_ctx = CannyEdgeDetection.NppStreamContext()
        stream_ctx.hStream = c_void_p(torch.cuda.current_stream().cuda_stream)
        return stream_ctx

    def _record_streams(self, scratch_buffer, image_tensor, output):
        torch_stream = torch.cuda.current_stream()
        scratch_buffer.record_stream(torch_stream)
        image_tensor.record_stream(torch_stream)
        output.record_stream(torch_stream)


    def __call__(self, image_tensor, low_threshold=50, high_threshold=100):

        # Prepare input
        image_tensor = self._prepare_input(image_tensor)

        blurred = image_tensor
        height, width = blurred.shape
        output = torch.empty_like(blurred)

        src_ptr = ctypes.cast(blurred.data_ptr(), POINTER(c_ubyte))
        dst_ptr = ctypes.cast(output.data_ptr(), POINTER(c_ubyte))

        roi = CannyEdgeDetection.NppiSize(width, height)
        buffer_size = self._get_buffer_size(roi)

        scratch_buffer = torch.empty(buffer_size, dtype=torch.uint8, device='cuda')
        buffer_ptr = ctypes.cast(scratch_buffer.data_ptr(), c_void_p)

        stream_ctx = self._setup_stream_context()

        status = self.canny_func(
            src_ptr,
            width * sizeof(c_ubyte),
            CannyEdgeDetection.NppiSize(width, height),
            CannyEdgeDetection.NppiPoint(0, 0),
            dst_ptr,
            width * sizeof(c_ubyte),
            roi,
            0,  # Sobel
            200,  # 3x3 mask
            c_int16(low_threshold),
            c_int16(high_threshold),
            2,  # L2 norm
            2,  # border replicate
            buffer_ptr,
            stream_ctx
        )

        self._record_streams(scratch_buffer, blurred, output)

        if status != 0:
            raise RuntimeError(f"NPP Canny edge detection failed with status {status}")

        return output


# Load the original input image
original_img = cv2.imread(input_image_path)
if original_img is None:
    raise ValueError(f"Could not read image from {input_image_path}")

print(f"Original image shape: {original_img.shape}")
print(f"Running performance test with {iterations} iterations (plus {warmup_iterations} warmup iterations)...")

# Check if CUDA is available for PyTorch
torch_gpu_available = torch.cuda.is_available()
if torch_gpu_available:
    print(f"PyTorch CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("PyTorch CUDA is required for this benchmark")

# Create a DataFrame to store results
results = []

# Process each resolution
for width, height, name in resolutions:
    print(f"\nProcessing resolution: {name}")

    # Resize image to target resolution
    resized_img = cv2.resize(original_img, (width, height), interpolation=cv2.INTER_AREA)

    # Save resized image
    resized_path = f"{output_dir}/Teapot_Resize_{name}.png"
    cv2.imwrite(resized_path, resized_img)
    print(f"Saved resized image to: {resized_path}")

    # Convert to grayscale for OpenCV
    resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Upload image to GPU once (simulating data already on GPU)
    torch_img_gray = torch.from_numpy(resized_gray).cuda()

    # Also upload the BGR image to GPU for NPP (ensuring it's contiguous for optimal performance)
    torch_img_bgr = torch.from_numpy(resized_img).cuda().contiguous()

    # ===== SCENARIO 1: OpenCV with data transfer costs =====
    # This simulates the real-world scenario where data needs to be moved from GPU to CPU and back

    # Warmup
    for _ in range(warmup_iterations):
        # GPU -> CPU transfer
        cpu_img = torch_img_gray.cpu().numpy()
        # Run OpenCV Canny on CPU
        cv2_canny = cv2.Canny(cpu_img.astype(np.uint8), thresh_weak_cv2, thresh_strong_cv2, L2gradient=True)
        # CPU -> GPU transfer (results back to GPU)
        _ = torch.from_numpy(cv2_canny).cuda()

    # Measure OpenCV performance (including transfer costs)
    opencv_times = []
    for i in range(iterations):
        start_time = time.time()
        # GPU -> CPU transfer
        cpu_img = torch_img_gray.cpu().numpy()
        # Run OpenCV Canny on CPU
        cv2_canny = cv2.Canny(cpu_img.astype(np.uint8), thresh_weak_cv2, thresh_strong_cv2, L2gradient=True)
        # CPU -> GPU transfer (results back to GPU)
        gpu_result = torch.from_numpy(cv2_canny).cuda()
        end_time = time.time()
        opencv_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

    # Save OpenCV output
    cv2.imwrite(f"{output_dir}/out_cv2_{name}.png", cv2_canny)

    # ===== SCENARIO 2: NPP with data already on GPU (OPTIMIZED) =====
    canny_torch = CannyEdgeDetection()
    img_tensor = torch.from_numpy(resized_img).cuda().to(torch.uint8)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = torchvision.transforms.functional.rgb_to_grayscale(img_tensor)
    img_tensor = img_tensor.squeeze(0)

    # Warmup
    for _ in range(warmup_iterations):
    	output_torch = canny_torch(img_tensor, thresh_weak_npp, thresh_strong_npp)

    # Measure NPP performance with data already on GPU
    npp_times = []
    for i in range(iterations):
        # Use PyTorch CUDA events for more accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        output_torch = canny_torch(img_tensor, thresh_weak_npp, thresh_strong_npp)

        # Keep result on GPU
        end_event.record()
        torch.cuda.synchronize()

        npp_times.append(start_event.elapsed_time(end_event))

        # Get one result for saving
        output_torch_npp = np.array(output_torch.cpu())

    # Save NPP output
    cv2.imwrite(f"{output_dir}/out_npp_{name}.png", output_torch_npp)

    # Calculate performance statistics
    opencv_mean = np.mean(opencv_times)
    opencv_median = np.median(opencv_times)
    opencv_min = np.min(opencv_times)
    opencv_max = np.max(opencv_times)
    opencv_std = np.std(opencv_times)

    npp_mean = np.mean(npp_times)
    npp_median = np.median(npp_times)
    npp_min = np.min(npp_times)
    npp_max = np.max(npp_times)
    npp_std = np.std(npp_times)

    # Calculate speedup
    speedup = opencv_mean / npp_mean

    # Print resolution-specific results
    print(f"\n--- Performance Results for {name} ({width}x{height}, {width * height / 1_000_000:.2f} MP) ---")

    print("\nOpenCV Canny (including GPU->CPU->GPU transfer):")
    print(f"  Mean time: {opencv_mean:.3f} ms")
    print(f"  Median time: {opencv_median:.3f} ms")
    print(f"  Min time: {opencv_min:.3f} ms")
    print(f"  Max time: {opencv_max:.3f} ms")
    print(f"  Std dev: {opencv_std:.3f} ms")

    print("\nNPP Canny (optimized with PyTorch tensor):")
    print(f"  Mean time: {npp_mean:.3f} ms")
    print(f"  Median time: {npp_median:.3f} ms")
    print(f"  Min time: {npp_min:.3f} ms")
    print(f"  Max time: {npp_max:.3f} ms")
    print(f"  Std dev: {npp_std:.3f} ms")

    print(f"\nSpeedup (OpenCV / NPP): {speedup:.2f}x")
    if speedup > 1:
        print(f"NPP is {speedup:.2f}x faster than OpenCV")
    else:
        print(f"OpenCV is {1/speedup:.2f}x faster than NPP")

    # Add results to DataFrame
    results.append({
        "Resolution": name,
        "Megapixels": (width * height) / 1_000_000,
        "OpenCV (CPU + transfer) (ms)": opencv_mean,
        "NPP (GPU optimized) (ms)": npp_mean,
        "Speedup": speedup
    })

# Create DataFrame and print results
df = pd.DataFrame(results)
print("\n--- Overall Performance Results ---")
print(tabulate(df, headers='keys', tablefmt='pretty', floatfmt='.3f'))

# Save results to CSV
csv_path = f"{output_dir}/performance_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")
print("\nProcessing complete.") 
