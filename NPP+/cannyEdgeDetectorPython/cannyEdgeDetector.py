# Copyright 2021-2025 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
# 

import os
import time
import cv2
import torch
import ctypes
import numpy as np
import pandas as pd
import torchvision
from tabulate import tabulate
from ctypes import c_int, c_int16, c_ubyte, POINTER, c_void_p, sizeof

# Settings
INPUT_IMAGE = "Teapot.jpg"
OUTPUT_DIR = "Teapot_resolutions"
WARMUP_ITERATIONS = 5
MEASURE_ITERATIONS = 1000
THRESH_WEAK = 72
THRESH_STRONG = 256

RESOLUTIONS = [
    (320, 180, "320x180"),
    (640, 360, "640x360"),
    (800, 600, "800x600"),
    (1280, 720, "1280x720"),
    (1920, 1080, "1920x1080"),
    (2560, 1440, "2560x1440"),
    (3840, 2160, "3840x2160"),
    (5120, 2880, "5120x2880"),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

class CannyEdgeDetector:
    class NppiSize(ctypes.Structure):
        _fields_ = [("width", c_int), ("height", c_int)]

    class NppiPoint(ctypes.Structure):
        _fields_ = [("x", c_int), ("y", c_int)]

    class NppStreamContext(ctypes.Structure):
        _fields_ = [
            ("hStream", c_void_p), ("nCudaDeviceId", c_int),
            ("nMultiProcessorCount", c_int), ("nMaxThreadsPerMultiProcessor", c_int),
            ("nMaxThreadsPerBlock", c_int), ("nSharedMemPerBlock", c_int),
            ("nCudaDevAttrComputeCapabilityMajor", c_int),
            ("nCudaDevAttrComputeCapabilityMinor", c_int),
            ("nStreamFlags", c_int)
        ]

    def __init__(self):
        #npp_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\npp_plus_if64_12.dll" # for windows user
        #self.npp_lib = ctypes.cdll.LoadLibrary(npp_path)
        self.npp_lib = ctypes.CDLL('libnpp_plus_if.so')  # for linux user
        self._setup_functions()

    def _setup_functions(self):
        self.get_buffer_size = self.npp_lib.nppiFilterCannyBorderGetBufferSize
        self.get_buffer_size.restype = c_int
        self.get_buffer_size.argtypes = [self.NppiSize, POINTER(c_int)]

        self.canny_func = self.npp_lib.nppiFilterCannyBorder_8u_C1R_Ctx
        self.canny_func.restype = c_int
        self.canny_func.argtypes = [
            POINTER(c_ubyte), c_int, self.NppiSize, self.NppiPoint,
            POINTER(c_ubyte), c_int, self.NppiSize,
            c_int, c_int, c_int16, c_int16,
            c_int, c_int, c_void_p, self.NppStreamContext
        ]

    def __call__(self, img_tensor, low_thresh, high_thresh):
        if img_tensor.dtype != torch.uint8:
            img_tensor = (img_tensor * 255).byte()
        if not img_tensor.is_cuda:
            img_tensor = img_tensor.cuda()

        h, w = img_tensor.shape
        output = torch.empty_like(img_tensor)
        scratch_size = c_int()
        self.get_buffer_size(self.NppiSize(w, h), ctypes.byref(scratch_size))
        scratch_buffer = torch.empty(scratch_size.value, dtype=torch.uint8, device='cuda')

        stream_ctx = self.NppStreamContext()
        stream_ctx.hStream = c_void_p(torch.cuda.current_stream().cuda_stream)

        status = self.canny_func(
            ctypes.cast(img_tensor.data_ptr(), POINTER(c_ubyte)),
            w * sizeof(c_ubyte), self.NppiSize(w, h), self.NppiPoint(0, 0),
            ctypes.cast(output.data_ptr(), POINTER(c_ubyte)),
            w * sizeof(c_ubyte), self.NppiSize(w, h),
            0, 200, c_int16(low_thresh), c_int16(high_thresh),
            2, 2,
            ctypes.cast(scratch_buffer.data_ptr(), c_void_p), stream_ctx
        )

        if status != 0:
            raise RuntimeError(f"NPP Canny edge detection failed with status {status}")

        return output

# Load input image
img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(f"Image not found: {INPUT_IMAGE}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA device not available!")

print(f"Input image size: {img.shape}")
print(f"Running benchmark... {MEASURE_ITERATIONS} iterations")

detector = CannyEdgeDetector()
results = []

for width, height, label in RESOLUTIONS:
    print(f"\nProcessing {label}")
    resized = cv2.resize(img, (width, height))
    img_tensor = torch.from_numpy(resized).cuda().permute(2, 0, 1)
    img_tensor = torchvision.transforms.functional.rgb_to_grayscale(img_tensor).squeeze(0)

    for _ in range(WARMUP_ITERATIONS):
        detector(img_tensor, THRESH_WEAK, THRESH_STRONG)

    timings = []
    for _ in range(MEASURE_ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        detector(img_tensor, THRESH_WEAK, THRESH_STRONG)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    output_image = detector(img_tensor, THRESH_WEAK, THRESH_STRONG).cpu().numpy()
    cv2.imwrite(f"{OUTPUT_DIR}/out_npp_{label}.png", output_image)

    results.append({
        "Resolution": label,
        "Megapixels": (width * height) / 1_000_000,
        "NPP Time (ms)": np.mean(timings)
    })

# Save performance summary
df = pd.DataFrame(results)
print("\n--- Performance Summary ---")
print(tabulate(df, headers='keys', tablefmt='pretty', floatfmt='.3f'))

csv_path = os.path.join(OUTPUT_DIR, "performance_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

