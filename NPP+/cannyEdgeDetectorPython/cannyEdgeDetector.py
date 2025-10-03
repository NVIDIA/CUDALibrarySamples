# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
