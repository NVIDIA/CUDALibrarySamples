// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cufft.h>

// cuFFT calls this after its forward C2C transform. caller_info points to a
// frequency-domain filter with the same layout as the output transform.
__device__ void cufftdx_tile_store_multiply(void* data_out,
                                            unsigned long long offset,
                                            cufftComplex element,
                                            void* caller_info,
                                            void*) {
    const cufftComplex filter = static_cast<const cufftComplex*>(caller_info)[offset];
    static_cast<cufftComplex*>(data_out)[offset] = {
        element.x * filter.x - element.y * filter.y,
        element.x * filter.y + element.y * filter.x,
    };
}
