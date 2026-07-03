/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Device callback for LTO + nvcc offline compilation example
 * Load callback for zero-padding
 *
 */

#include <cufftXt.h>

// Callback parameters structure for padding
struct PaddingCallbackParams {
    unsigned int signal_size;
    unsigned int padded_signal_size;
};

// Load callback for R2C: returns 0 for indices beyond signal_size
__device__ cufftReal padding_load_callback(void* dataIn,
                                        unsigned long long offset,
                                        void* callerInfo,
                                        void* sharedPointer) {
    PaddingCallbackParams* params = static_cast<PaddingCallbackParams*>(callerInfo);
    cufftReal* in = static_cast<cufftReal*>(dataIn);
    
    // If offset is beyond the original data, return zero
    if (offset >= params->signal_size) {
        return 0.0f;
    }
    
    // Return the real input data (non-interleaved format)
    return in[offset];
}
