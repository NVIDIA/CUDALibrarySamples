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
 * Store callback for zero-padding
 *
 */

#include <cufftXt.h>

// Callback parameters structure for padding
struct PaddingCallbackParams {
    unsigned int signal_size;
    unsigned int padded_signal_size;
};

// Store callback for C2R: only stores first signal_size elements, zeros the rest
__device__ void padding_store_callback(void* dataOut,
                                   unsigned long long offset,
                                   cufftReal element,
                                   void* callerInfo,
                                   void* sharedPointer) {
    PaddingCallbackParams* params = static_cast<PaddingCallbackParams*>(callerInfo);
    cufftReal* out = static_cast<cufftReal*>(dataOut);
    
    // Only store if within original size, zero and normalize otherwise
    if (offset < params->signal_size) {
        out[offset] = element / params->padded_signal_size;
    } else {
        out[offset] = 0.0f;
    }
}
