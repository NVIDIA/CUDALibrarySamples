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
 * Example showing the use of LTO callbacks with CUFFT to perform
 * truncation with zero padding.
 *
*/

#include <cufftXt.h>
#include "callback_params.h"

// This is the store callback routine. It filters high frequencies
// based on a truncation window specified by the user
__device__ cufftComplex windowing_callback(void*              input,
                                           unsigned long long idx,
                                           void*              info,
                                           void*              sharedmem) {

	const cb_params* params = static_cast<const cb_params*>(info);
	cufftComplex* cb_output = static_cast<cufftComplex*>(input);

	const unsigned sample   = idx % params->signal_size;

	return (sample < params->window_size) ? cb_output[idx] : cufftComplex{0.f, 0.f};
}

// Same as above, but using constant memory
__constant__ unsigned cmem_window_size = window_size;
__constant__ unsigned cmem_signal_size = complex_signal_size;
__device__ cufftComplex windowing_constant_memory_callback(void*              input,
														   unsigned long long idx,
														   void*              info,
														   void*              sharedmem) {

	cufftComplex* cb_output = static_cast<cufftComplex*>(input);

	const unsigned sample   = idx % cmem_signal_size;

	return (sample < cmem_window_size) ? cb_output[idx] : cufftComplex{0.f, 0.f};
}