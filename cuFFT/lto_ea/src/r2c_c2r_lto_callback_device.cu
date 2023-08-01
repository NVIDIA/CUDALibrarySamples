/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/* 
 * Example showing the use of LTO callbacks with CUFFT to perform 
 * truncation with zero padding.
 * 
*/

#include <cufftXt.h>

struct cb_params {
	unsigned window_N;
	unsigned signal_size;
};

// This is the store callback routine. It filters high frequencies
// based on a truncation window specified by the user
// NOTE: unlike the non-LTO version, the callback device function
// must have the name cufftJITCallbackLoadComplex, it cannot be aliased
__device__ cufftComplex cufftJITCallbackLoadComplex(void *input,
                                                    size_t index,
                                                    void *info,
                                                    void *sharedmem) {
  
	const cb_params* params = static_cast<const cb_params*>(info);
	cufftComplex* cb_output = static_cast<cufftComplex*>(input);
	const unsigned sample   = index % params->signal_size;

	return (sample < params->window_N) ? cb_output[index] : cufftComplex{0.f, 0.f};
}

