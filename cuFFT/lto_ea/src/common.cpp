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

#ifndef _COMMON__CPP_
#define _COMMON__CPP_

#include <random>
#include "common.h"

// Wave parameters
constexpr unsigned waves                =  12;
constexpr float    signal_max_A         =  20.;
constexpr float    signal_max_f         = 500.;
constexpr float    sampling_dt          = 1e-3;

// Initialize the input signal as a composite of sine waves
// with random amplitudes and frequencies
void init_input_signals(unsigned batches, unsigned signal_size, float* signals) {

	std::mt19937 e2(0);

	std::uniform_real_distribution<> A_dist(0., signal_max_A);
	std::uniform_real_distribution<> f_dist(0., signal_max_f);

	const unsigned complex_signal_size = signal_size / 2 + 1;

	for(unsigned batch = 0; batch < batches; ++batch) {
		std::vector<float> wave_amplitudes;
		std::vector<float> wave_frequencies;

		// Generate the amplitudes and frequencies of the waves
		for(unsigned i = 0; i < waves; ++i) {
			wave_amplitudes.push_back(A_dist(e2));
			wave_frequencies.push_back(f_dist(e2));
		}

		// Compose the signal
		float time = 0.;
		for(unsigned s = 0; s < signal_size; ++s) {
			for(unsigned i = 0; i < waves; ++i) {
				unsigned idx = batch * (2 * complex_signal_size) + s;
				signals[idx] += wave_amplitudes[i] * sin(2. * PI * wave_frequencies[i] * time) ;
			}
			time += sampling_dt;
		}
	}
}

#endif // _COMMON__CPP_

