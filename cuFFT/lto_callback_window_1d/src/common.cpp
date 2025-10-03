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


#ifndef _COMMON__CPP_
#define _COMMON__CPP_

#include <random>
#include "common.h"

// Wave parameters
constexpr unsigned waves        =  12;
constexpr float    signal_max_A =  20.;
constexpr float    signal_max_f = 500.;
constexpr float    sampling_dt  = 1e-3;

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