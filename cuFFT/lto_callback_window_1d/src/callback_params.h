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


#ifndef _CALLBACK_PARAMS__H_
#define _CALLBACK_PARAMS__H_

// Callback parameters structure
struct cb_params {
	unsigned window_size;
	unsigned signal_size;
};

// Callback parameters structure for padding
struct PaddingCallbackParams {
    unsigned int signal_size;
    unsigned int padded_signal_size;
};

// Problem input parameters
constexpr unsigned batches              = 830;
constexpr unsigned signal_size          = 328;
constexpr unsigned window_size          =  32;
constexpr unsigned complex_signal_size  = signal_size / 2 + 1;

// Padding example parameters - use single batch for simplicity
constexpr unsigned padding_batches     = 1;
constexpr unsigned padded_signal_size   = 2 * signal_size;
constexpr unsigned padding_complex_signal_size = padded_signal_size / 2 + 1;

// Precision threshold
constexpr float threshold = 1e-6;

#endif // _CALLBACK_PARAMS__H_