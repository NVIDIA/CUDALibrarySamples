/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _CUFFTMP_JAX_KERNELS_H_
#define _CUFFTMP_JAX_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cufftmp_jax {

/**
 * Description of an FFT
 * - global_x, global_y, global_z are the global size of the tensor to transform
 * - distribution is 0 for a CUFFT_XT_FORMAT_INPLACE (== Slabs_X) and
 *   1 for a CUFFT_XT_FORMAT_INPLACE_SHUFFLED (== Slabs_Y) data distribution
 * - direction is 0 for a CUFFT_FORWARD transform, 1 for CUFFT_INVERSE
 */

struct cufftmpDescriptor {
    std::int64_t global_x;
    std::int64_t global_y;
    std::int64_t global_z;
    int distribution;
    int direction;
};

/**
 * Generic signature for a custom op with CUDA
 */
void gpu_cufftmp(cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len);

}  // namespace cufftmp_jax

#endif