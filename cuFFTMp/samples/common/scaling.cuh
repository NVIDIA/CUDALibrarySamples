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

#include "../iterators/box_iterator.hpp"

__global__
void scaling_kernel(BoxIterator<cufftComplex> begin, BoxIterator<cufftComplex> end, int rank, int size, size_t nx, size_t ny, size_t nz, bool printing = true) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    begin += tid;
    if(begin < end) {
        // begin.x(), begin.y() and begin.z() are the global 3D coordinate of the data pointed by the iterator
        // begin->x and begin->y are the real and imaginary part of the corresponding cufftComplex element
        if(tid < 10 && printing) {
            printf("GPU data (after first transform): global 3D index [%d %d %d], local index %d, rank %d is (%f,%f)\n", 
                (int)begin.x(), (int)begin.y(), (int)begin.z(), (int)begin.i(), rank, begin->x, begin->y);
        }
        *begin = {begin->x / (float)(nx * ny * nz), begin->y / (float)(nx * ny * nz)};
    }
};