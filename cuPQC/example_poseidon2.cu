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

#include <vector>
#include <iomanip>
#include <stdio.h>
#include <iostream>

#include <cuhash.hpp>

using namespace cupqc;
using HASH = decltype(POSEIDON2_8_16() + Thread());

__global__ void hash_poseidon2_kernel(uint32_t* digest, const uint32_t* msg, size_t inbuf_len, size_t out_len)
{
    // Poseidon2 with Capacity 8 and Width 16 with BabyBear field
    HASH hash {};
    hash.reset();
    hash.update(msg, inbuf_len);
    hash.finalize();
    hash.digest(digest, out_len);
}

void hash_poseidon2(std::vector<uint32_t>& digest, std::vector<uint32_t>& msg, size_t out_len)
{
    uint32_t* d_msg;
    uint32_t* d_digest;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size() * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_digest), digest.size() * sizeof(uint32_t));

    cudaMemcpy(d_msg, msg.data(), msg.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    //Note that the poseidon2 hash function is a thread configuration, so we use a single thread to hash a single message.
    hash_poseidon2_kernel<<<1, 1>>>(d_digest, d_msg, msg.size(), out_len);

    cudaMemcpy(digest.data(), d_digest, digest.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_msg);
    cudaFree(d_digest);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    constexpr size_t in_len = 64;
    constexpr size_t out_len = 16;
    std::vector<uint32_t> msg(in_len, 0);

    // Note that in_len is strictly less than BabyBearPrime, so we don't actually need to mod by BabyBearPrime.
    // This is just for illustration, as our Poseidon2 hash is built on the BabyBear field.
    for (size_t i = 0; i < in_len; i++) {
        msg[i] = i % cupqc_common::BabyBearPrime;
    }


    std::vector<uint32_t> digest(out_len, 0);
    hash_poseidon2(digest, msg, out_len);
    std::cout << "Poseidon2 Hashed output: " << std::endl;
    for (uint32_t num : digest) {
        std::cout << num << '\t';
    }
    std::cout << std::endl;
}