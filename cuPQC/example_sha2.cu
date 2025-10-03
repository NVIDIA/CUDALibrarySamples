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
#include <cuhash.hpp>
#include <stdio.h>

using namespace cupqc;

using SHA2_256_THREAD = decltype(SHA2_256() + Thread());

__global__ void hash_sha2_kernel(uint8_t* digest, const uint8_t* msg, size_t inbuf_len)
{
    SHA2_256_THREAD hash {};
    if (threadIdx.x == 0) {
        hash.reset();
        hash.update(msg, inbuf_len);
        hash.finalize();
        hash.digest(digest, SHA2_256_THREAD::digest_size);
    }
}


void hash_sha2(std::vector<uint8_t>& digest, std::vector<uint8_t>& msg)
{
    uint8_t* d_msg;
    uint8_t* d_digest;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size());
    cudaMalloc(reinterpret_cast<void**>(&d_digest), digest.size());

    cudaMemcpy(d_msg, msg.data(), msg.size(), cudaMemcpyHostToDevice);

    hash_sha2_kernel<<<1, 32>>>(d_digest, d_msg, msg.size());

    cudaMemcpy(digest.data(), d_digest, digest.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_msg);
    cudaFree(d_digest);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    const char * msg_str = "The quick brown fox jumps over the lazy dog";
    std::vector<uint8_t> msg(reinterpret_cast<const uint8_t*>(msg_str), reinterpret_cast<const uint8_t*>(msg_str) + strlen(msg_str));
    std::vector<uint8_t> digest(SHA2_256::digest_size, 0);
    hash_sha2(digest, msg);
    printf("SHA2-256: ");
    for (uint8_t num : digest) {
        printf("%02x", num);
    }
    printf("\n");
}