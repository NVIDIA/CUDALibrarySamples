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
#include <cupqc.hpp>
using namespace cupqc;

using MLDSA44Key = decltype(ML_DSA_44()
                            + Function<function::Keygen>()
                            + Block()
                            + BlockDim<128>());  //Optional operator with default config

using MLDSA44Sign = decltype(ML_DSA_44()
                             + Function<function::Sign>()
                             + Block()
                             + BlockDim<128>());  //Optional operator with default config

using MLDSA44Verify = decltype(ML_DSA_44()
                               + Function<function::Verify>()
                               + Block()
                               + BlockDim<128>());  //Optional operator with default config

__global__ void keygen_kernel(uint8_t* public_keys, uint8_t* secret_keys, uint8_t* randombytes, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA44Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLDSA44Key::public_key_size;
    auto secret_key = secret_keys + block * MLDSA44Key::secret_key_size;
    auto entropy    = randombytes + block * MLDSA44Key::entropy_size;
    auto work       = workspace   + block * MLDSA44Key::workspace_size;

    MLDSA44Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void sign_kernel(uint8_t* signatures, const uint8_t* messages, const size_t message_size, const uint8_t* secret_keys, uint8_t *randombytes, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA44Sign::shared_memory_size];
    int block = blockIdx.x;
    auto signature  = signatures  + block * ((MLDSA44Sign::signature_size + 15) / 16 * 16);
    auto message    = messages    + block * message_size;
    auto secret_key = secret_keys + block * MLDSA44Sign::secret_key_size;
    auto entropy    = randombytes + block * MLDSA44Sign::entropy_size;
    auto work       = workspace   + block * MLDSA44Sign::workspace_size;

    MLDSA44Sign().execute(signature, message, message_size, secret_key, entropy, work, smem_ptr);
}

__global__ void verify_kernel(bool* valids, const uint8_t* signatures, const uint8_t* messages, const size_t message_size, const uint8_t* public_keys, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLDSA44Verify::shared_memory_size];
    int block = blockIdx.x;
    auto signature   = signatures + block * ((MLDSA44Sign::signature_size + 15) / 16 * 16);
    auto message     = messages    + block * message_size;
    auto public_key  = public_keys + block * MLDSA44Verify::public_key_size;
    auto work        = workspace   + block * MLDSA44Verify::workspace_size;

    valids[block] = MLDSA44Verify().execute(message, message_size, signature, public_key, work, smem_ptr);
}

void ml_dsa_keygen(std::vector<uint8_t>& public_keys, std::vector<uint8_t>& secret_keys, const unsigned int batch)
{
    /*
     * Set up for utilizing cuPQCDx ML-DSA Keygen.
     * Allocates device workspace for computation
     */
    auto length_public_key = MLDSA44Key::public_key_size;
    auto length_secret_key = MLDSA44Key::secret_key_size;

    auto workspace         = make_workspace<MLDSA44Key>(batch);
    auto randombytes       = get_entropy<MLDSA44Key>(batch);
    /*
     * Allocate device memory for public and secret keys
     */
    uint8_t* d_public_key = nullptr;
    uint8_t* d_secret_key = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch); //These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);

    keygen_kernel<<<batch, MLDSA44Key::BlockDim>>>(d_public_key, d_secret_key, randombytes, workspace);

    /*
     * Transfer generated keys to the host for communication or storage
     */
    cudaMemcpy(public_keys.data(), d_public_key, length_public_key * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(secret_keys.data(), d_secret_key, length_secret_key * batch, cudaMemcpyDeviceToHost);

    /*
     * Delete device memory associated with the cryptograpic process.
     */
    cudaFree(d_public_key);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
    release_entropy(randombytes);
}

void ml_dsa_sign(std::vector<uint8_t>& signatures, std::vector<uint8_t>& messages,  size_t message_size,
                      const std::vector<uint8_t>& secret_keys, const unsigned int batch)
{
    /*
     * Set up for utilizing cuPQCDx ML-DSA Sign.
     * Allocates device workspace for computing
     */
    auto length_secret_key = MLDSA44Sign::secret_key_size;
    auto length_signature  = MLDSA44Sign::signature_size;
    auto length_signature_aligned = ((length_signature + 15) / 16) * 16;

    auto workspace         = make_workspace<MLDSA44Sign>(batch);
    auto randombytes       = get_entropy<MLDSA44Sign>(batch);

    /*
     * Allocate device memory for messages, secret_keys, messages
     */
    uint8_t* d_signatures  = nullptr;
    uint8_t* d_secret_keys = nullptr;
    uint8_t* d_messages    = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_signatures), length_signature_aligned * batch); //These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void**>(&d_secret_keys), length_secret_key * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_messages), message_size * batch);

    /*
     * Transfer messages and secret_keys from host memory
     */
    cudaMemcpy(d_secret_keys, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_messages, messages.data(), message_size * batch, cudaMemcpyHostToDevice);

    sign_kernel<<<batch, MLDSA44Sign::BlockDim>>>(d_signatures, d_messages, message_size, d_secret_keys, randombytes, workspace);

    /*
     * Transfer signatures from device memory
     */
    cudaMemcpy(signatures.data(), d_signatures, length_signature_aligned * batch, cudaMemcpyDeviceToHost);

    /*
     * Delete device memory associated with the cryptograpic process.
     */
    cudaFree(d_secret_keys);
    cudaFree(d_signatures);
    cudaFree(d_messages);
    destroy_workspace(workspace);
    release_entropy(randombytes);
}

void ml_dsa_verify(bool* is_valids, const std::vector<uint8_t>& signatures, const std::vector<uint8_t>& messages, size_t message_size,
                          const std::vector<uint8_t>& public_keys, const unsigned int batch)
{
    /*
     * Set up for utilizing cuPQCDx ML-DSA Verify.
     * Allocates device workspace for computing
     */
    auto workspace         = make_workspace<MLDSA44Verify>(batch);
    auto length_signature  = MLDSA44Verify::signature_size;
    auto length_signature_aligned = ((length_signature + 15) / 16) * 16;
    auto length_public_key = MLDSA44Verify::public_key_size;

    /*
     * Allocate device memory for public and secret keys
     */
    uint8_t* d_signatures  = nullptr;
    uint8_t* d_messages    = nullptr;
    uint8_t* d_public_keys = nullptr;
    bool*    d_valids      = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_signatures), length_signature_aligned * batch); //These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void**>(&d_public_keys), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_messages), message_size * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_valids), batch);

    /*
     * Transfer signatures, messages, and public_keys from host.
     */
    cudaMemcpy(d_public_keys, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_signatures, signatures.data(), length_signature_aligned * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_messages, messages.data(), message_size * batch, cudaMemcpyHostToDevice);

    verify_kernel<<<batch, MLDSA44Verify::BlockDim>>>(d_valids, d_signatures, d_messages, message_size, d_public_keys, workspace);

    /*
     * Transfer valid messages from device to host mem
     */
    cudaMemcpy(is_valids, d_valids, batch, cudaMemcpyDeviceToHost);

    /*
     * Delete device memory associated with the cryptograpic process.
     */
    cudaFree(d_public_keys);
    cudaFree(d_signatures);
    cudaFree(d_messages);
    cudaFree(d_valids);
    destroy_workspace(workspace);
}


/*
 * Normally different actors will be performing the various functions associated with ML-DSA.
 * In this example, we perform all functions, Keygen, Sign, and Verify. 
 * You could optimize this example by reusing the device memory, and not transfering to host,
 * however, this is not the normal scenario. So, to better illustrate the use of the cuPQCDx API we wrote
 * host functions ml_dsa_keygen, ml_dsa_sign, ml_dsa_verify that don't assume they will be called in
 * conjunction with the others. 
 * 
 * In this example we produce 10 keys, signatures and verified messages, using batching.
 * 
 * In a typical scenerio:
 * Actor 1 (keygen + Sign)
 * Actor 2 (Verify)
 * 
 * or 
 * 
 * Actor 1 as a keygeneration service (keygen)
 * Actor 2 Sign using key from 1.
 * Actor 3 Verify using key from 1 and signature from 2.
 */

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    constexpr unsigned int batch = 10;
    constexpr size_t message_size = 1024;

    /*
     * Generate Public and Secret Keys!
     */
    std::vector<uint8_t> public_keys(MLDSA44Key::public_key_size * batch);
    std::vector<uint8_t> secret_keys(MLDSA44Key::secret_key_size * batch);

    ml_dsa_keygen(public_keys, secret_keys, batch);

    /*
     * Create signature from message and secret key!
     */
    std::vector<uint8_t> signatures(((MLDSA44Sign::signature_size + 15) / 16 * 16) * batch);
    std::vector<uint8_t> messages(message_size * batch);
    ml_dsa_sign(signatures, messages, message_size, secret_keys, batch);

    bool is_valids[batch];
    /*
     * Verify that the signature is valid using public key!
     */
    ml_dsa_verify(is_valids, signatures, messages, message_size, public_keys, batch);
}