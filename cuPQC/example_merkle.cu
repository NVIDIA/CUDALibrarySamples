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

using HASH   = decltype(POSEIDON2_8_16() + Thread());

//Will use MERKLE_FIELD_2048 in this example, we require FIELD when using the Poseidon2 hash function, and we use 256 threads per block
using MERKLE = decltype(MERKLE_FIELD_2048() + BlockDim<256>());

template<class Merkle, class Hash, typename Precision>
__global__ void generate_merkle_tree_kernel(tree<Merkle::MerkleSize, Hash, Precision> merkle, const Precision* msg, size_t inbuf_len)
{
    Hash hash{};
    for(uint32_t i = threadIdx.x; i < Merkle::MerkleSize; i += blockDim.x) {
        Merkle().create_leaf(merkle.nodes + i * merkle.digest_size, msg + i * inbuf_len, hash, inbuf_len);
    }

    Merkle().generate_tree(merkle.nodes, hash, merkle);
}

template<class Merkle, class Hash, typename Precision>
__global__ void single_proof_kernel(proof<Merkle::MerkleSize, Hash, Precision> this_proof, const Precision* leaf_to_prove, const uint32_t leaf_index, const tree<Merkle::MerkleSize, Hash, Precision> merkle)
{
    Merkle().generate_proof(this_proof, leaf_to_prove, leaf_index, merkle);
}


template<class Merkle, class Hash, typename Precision>
__global__ void generate_proof_kernel(proof<Merkle::MerkleSize, Hash, Precision>* proofs, const Precision* leaves_to_prove, const uint32_t* leaf_indices, const tree<Merkle::MerkleSize, Hash, Precision> merkle)
{
    constexpr size_t digest_size = proof<Merkle::MerkleSize, Hash, Precision>::digest_size;
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const Precision* leaf_to_prove = leaves_to_prove + thread_index * digest_size;
    const uint32_t leaf_index = leaf_indices[thread_index];
    Merkle().generate_proof(proofs[thread_index], leaf_to_prove, leaf_index, merkle); // Each thread generates a proof for the given leaf and corresponding index
}

template<class Merkle, class Hash, typename Precision>
__global__ void single_verify_kernel(const proof<Merkle::MerkleSize, Hash, Precision> this_proof, const Precision* verify_leaf, const uint32_t verify_index, const Precision* root, bool* verified) 
{
    Hash hash{};
    *verified = Merkle().verify_proof(this_proof, verify_leaf, verify_index, root, hash);
}


template<class Merkle, class Hash, typename Precision>
__global__ void verify_proof_kernel(const proof<Merkle::MerkleSize, Hash, Precision>* proofs, const Precision* verify_leaves, const uint32_t* verify_indices, const Precision* root, bool* verified) 
{
    Hash hash{};
    constexpr size_t digest_size = proof<Merkle::MerkleSize, Hash, Precision>::digest_size;
    uint32_t thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t verify_index = verify_indices[thread_index]; // This is the leaf's index in the tree
    const Precision* verify_leaf = verify_leaves + thread_index * digest_size;
    verified[thread_index] = Merkle().verify_proof(proofs[thread_index], verify_leaf, verify_index, root, hash);
}

template<class Merkle, class Hash, typename Precision>
void generate_tree(const std::vector<Precision>& msg, size_t inbuf_len, tree<Merkle::MerkleSize, Hash, Precision> merkle)
{
    Precision* d_msg;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size() * sizeof(Precision)); //Here msg.size has the number of messages and the length of each message is inbuf_len
    cudaMemcpy(d_msg, msg.data(), msg.size() * sizeof(Precision), cudaMemcpyHostToDevice);
    generate_merkle_tree_kernel<Merkle, Hash, Precision><<<1, Merkle::BlockDim>>>(merkle, d_msg, inbuf_len);
    cudaFree(d_msg); //Device side message is no longer needed
}

template<class Merkle, class Hash, typename Precision>
void generate_proof(const Precision* leaves_to_prove, const uint32_t* indices_to_prove, const size_t num_proofs, proof<Merkle::MerkleSize, Hash, Precision>* proofs, 
                    const tree<Merkle::MerkleSize, Hash, Precision> merkle)
{
    dim3 gridDim((num_proofs + 255) / 256);
    dim3 blockDim(256);
    if(num_proofs < 256) {
        gridDim.x = 1;
        blockDim.x = num_proofs;
    }
    generate_proof_kernel<Merkle, Hash, Precision><<<gridDim, blockDim>>>(proofs, leaves_to_prove, indices_to_prove, merkle);
}

template<class Merkle, class Hash, typename Precision>
void verify_proof(const Precision* verify_leaves, const uint32_t* verify_indices, const size_t num_proofs, const proof<Merkle::MerkleSize, Hash, Precision>* proofs, 
                  const Precision* root, bool* verified)
{
    dim3 gridDim((num_proofs + 255) / 256);
    dim3 blockDim(256);
    if(num_proofs < 256) {
        gridDim.x = 1;
        blockDim.x = num_proofs;
    }
    verify_proof_kernel<Merkle, Hash, Precision><<<gridDim, blockDim>>>(proofs, verify_leaves, verify_indices, root, verified);
}

//This is simply a helper kernel to set the indices for this example.
__global__ void index_set(uint32_t* indices, size_t num_indices) {
    for (size_t i = threadIdx.x; i < num_indices; i += blockDim.x) {
        indices[i] = i;
    }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {

    using tree_type = tree<MERKLE::MerkleSize, HASH, uint32_t>;
    using proof_type = proof<MERKLE::MerkleSize, HASH, uint32_t>;
    constexpr auto N = MERKLE::MerkleSize;

    constexpr size_t in_len = 64;
    std::vector<uint32_t> msg(in_len * N, 0); // Here we have 2048 messages, each of length in_len

    // Note that in_len is strictly less than BabyBearPrime, so we don't actually need to mod by BabyBearPrime.
    // This is just for illustration, as our Poseidon2 hash is built on the BabyBear field.
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < in_len; j++) {
            msg[i * in_len + j] = (i * j) % cupqc_common::BabyBearPrime; // Ensure that the message is within the field
        }
    }

    // To begin this sample, we will generate a Merkle tree from the messages.
    // The host function will handle device memory for messages, and generate the tree using a single kernel.
    tree_type merkle;
    merkle.allocate_tree(); // This will allocate the tree on the device, as well as the root.
    generate_tree<MERKLE, HASH, uint32_t>(msg, in_len, merkle);
    //Our tree is now generated. We can now generate proofs for selected leaves. 

    //As a first illustration of proof generation and verification, we will generate and validate a proof for a single leaf.

    proof_type first_proof;
    first_proof.allocate_proof(); // This will allocate the proof on the device.
    uint32_t* leaf_to_prove; 

    cudaMalloc(reinterpret_cast<void**>(&leaf_to_prove), proof_type::digest_size * sizeof(uint32_t));
    cudaMemcpy(leaf_to_prove, merkle.nodes, proof_type::digest_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    single_proof_kernel<MERKLE, HASH, uint32_t><<<1, 1>>>(first_proof, leaf_to_prove, 0, merkle);
    //Our proof is now generated. We can now verify this leaf.

    bool verified = false;
    bool* d_verified;
    cudaMalloc(reinterpret_cast<void**>(&d_verified), sizeof(bool));
    cudaMemcpy(d_verified, &verified, sizeof(bool), cudaMemcpyHostToDevice);
    single_verify_kernel<MERKLE, HASH, uint32_t><<<1, 1>>>(first_proof, leaf_to_prove, 0, merkle.root, d_verified);
    cudaMemcpy(&verified, d_verified, sizeof(bool), cudaMemcpyDeviceToHost);
    std::cout << "Leaf " << 0 << " is " << (verified ? "verified" : "not verified") << std::endl;

    cudaFree(leaf_to_prove);
    cudaFree(d_verified);
    first_proof.free_proof();

    // We can now generate proofs for 256 leaves, and verify them all.
    uint32_t* d_leaves_to_prove;
    uint32_t* d_indices_to_prove;
    cudaMalloc(reinterpret_cast<void**>(&d_leaves_to_prove), 256 * proof_type::digest_size * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_indices_to_prove), 256 * sizeof(uint32_t));
    cudaMemcpy(d_leaves_to_prove, merkle.nodes, 256 * proof_type::digest_size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
    
    index_set<<<1, 256>>>(d_indices_to_prove, 256);


    //Here we will use cudaMallocManaged to allocate the proofs. We will need to directly allocate the nodes and indicies here. 
    proof_type* proofs;
    cudaMallocManaged(reinterpret_cast<void**>(&proofs), 256 * sizeof(proof_type));
    for (size_t i = 0; i < 256; i++) {
        cudaMallocManaged(reinterpret_cast<void**>(&proofs[i].nodes), N * proof_type::digest_size * sizeof(uint32_t));
        cudaMallocManaged(reinterpret_cast<void**>(&proofs[i].indices), N * sizeof(uint32_t));
    }
    cudaDeviceSynchronize();

    std::cout << " ================================================= " << std::endl;
    std::cout << "Generating proofs for 256 leaves" << std::endl;
    generate_proof<MERKLE, HASH, uint32_t>(d_leaves_to_prove, d_indices_to_prove, 256, proofs, merkle);

    std::cout << " ================================================= " << std::endl;
    std::cout << "Verifying proofs for 256 leaves" << std::endl;
    //We can now verify all of the proofs.
    cudaMalloc(reinterpret_cast<void**>(&d_verified), 256 * sizeof(bool));
    cudaMemset(d_verified, 0, 256 * sizeof(bool));
    verify_proof<MERKLE, HASH, uint32_t>(d_leaves_to_prove, d_indices_to_prove, 256, proofs, merkle.root, d_verified);
    for (size_t i = 0; i < 256; i++) {
        cudaFree(proofs[i].nodes);
        cudaFree(proofs[i].indices);
    }
    cudaFree(proofs);
    cudaFree(d_leaves_to_prove);
    cudaFree(d_indices_to_prove);
    
    bool verify[256];
    cudaMemcpy(verify, d_verified, 256 * sizeof(bool), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < 256; i++) {
        std::cout << "Leaf " << i << " is " << (verify[i] ? "verified" : "not verified") << std::endl;
    }
    cudaFree(d_verified);

    //We can now free the tree.
    merkle.free_tree();

    return 0;
}