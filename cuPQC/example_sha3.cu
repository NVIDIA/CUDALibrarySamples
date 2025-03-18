#include <vector>
#include <iomanip>
#include <cuhash.hpp>
#include <stdio.h>

using namespace cupqc;

using SHA3_256_WARP = decltype(SHA3_256() + Warp());

__global__ void hash_sha3_kernel(uint8_t* digest, const uint8_t* msg, size_t inbuf_len)
{
    SHA3_256_WARP hash {};
    hash.reset();
    hash.update(msg, inbuf_len);
    hash.finalize();
    hash.digest(digest, SHA3_256_WARP::digest_size);
}

void hash_sha3(std::vector<uint8_t>& digest, std::vector<uint8_t>& msg)
{
    uint8_t* d_msg;
    uint8_t* d_digest;
    cudaMalloc(reinterpret_cast<void**>(&d_msg), msg.size());
    cudaMalloc(reinterpret_cast<void**>(&d_digest), digest.size());

    cudaMemcpy(d_msg, msg.data(), msg.size(), cudaMemcpyHostToDevice);

    hash_sha3_kernel<<<1, 32>>>(d_digest, d_msg, msg.size());

    cudaMemcpy(digest.data(), d_digest, digest.size(), cudaMemcpyDeviceToHost);

    cudaFree(d_msg);
    cudaFree(d_digest);
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    const char * msg_str = "The quick brown fox jumps over the lazy dog";
    std::vector<uint8_t> msg(reinterpret_cast<const uint8_t*>(msg_str), reinterpret_cast<const uint8_t*>(msg_str) + strlen(msg_str));
    std::vector<uint8_t> digest(SHA3_256::digest_size, 0);
    hash_sha3(digest, msg);
    printf("SHA3-256: ");
    for (uint8_t num : digest) {
        printf("%02x", num);
    }
    printf("\n");
}
