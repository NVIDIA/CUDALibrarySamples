#include <vector>
#include <cupqc.hpp>
using namespace cupqc;

using MLKEM512Key = decltype(ML_KEM_512()
                            + Function<function::Keygen>()
                            + Block()
                            + BlockDim<128>());  //Optional operator with default config

using MLKEM512Encaps = decltype(ML_KEM_512()
                               + Function<function::Encaps>()
                               + Block()
                               + BlockDim<128>());  //Optional operator with default config

using MLKEM512Decaps = decltype(ML_KEM_512()
                               + Function<function::Decaps>()
                               + Block()
                               + BlockDim<128>());  //Optional operator with default config

__global__ void keygen_kernel(uint8_t* public_keys, uint8_t* secret_keys, uint8_t* workspace, uint8_t* randombytes)
{
    __shared__ uint8_t smem_ptr[MLKEM512Key::shared_memory_size];
    int block = blockIdx.x;
    auto public_key = public_keys + block * MLKEM512Key::public_key_size;
    auto secret_key = secret_keys + block * MLKEM512Key::secret_key_size;
    auto entropy    = randombytes + block * MLKEM512Key::entropy_size;
    auto work       = workspace   + block * MLKEM512Key::workspace_size;

    MLKEM512Key().execute(public_key, secret_key, entropy, work, smem_ptr);
}

__global__ void encaps_kernel(uint8_t* ciphertexts, uint8_t* shared_secrets, const uint8_t* public_keys, uint8_t* workspace, uint8_t *randombytes)
{
    __shared__ uint8_t smem_ptr[MLKEM512Encaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM512Encaps::shared_secret_size;
    auto ciphertext    = ciphertexts + block * MLKEM512Encaps::ciphertext_size;
    auto public_key    = public_keys + block * MLKEM512Encaps::public_key_size;
    auto entropy    = randombytes + block * MLKEM512Encaps::entropy_size;
    auto work       = workspace   + block * MLKEM512Encaps::workspace_size;

    MLKEM512Encaps().execute(ciphertext, shared_secret, public_key, entropy, work, smem_ptr);
}

__global__ void decaps_kernel(uint8_t* shared_secrets, const uint8_t* ciphertexts,  const uint8_t* secret_keys, uint8_t* workspace)
{
    __shared__ uint8_t smem_ptr[MLKEM512Decaps::shared_memory_size];
    int block = blockIdx.x;
    auto shared_secret = shared_secrets + block * MLKEM512Decaps::shared_secret_size;
    auto ciphertext    = ciphertexts + block * MLKEM512Decaps::ciphertext_size;
    auto secret_key    = secret_keys + block * MLKEM512Decaps::secret_key_size;
    auto work          = workspace   + block * MLKEM512Decaps::workspace_size;

    MLKEM512Decaps().execute(shared_secret, ciphertext, secret_key, work, smem_ptr);
}

void ml_kem_keygen(std::vector<uint8_t>& public_keys, std::vector<uint8_t>& secret_keys, const unsigned int batch)
{
    /*
     * Set up for utilizing cuPQCDx ML-KEM Keygen.
     * Allocates device workspace for computing
     */
    auto length_public_key = MLKEM512Key::public_key_size;
    auto length_secret_key = MLKEM512Key::secret_key_size;

    auto workspace         = make_workspace<MLKEM512Key>(batch);
    auto randombytes       = get_entropy<MLKEM512Key>(batch);
    /*
     * Allocate device memory for public and secret keys
     */
    uint8_t* d_public_key = nullptr;
    uint8_t* d_secret_key = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch); //These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);

    keygen_kernel<<<batch, MLKEM512Key::BlockDim>>>(d_public_key, d_secret_key, workspace, randombytes);

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

void ml_kem_encaps(std::vector<uint8_t> &ciphertexts, std::vector<uint8_t> &sharedsecrets,
                      const std::vector<uint8_t> &public_keys, const unsigned int batch)
{
    /*
     * Set up for utilizing cuPQCDx ML-KEM Encaps.
     * Allocates device workspace for computing
     */
    auto length_ciphertext   = MLKEM512Encaps::ciphertext_size;
    auto length_public_key   = MLKEM512Encaps::public_key_size;
    auto length_sharedsecret = MLKEM512Encaps::shared_secret_size;

    auto workspace         = make_workspace<MLKEM512Encaps>(batch);
    auto randombytes       = get_entropy<MLKEM512Encaps>(batch);

    /*
     * Allocate device memory for public keys, ciphertexts and shared secrets
     */
    uint8_t* d_ciphertext   = nullptr;
    uint8_t* d_public_key   = nullptr;
    uint8_t* d_sharedsecret = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_ciphertext), length_ciphertext * batch); //These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void**>(&d_public_key), length_public_key * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_sharedsecret), length_sharedsecret * batch);

    /*
     * Transfer public_key from host memory
     */
    cudaMemcpy(d_public_key, public_keys.data(), length_public_key * batch, cudaMemcpyHostToDevice);

    encaps_kernel<<<batch, MLKEM512Encaps::BlockDim>>>(d_ciphertext, d_sharedsecret, d_public_key, workspace, randombytes);

    /*
     * Transfer ciphertext and shared secret from device memory
     */
    cudaMemcpy(ciphertexts.data(), d_ciphertext, length_ciphertext * batch, cudaMemcpyDeviceToHost);
    cudaMemcpy(sharedsecrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost);

    /*
     * Delete device memory associated with the cryptograpic process.
     */
    cudaFree(d_public_key);
    cudaFree(d_ciphertext);
    cudaFree(d_sharedsecret);
    destroy_workspace(workspace);
    release_entropy(randombytes);
}

void ml_kem_decaps(std::vector<uint8_t> &sharedsecrets, const std::vector<uint8_t> &ciphertexts, 
                      const std::vector<uint8_t> &secret_keys, const unsigned int batch)
{
    /*
     * Set up for utilizing cuPQCDx ML-KEM Decaps.
     * Allocates device workspace for computing
     */
    auto length_ciphertext   = MLKEM512Encaps::ciphertext_size;
    auto length_secret_key   = MLKEM512Encaps::secret_key_size;
    auto length_sharedsecret = MLKEM512Encaps::shared_secret_size;

    auto workspace         = make_workspace<MLKEM512Decaps>(batch);

    /*
     * Allocate device memory for public and secret keys
     */
    uint8_t* d_sharedsecret = nullptr;
    uint8_t* d_ciphertext   = nullptr;
    uint8_t* d_secret_key   = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_ciphertext), length_ciphertext * batch); //These are uint8_t so length and batch are in bytes
    cudaMalloc(reinterpret_cast<void**>(&d_secret_key), length_secret_key * batch);
    cudaMalloc(reinterpret_cast<void**>(&d_sharedsecret), length_sharedsecret * batch);

    /*
     * Transfer secret_key and ciphertext from host memory
     */
    cudaMemcpy(d_secret_key, secret_keys.data(), length_secret_key * batch, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ciphertext, ciphertexts.data(), length_ciphertext * batch, cudaMemcpyHostToDevice);

    decaps_kernel<<<batch, MLKEM512Decaps::BlockDim>>>(d_sharedsecret, d_ciphertext, d_secret_key, workspace);

    /*
     * Transfer sharedsecret from device to host mem
     */
    cudaMemcpy(sharedsecrets.data(), d_sharedsecret, length_sharedsecret * batch, cudaMemcpyDeviceToHost);

    /*
     * Delete device memory associated with the cryptograpic process.
     */
    cudaFree(d_ciphertext);
    cudaFree(d_sharedsecret);
    cudaFree(d_secret_key);
    destroy_workspace(workspace);
}


/*
 * Normally different actors will be performing the various functions associated with ML-KEM.
 * In this example, we perform all functions, Keygen, Encaps, and Decaps. 
 * You could optimize this example by reusing the device memory, and not transfering to host,
 * however, this is not the normal scenario. So, to better illustrate the use of the cuPQCDx API we wrote
 * host functions ml_kem_kygen, ml_kem_encaps, ml_kem_decaps that don't assume they will be called in 
 * conjunction with the others. 
 * 
 * In this example we produce 10 keys, ciphertexts and shared secrets, using batching.
 * 
 * In a typical scenerio:
 * Actor 1 (keygen + decaps)
 * Actor 2 (decaps)
 * 
 * or 
 * 
 * Actor 1 as a keygeneration service (keygen)
 * Actor 2 encaps using key from 1.
 * Actor 3 decaps using key from 1 and ciphertext from 2.
 */

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
    unsigned int batch = 10;

    /*
     * Generate Public and Secret Keys!
     */
    std::vector<uint8_t> public_keys(MLKEM512Key::public_key_size * batch);
    std::vector<uint8_t> secret_keys(MLKEM512Key::secret_key_size * batch);

    ml_kem_keygen(public_keys, secret_keys, batch);

    /*
     * Create shared secret and encapsulate it into ciphertext!
     */
    std::vector<uint8_t> ciphertexts(MLKEM512Encaps::ciphertext_size * batch);
    std::vector<uint8_t> sharedsecrets(MLKEM512Encaps::shared_secret_size * batch);
    ml_kem_encaps(ciphertexts, sharedsecrets, public_keys, batch);

    /*
     * Decapsulate ciphertext and store into shared secret!
     */
    ml_kem_decaps(sharedsecrets, ciphertexts, secret_keys, batch);
}
