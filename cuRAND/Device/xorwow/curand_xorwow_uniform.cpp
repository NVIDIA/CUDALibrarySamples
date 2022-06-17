/*
 * This program uses the host CURAND API to generate 100
 * pseudorandom floats.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>

#include "curand_utils.h"

using data_type = float;

void run_on_device(const int &n, const cudaStream_t &stream,
                   curandGenerator_t &gen, std::vector<data_type> &h_data) {

  data_type *d_data = nullptr;

  /* C data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_data),
                        sizeof(data_type) * h_data.size()));

  /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW));

  /* Set cuRAND to stream */
  CURAND_CHECK(curandSetStream(gen, stream));

  /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  /* Generate n floats on device */
  CURAND_CHECK(curandGenerateUniform(gen, d_data, h_data.size()));

  /* Copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(h_data.data(), d_data,
                             sizeof(data_type) * h_data.size(),
                             cudaMemcpyDeviceToHost, stream));

  /* Sync stream */
  CUDA_CHECK(cudaStreamSynchronize(stream));

  /* Cleanup */
  CUDA_CHECK(cudaFree(d_data));
}

void run_on_host(const int &n, const cudaStream_t &stream,
                 curandGenerator_t &gen, std::vector<data_type> &h_data) {

  /* Create pseudo-random number generator */
  CURAND_CHECK(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_XORWOW));

  /* Set cuRAND to stream */
  CURAND_CHECK(curandSetStream(gen, stream));

  /* Set seed */
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  /* Generate n floats on device */
  CURAND_CHECK(curandGenerateUniform(gen, h_data.data(), h_data.size()));
}

int main(int argc, char *argv[]) {

  cudaStream_t stream = NULL;
  curandGenerator_t gen = NULL;

  int n = 10;

  /* Create stream */
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  /* Allocate n floats on host */
  std::vector<data_type> h_data(n, 0);

  run_on_host(n, stream, gen, h_data);

  printf("Host\n");
  print_vector(h_data);
  printf("=====\n");

  run_on_device(n, stream, gen, h_data);

  printf("Device\n");
  print_vector(h_data);
  printf("=====\n");

  /* Cleanup */
  CURAND_CHECK(curandDestroyGenerator(gen));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  return EXIT_SUCCESS;
}
